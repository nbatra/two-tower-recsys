"""
Train ComiRec (Controllable Multi-Interest Recommendation) model.

ComiRec generates K interest embeddings per user via multi-head self-attention
over their interaction sequence, then trains with sampled softmax loss.

Outputs saved to models/comirec/:
  - comirec_model.pt (trained model weights)
  - item_embeddings.npy (21K x 128)
  - user_embeddings.npy (138K x K x 128)
  - faiss_index.bin (IndexFlatIP on item embeddings)
  - training_history.pkl
"""
import numpy as np
import pandas as pd
import pickle
import time
import os
import gc
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_DIR = Path('data/processed')
MODEL_DIR = Path('models/comirec')
MODEL_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Device: {device}')

# ============================================================
# 1. Load data and build user sequences
# ============================================================
with open(DATA_DIR / 'metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

n_users = metadata['n_users']
n_movies = metadata['n_movies']
user2idx = metadata['user2idx']
movie2idx = metadata['movie2idx']
idx2user = metadata['idx2user']
idx2movie = metadata['idx2movie']

print(f'Users: {n_users:,}, Movies: {n_movies:,}')

# Build per-user sequences sorted by timestamp (positives only)
train_df = pd.read_parquet(DATA_DIR / 'train_set.parquet')
positives = train_df[train_df['label'] == 1].copy()
positives = positives.sort_values(['user_idx', 'timestamp'])

print(f'Positive interactions: {len(positives):,}')

# Build sequences: list of movie_idx per user, time-sorted
user_sequences = {}
for user_idx, group in positives.groupby('user_idx'):
    seq = group['movie_idx'].values.tolist()
    if len(seq) >= 5:  # Minimum sequence length
        user_sequences[user_idx] = seq

del positives, train_df
gc.collect()

print(f'Users with sequences (>= 5 items): {len(user_sequences):,}')
seq_lengths = [len(v) for v in user_sequences.values()]
print(f'Sequence lengths: mean={np.mean(seq_lengths):.1f}, median={np.median(seq_lengths):.0f}, max={max(seq_lengths)}')

# ============================================================
# 2. Model Definition
# ============================================================
MAX_SEQ_LEN = 50  # Use last 50 interactions
EMBEDDING_DIM = 128
N_INTERESTS = 4  # Number of interest embeddings per user
N_HEADS = 4
DROPOUT = 0.1

class ComiRecModel(nn.Module):
    """
    ComiRec-SA: Multi-Interest extraction via Self-Attention.

    For each user, takes their interaction sequence and produces K interest
    embeddings via K attention heads. Each head learns to attend to a
    different subset of the history.
    """
    def __init__(self, n_items, embedding_dim, n_interests, max_seq_len, dropout=0.1):
        super().__init__()
        self.n_interests = n_interests
        self.embedding_dim = embedding_dim

        # Shared item embedding table
        self.item_embedding = nn.Embedding(n_items, embedding_dim, padding_idx=0)

        # Positional encoding for sequence order
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)

        # Multi-head attention for interest extraction
        # Each head produces one interest vector
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.Tanh(),
                nn.Linear(embedding_dim, 1)
            )
            for _ in range(n_interests)
        ])

        # Output projection per interest
        self.interest_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, embedding_dim)
            )
            for _ in range(n_interests)
        ])

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.item_embedding.weight[1:])
        nn.init.xavier_uniform_(self.position_embedding.weight)
        for head in self.attention_heads:
            for layer in head:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        for proj in self.interest_projections:
            for layer in proj:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def get_item_embeddings(self, item_ids):
        """Get L2-normalized item embeddings."""
        emb = self.item_embedding(item_ids)
        return F.normalize(emb, p=2, dim=-1)

    def extract_interests(self, seq_ids, seq_mask):
        """
        Extract K interest vectors from a user's interaction sequence.

        Args:
            seq_ids: (batch, seq_len) - item indices in the sequence
            seq_mask: (batch, seq_len) - 1 for real items, 0 for padding

        Returns:
            interests: (batch, K, embedding_dim) - K interest vectors per user
        """
        batch_size, seq_len = seq_ids.shape

        # Get item embeddings for sequence
        item_embs = self.item_embedding(seq_ids)  # (batch, seq_len, dim)

        # Add positional encoding
        positions = torch.arange(seq_len, device=seq_ids.device).unsqueeze(0)
        pos_embs = self.position_embedding(positions)
        item_embs = self.layer_norm(item_embs + pos_embs)
        item_embs = self.dropout(item_embs)

        # Mask for attention (padding positions get -inf)
        attn_mask = (1 - seq_mask.float()).unsqueeze(-1) * -1e9  # (batch, seq_len, 1)

        # Extract K interests via K attention heads
        interests = []
        for k in range(self.n_interests):
            # Compute attention weights
            attn_scores = self.attention_heads[k](item_embs)  # (batch, seq_len, 1)
            attn_scores = attn_scores + attn_mask
            attn_weights = F.softmax(attn_scores, dim=1)  # (batch, seq_len, 1)

            # Weighted sum of item embeddings
            interest_k = (attn_weights * item_embs).sum(dim=1)  # (batch, dim)

            # Project and normalize
            interest_k = self.interest_projections[k](interest_k)
            interest_k = F.normalize(interest_k, p=2, dim=-1)
            interests.append(interest_k)

        # Stack: (batch, K, dim)
        interests = torch.stack(interests, dim=1)
        return interests

    def forward(self, seq_ids, seq_mask, pos_ids, neg_ids):
        """
        Training forward pass with sampled softmax loss.

        For each positive item, we find the best-matching interest head
        and optimize that head to score the positive above negatives.
        """
        # Extract user interests: (batch, K, dim)
        interests = self.extract_interests(seq_ids, seq_mask)

        # Get target item embeddings
        pos_emb = self.get_item_embeddings(pos_ids)  # (batch, dim)
        neg_emb = self.get_item_embeddings(neg_ids)  # (batch, n_neg, dim)

        # Score positive with each interest head: (batch, K)
        pos_scores = torch.sum(interests * pos_emb.unsqueeze(1), dim=-1)

        # Best-matching interest for the positive item
        best_interest_scores, _ = pos_scores.max(dim=1)  # (batch,)

        # Score negatives with best interest for that positive
        # Use the full interest set and take max (multi-interest retrieval)
        # neg_emb: (batch, n_neg, dim), interests: (batch, K, dim)
        neg_scores = torch.einsum('bkd,bnd->bkn', interests, neg_emb)  # (batch, K, n_neg)
        neg_scores, _ = neg_scores.max(dim=1)  # (batch, n_neg) - best interest per negative

        # Sampled softmax loss
        # log_softmax(pos_score / (pos_score + sum(neg_scores)))
        scale = np.sqrt(EMBEDDING_DIM)
        pos_logits = best_interest_scores * scale  # (batch,)
        neg_logits = neg_scores * scale  # (batch, n_neg)

        # Concatenate and compute cross-entropy with first position as target
        all_logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)  # (batch, 1+n_neg)
        labels = torch.zeros(all_logits.shape[0], dtype=torch.long, device=all_logits.device)
        loss = F.cross_entropy(all_logits, labels)

        return loss, best_interest_scores.mean().item()


# ============================================================
# 3. Dataset and Training Loop
# ============================================================
N_NEGATIVES = 4
BATCH_SIZE = 4096
EPOCHS = 8
LR = 1e-3

class SequenceDataset:
    """Generates (sequence, positive_next_item, negative_items) triples."""

    def __init__(self, user_sequences, n_items, max_seq_len, n_negatives):
        self.n_items = n_items
        self.max_seq_len = max_seq_len
        self.n_negatives = n_negatives

        # Build training samples: for each user, create multiple (seq, target) pairs
        # by sliding a window over their history
        self.samples = []
        for user_idx, seq in user_sequences.items():
            # Use each position (except first 5) as a target
            for i in range(5, len(seq)):
                # Context: items before position i (last max_seq_len items)
                context = seq[max(0, i - max_seq_len):i]
                target = seq[i]
                self.samples.append((context, target, user_idx))

        print(f'Training samples: {len(self.samples):,}')

    def __len__(self):
        return len(self.samples)

    def get_batch(self, indices):
        """Build a padded batch from sample indices."""
        batch_seqs = []
        batch_masks = []
        batch_pos = []
        batch_neg = []

        for idx in indices:
            context, target, user_idx = self.samples[idx]

            # Pad sequence to max_seq_len
            seq_len = min(len(context), self.max_seq_len)
            padded_seq = [0] * (self.max_seq_len - seq_len) + context[-seq_len:]
            mask = [0] * (self.max_seq_len - seq_len) + [1] * seq_len

            # Sample negatives (uniform random, excluding target)
            negs = []
            while len(negs) < self.n_negatives:
                neg = np.random.randint(1, self.n_items)
                if neg != target:
                    negs.append(neg)

            batch_seqs.append(padded_seq)
            batch_masks.append(mask)
            batch_pos.append(target)
            batch_neg.append(negs)

        return (
            torch.tensor(batch_seqs, dtype=torch.long),
            torch.tensor(batch_masks, dtype=torch.long),
            torch.tensor(batch_pos, dtype=torch.long),
            torch.tensor(batch_neg, dtype=torch.long),
        )


print('\nBuilding training dataset...')
dataset = SequenceDataset(user_sequences, n_movies, MAX_SEQ_LEN, N_NEGATIVES)

# ============================================================
# 4. Training
# ============================================================
model = ComiRecModel(
    n_items=n_movies,
    embedding_dim=EMBEDDING_DIM,
    n_interests=N_INTERESTS,
    max_seq_len=MAX_SEQ_LEN,
    dropout=DROPOUT
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f'Model parameters: {total_params:,}')
print(f'Config: dim={EMBEDDING_DIM}, K={N_INTERESTS}, max_seq={MAX_SEQ_LEN}')
print(f'Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}, LR: {LR}')

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR, total_steps=EPOCHS * (len(dataset) // BATCH_SIZE + 1),
    pct_start=0.1, anneal_strategy='cos'
)

# Validation: for each val user, check if next item is retrievable
val_df = pd.read_parquet(DATA_DIR / 'val_set.parquet')
val_positives = val_df[val_df['label'] == 1].sort_values(['user_idx', 'timestamp'])
val_user_targets = {}
for user_idx, group in val_positives.groupby('user_idx'):
    if user_idx in user_sequences and len(group) >= 1:
        val_user_targets[user_idx] = group['movie_idx'].values.tolist()
del val_df, val_positives
gc.collect()
print(f'Validation users: {len(val_user_targets):,}')

def compute_val_recall(model, user_sequences, val_user_targets, k=200, max_users=2000):
    """Compute Recall@K on validation set using multi-interest retrieval."""
    model.eval()
    import faiss

    # Extract all item embeddings
    with torch.no_grad():
        all_item_ids = torch.arange(n_movies, device=device)
        item_embs = []
        for start in range(0, n_movies, 4096):
            end = min(start + 4096, n_movies)
            batch_ids = all_item_ids[start:end]
            emb = model.get_item_embeddings(batch_ids)
            item_embs.append(emb.cpu().numpy())
        item_embs = np.vstack(item_embs).astype(np.float32)

    # Build FAISS index
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(item_embs)

    recalls = []
    users_evaluated = 0

    val_users = list(val_user_targets.keys())[:max_users]

    with torch.no_grad():
        for user_idx in val_users:
            seq = user_sequences.get(user_idx)
            if seq is None or len(seq) < 5:
                continue

            targets = set(val_user_targets[user_idx])
            if len(targets) == 0:
                continue

            # Build sequence tensor
            seq_truncated = seq[-MAX_SEQ_LEN:]
            seq_len = len(seq_truncated)
            padded = [0] * (MAX_SEQ_LEN - seq_len) + seq_truncated
            mask = [0] * (MAX_SEQ_LEN - seq_len) + [1] * seq_len

            seq_tensor = torch.tensor([padded], dtype=torch.long, device=device)
            mask_tensor = torch.tensor([mask], dtype=torch.long, device=device)

            # Extract interests: (1, K, dim)
            interests = model.extract_interests(seq_tensor, mask_tensor)
            interests_np = interests.cpu().numpy().reshape(N_INTERESTS, EMBEDDING_DIM)

            # Multi-probe FAISS: search with each interest, merge results
            all_candidates = set()
            for interest_vec in interests_np:
                _, positions = index.search(interest_vec.reshape(1, -1), k // N_INTERESTS)
                all_candidates.update(positions[0].tolist())

            hits = len(all_candidates & targets)
            recalls.append(hits / len(targets))
            users_evaluated += 1

    model.train()
    return np.mean(recalls) if recalls else 0.0, users_evaluated


print(f'\nStarting ComiRec training...')
print('=' * 70)

history = []
best_recall = 0.0
patience_counter = 0
PATIENCE = 3

n_batches = len(dataset) // BATCH_SIZE + 1

for epoch in range(EPOCHS):
    model.train()
    t0 = time.time()

    # Shuffle samples
    perm = np.random.permutation(len(dataset))
    epoch_loss = 0.0
    n_steps = 0

    for batch_start in range(0, len(dataset), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(dataset))
        batch_indices = perm[batch_start:batch_end]

        seq_ids, seq_mask, pos_ids, neg_ids = dataset.get_batch(batch_indices)
        seq_ids = seq_ids.to(device)
        seq_mask = seq_mask.to(device)
        pos_ids = pos_ids.to(device)
        neg_ids = neg_ids.to(device)

        optimizer.zero_grad()
        loss, avg_score = model(seq_ids, seq_mask, pos_ids, neg_ids)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        n_steps += 1

        if n_steps % 500 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f'  Batch {n_steps}/{n_batches} - Loss: {epoch_loss/n_steps:.4f}, LR: {lr:.6f}', flush=True)

    avg_loss = epoch_loss / n_steps
    epoch_time = time.time() - t0

    # Validation
    print(f'  Computing validation Recall@200...', flush=True)
    val_recall, val_users = compute_val_recall(model, user_sequences, val_user_targets, k=200, max_users=2000)

    print(f'Epoch {epoch+1}/{EPOCHS}: Loss={avg_loss:.4f}, Val Recall@200={val_recall:.4f}, Time={epoch_time:.0f}s', flush=True)

    history.append({
        'epoch': epoch + 1,
        'train_loss': avg_loss,
        'val_recall_200': val_recall,
        'time': epoch_time
    })

    if val_recall > best_recall:
        best_recall = val_recall
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_DIR / 'comirec_model.pt')
        print(f'  --> Best model saved (Recall@200={val_recall:.4f})', flush=True)
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f'  Early stopping (no improvement for {PATIENCE} epochs)', flush=True)
            break

print(f'\nBest Val Recall@200: {best_recall:.4f}')

# ============================================================
# 5. Extract and save embeddings
# ============================================================
print('\nExtracting final embeddings...')
model.load_state_dict(torch.load(MODEL_DIR / 'comirec_model.pt', map_location=device, weights_only=True))
model.eval()

# Item embeddings
with torch.no_grad():
    all_item_ids = torch.arange(n_movies, device=device)
    item_embs = []
    for start in range(0, n_movies, 4096):
        end = min(start + 4096, n_movies)
        batch_ids = all_item_ids[start:end]
        emb = model.get_item_embeddings(batch_ids)
        item_embs.append(emb.cpu().numpy())
    item_embs = np.vstack(item_embs).astype(np.float32)

print(f'Item embeddings: {item_embs.shape}, norm={np.linalg.norm(item_embs[1:], axis=1).mean():.4f}')

# User embeddings (K interests per user)
user_embs = np.zeros((n_users, N_INTERESTS, EMBEDDING_DIM), dtype=np.float32)

with torch.no_grad():
    user_list = sorted(user_sequences.keys())
    for batch_start in range(0, len(user_list), 256):
        batch_end = min(batch_start + 256, len(user_list))
        batch_users = user_list[batch_start:batch_end]

        batch_seqs = []
        batch_masks = []
        for user_idx in batch_users:
            seq = user_sequences[user_idx][-MAX_SEQ_LEN:]
            seq_len = len(seq)
            padded = [0] * (MAX_SEQ_LEN - seq_len) + seq
            mask = [0] * (MAX_SEQ_LEN - seq_len) + [1] * seq_len
            batch_seqs.append(padded)
            batch_masks.append(mask)

        seq_tensor = torch.tensor(batch_seqs, dtype=torch.long, device=device)
        mask_tensor = torch.tensor(batch_masks, dtype=torch.long, device=device)

        interests = model.extract_interests(seq_tensor, mask_tensor)  # (batch, K, dim)
        interests_np = interests.cpu().numpy()

        for i, user_idx in enumerate(batch_users):
            user_embs[user_idx] = interests_np[i]

print(f'User embeddings: {user_embs.shape}')
print(f'Users with non-zero embeddings: {(user_embs.sum(axis=(1,2)) != 0).sum():,}')

# Save
np.save(MODEL_DIR / 'item_embeddings.npy', item_embs)
np.save(MODEL_DIR / 'user_embeddings.npy', user_embs)

# Build FAISS index
import faiss
index = faiss.IndexFlatIP(EMBEDDING_DIM)
index.add(item_embs)
faiss.write_index(index, str(MODEL_DIR / 'faiss_index.bin'))
print(f'FAISS index: {index.ntotal} vectors, dim={index.d}')

# Save training history
with open(MODEL_DIR / 'training_history.pkl', 'wb') as f:
    pickle.dump(history, f)

# ============================================================
# 6. Evaluate: multi-interest Recall@K
# ============================================================
print('\nComputing final Recall@K with multi-probe retrieval...')

val_df = pd.read_parquet(DATA_DIR / 'val_set.parquet')
val_pos = val_df[val_df['label'] == 1].groupby('user_idx')['movie_idx'].apply(set).to_dict()
del val_df

K_values = [10, 50, 100, 200, 500]
recalls = {k: [] for k in K_values}
sample_users = [u for u in list(val_pos.keys())[:3000] if u in user_sequences]

for user_idx in sample_users:
    targets = val_pos.get(user_idx, set())
    if len(targets) == 0:
        continue

    user_interests = user_embs[user_idx]  # (K, dim)

    # Multi-probe: search each interest separately, merge
    for k in K_values:
        all_candidates = set()
        per_interest_k = k // N_INTERESTS
        for interest_vec in user_interests:
            if np.linalg.norm(interest_vec) < 0.01:
                continue
            _, positions = index.search(interest_vec.reshape(1, -1).astype(np.float32), per_interest_k)
            all_candidates.update(positions[0].tolist())

        hits = len(all_candidates & targets)
        recalls[k].append(hits / len(targets))

print(f'\nComiRec Multi-Interest Recall@K ({len(sample_users)} users):')
for k in K_values:
    print(f'  Recall@{k:<4}: {np.mean(recalls[k]):.4f}')

# Compare with Two-Tower baseline
print(f'\nTwo-Tower Baseline Recall@K (from Notebook 03):')
print(f'  Recall@10  : 0.0161')
print(f'  Recall@50  : 0.0833')
print(f'  Recall@100 : 0.1595')
print(f'  Recall@200 : 0.2694')
print(f'  Recall@500 : 0.4704')

print('\nDone!')
