"""
Train SASRec (Self-Attentive Sequential Recommendation) model.

SASRec applies a causal (left-to-right) Transformer over the user's
interaction sequence. At each position t, it predicts what the user will
interact with at t+1 using only items 1..t as context.

Key differences from ComiRec:
- Single output embedding per user (not K heads)
- Deep self-attention (multi-layer Transformer) vs shallow per-head attention
- Causal masking: position t cannot attend to positions > t
- Binary cross-entropy loss per position (not sampled softmax)

Outputs saved to models/sasrec/:
  - sasrec_model.pt (trained model weights)
  - item_embeddings.npy (21K x 128)
  - user_embeddings.npy (138K x 128) -- last hidden state for each user
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
MODEL_DIR = Path('models/sasrec')
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

train_df = pd.read_parquet(DATA_DIR / 'train_set.parquet')
positives = train_df[train_df['label'] == 1].copy()
positives = positives.sort_values(['user_idx', 'timestamp'])

print(f'Positive interactions: {len(positives):,}')

user_sequences = {}
for user_idx, group in positives.groupby('user_idx'):
    seq = group['movie_idx'].values.tolist()
    if len(seq) >= 5:
        user_sequences[user_idx] = seq

del positives, train_df
gc.collect()

print(f'Users with sequences (>= 5 items): {len(user_sequences):,}')
seq_lengths = [len(v) for v in user_sequences.values()]
print(f'Sequence lengths: mean={np.mean(seq_lengths):.1f}, median={np.median(seq_lengths):.0f}, max={max(seq_lengths)}')

# ============================================================
# 2. Model Definition
# ============================================================
MAX_SEQ_LEN = 50
EMBEDDING_DIM = 128
N_HEADS = 2
N_LAYERS = 2
DROPOUT = 0.2
N_AUGMENTED_SEQS = 5  # Number of subsequences per user per epoch


class SASRecModel(nn.Module):
    """
    Self-Attentive Sequential Recommendation.

    Uses a causal Transformer encoder over the item sequence.
    The output at position t is used to predict the item at position t+1.
    """
    def __init__(self, n_items, embedding_dim, max_seq_len, n_heads, n_layers, dropout):
        super().__init__()
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

        self.item_embedding = nn.Embedding(n_items, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        self.emb_dropout = nn.Dropout(dropout)
        self.emb_norm = nn.LayerNorm(embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_norm = nn.LayerNorm(embedding_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.item_embedding.weight[1:])
        nn.init.xavier_uniform_(self.position_embedding.weight)

    def get_item_embeddings(self, item_ids):
        emb = self.item_embedding(item_ids)
        return F.normalize(emb, p=2, dim=-1)

    def encode_sequence(self, seq_ids, seq_mask):
        """
        Encode a sequence with causal self-attention.

        Args:
            seq_ids: (batch, seq_len) item indices
            seq_mask: (batch, seq_len) 1=real, 0=padding

        Returns:
            hidden: (batch, seq_len, dim) contextualized representations
        """
        batch_size, seq_len = seq_ids.shape

        # Item + positional embeddings
        item_embs = self.item_embedding(seq_ids)
        positions = torch.arange(seq_len, device=seq_ids.device).unsqueeze(0)
        pos_embs = self.position_embedding(positions)
        hidden = self.emb_norm(item_embs + pos_embs)
        hidden = self.emb_dropout(hidden)

        # Causal mask: position t can only attend to positions <= t
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=seq_ids.device), diagonal=1
        ).bool()

        # Padding mask: True = ignore this position
        padding_mask = (seq_mask == 0)

        # Transformer encoder
        hidden = self.transformer(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )
        hidden = self.output_norm(hidden)

        return hidden

    def forward(self, seq_ids, seq_mask, pos_ids, neg_ids):
        """
        Training forward: predict next item at each position.

        Args:
            seq_ids: (batch, seq_len) input sequence (items 1..T-1)
            seq_mask: (batch, seq_len) mask
            pos_ids: (batch, seq_len) positive targets (items 2..T)
            neg_ids: (batch, seq_len) negative samples

        Returns:
            loss, accuracy
        """
        hidden = self.encode_sequence(seq_ids, seq_mask)  # (batch, seq_len, dim)

        # Get embeddings for positive and negative targets
        pos_emb = self.item_embedding(pos_ids)  # (batch, seq_len, dim)
        neg_emb = self.item_embedding(neg_ids)  # (batch, seq_len, dim)

        # Dot-product scores
        pos_scores = (hidden * pos_emb).sum(dim=-1)  # (batch, seq_len)
        neg_scores = (hidden * neg_emb).sum(dim=-1)  # (batch, seq_len)

        # Binary cross-entropy loss (only on non-padding positions)
        mask = seq_mask.float()
        pos_loss = -F.logsigmoid(pos_scores) * mask
        neg_loss = -F.logsigmoid(-neg_scores) * mask

        loss = (pos_loss + neg_loss).sum() / mask.sum()

        # Accuracy for monitoring
        with torch.no_grad():
            correct = ((pos_scores > neg_scores) * mask).sum()
            accuracy = correct / mask.sum()

        return loss, accuracy.item()


# ============================================================
# 3. Dataset
# ============================================================
BATCH_SIZE = 4096
EPOCHS = 12
LR = 1e-3


class SASRecDataset:
    """
    Generates (input_seq, positive_next, negative_next) for SASRec training.

    Uses sliding-window augmentation: for users with sequences longer than
    max_seq_len, we generate multiple subsequences at different offsets.
    This dramatically increases effective training data.
    """

    def __init__(self, user_sequences, n_items, max_seq_len, n_augmented=5):
        self.n_items = n_items
        self.max_seq_len = max_seq_len
        self.sequences = user_sequences

        # Build training samples: (user_idx, end_position) pairs
        self.samples = []
        for user_idx, seq in user_sequences.items():
            seq_len = len(seq)
            if seq_len <= max_seq_len + 1:
                # Short sequence: one sample using full sequence
                self.samples.append((user_idx, seq_len))
            else:
                # Long sequence: sample n_augmented random end positions
                for _ in range(n_augmented):
                    # End position ranges from max_seq_len+1 to seq_len
                    end = np.random.randint(max_seq_len + 1, seq_len + 1)
                    self.samples.append((user_idx, end))

        print(f'Dataset: {len(self.samples):,} samples from {len(user_sequences):,} users')

    def __len__(self):
        return len(self.samples)

    def get_batch(self, indices):
        """Build training batch from sample indices."""
        batch_input = []
        batch_mask = []
        batch_pos = []
        batch_neg = []

        for idx in indices:
            user_idx, end_pos = self.samples[idx]
            seq = self.sequences[user_idx][:end_pos]

            # Use last max_seq_len+1 items: input is [:-1], target is [1:]
            seq_truncated = seq[-(self.max_seq_len + 1):]
            input_seq = seq_truncated[:-1]
            target_seq = seq_truncated[1:]

            seq_len = len(input_seq)

            # Pad to max_seq_len (left-pad)
            pad_len = self.max_seq_len - seq_len
            padded_input = [0] * pad_len + input_seq
            padded_target = [0] * pad_len + target_seq
            mask = [0] * pad_len + [1] * seq_len

            # Sample negatives for each position
            neg_items = []
            item_set = set(seq)
            for _ in range(self.max_seq_len):
                neg = np.random.randint(1, self.n_items)
                while neg in item_set:
                    neg = np.random.randint(1, self.n_items)
                neg_items.append(neg)

            batch_input.append(padded_input)
            batch_mask.append(mask)
            batch_pos.append(padded_target)
            batch_neg.append(neg_items)

        return (
            torch.tensor(batch_input, dtype=torch.long),
            torch.tensor(batch_mask, dtype=torch.long),
            torch.tensor(batch_pos, dtype=torch.long),
            torch.tensor(batch_neg, dtype=torch.long),
        )


print('\nBuilding dataset...')
dataset = SASRecDataset(user_sequences, n_movies, MAX_SEQ_LEN, n_augmented=N_AUGMENTED_SEQS)

# ============================================================
# 4. Training
# ============================================================
model = SASRecModel(
    n_items=n_movies,
    embedding_dim=EMBEDDING_DIM,
    max_seq_len=MAX_SEQ_LEN,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    dropout=DROPOUT,
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f'Model parameters: {total_params:,}')
print(f'Config: dim={EMBEDDING_DIM}, heads={N_HEADS}, layers={N_LAYERS}, max_seq={MAX_SEQ_LEN}')
print(f'Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}, LR: {LR}')

optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), weight_decay=1e-5)
n_batches = len(dataset) // BATCH_SIZE + 1
total_steps = EPOCHS * n_batches

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR, total_steps=total_steps,
    pct_start=0.05, anneal_strategy='cos'
)

# Validation setup
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
    """Compute Recall@K using SASRec's last-position output as user embedding."""
    model.eval()
    import faiss

    with torch.no_grad():
        all_item_ids = torch.arange(n_movies, device=device)
        item_embs = []
        for start in range(0, n_movies, 4096):
            end = min(start + 4096, n_movies)
            batch_ids = all_item_ids[start:end]
            emb = model.get_item_embeddings(batch_ids)
            item_embs.append(emb.cpu().numpy())
        item_embs = np.vstack(item_embs).astype(np.float32)

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(item_embs)

    recalls = []
    val_users = list(val_user_targets.keys())[:max_users]

    with torch.no_grad():
        # Process in batches for efficiency
        batch_size = 256
        for batch_start in range(0, len(val_users), batch_size):
            batch_end = min(batch_start + batch_size, len(val_users))
            batch_users = val_users[batch_start:batch_end]

            batch_seqs = []
            batch_masks = []
            batch_targets_list = []

            for user_idx in batch_users:
                seq = user_sequences.get(user_idx)
                if seq is None or len(seq) < 5:
                    continue

                targets = set(val_user_targets[user_idx])
                if len(targets) == 0:
                    continue

                seq_truncated = seq[-MAX_SEQ_LEN:]
                seq_len = len(seq_truncated)
                padded = [0] * (MAX_SEQ_LEN - seq_len) + seq_truncated
                mask = [0] * (MAX_SEQ_LEN - seq_len) + [1] * seq_len

                batch_seqs.append(padded)
                batch_masks.append(mask)
                batch_targets_list.append(targets)

            if not batch_seqs:
                continue

            seq_tensor = torch.tensor(batch_seqs, dtype=torch.long, device=device)
            mask_tensor = torch.tensor(batch_masks, dtype=torch.long, device=device)

            hidden = model.encode_sequence(seq_tensor, mask_tensor)

            # Use the last non-padding position as user embedding
            for i in range(len(batch_seqs)):
                last_pos = sum(batch_masks[i]) - 1
                user_emb = hidden[i, last_pos].cpu().numpy()
                user_emb = user_emb / (np.linalg.norm(user_emb) + 1e-8)
                user_emb = user_emb.reshape(1, -1).astype(np.float32)

                _, positions = index.search(user_emb, k)
                candidates = set(positions[0].tolist())
                hits = len(candidates & batch_targets_list[i])
                recalls.append(hits / len(batch_targets_list[i]))

    model.train()
    return np.mean(recalls) if recalls else 0.0, len(recalls)


print(f'\nStarting SASRec training...')
print('=' * 70)

history = []
best_recall = 0.0
patience_counter = 0
PATIENCE = 3

for epoch in range(EPOCHS):
    model.train()
    t0 = time.time()

    # Shuffle samples
    perm = np.random.permutation(len(dataset))

    epoch_loss = 0.0
    epoch_acc = 0.0
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
        loss, acc = model(seq_ids, seq_mask, pos_ids, neg_ids)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        epoch_acc += acc
        n_steps += 1

        if n_steps % 50 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f'  Batch {n_steps}/{n_batches} - Loss: {epoch_loss/n_steps:.4f}, '
                  f'Acc: {epoch_acc/n_steps:.4f}, LR: {lr:.6f}', flush=True)

    avg_loss = epoch_loss / n_steps
    avg_acc = epoch_acc / n_steps
    epoch_time = time.time() - t0

    # Validation
    print(f'  Computing validation Recall@200...', flush=True)
    val_recall, val_users_count = compute_val_recall(
        model, user_sequences, val_user_targets, k=200, max_users=2000
    )

    print(f'Epoch {epoch+1}/{EPOCHS}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}, '
          f'Val Recall@200={val_recall:.4f}, Time={epoch_time:.0f}s', flush=True)

    history.append({
        'epoch': epoch + 1,
        'train_loss': avg_loss,
        'train_acc': avg_acc,
        'val_recall_200': val_recall,
        'time': epoch_time
    })

    if val_recall > best_recall:
        best_recall = val_recall
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_DIR / 'sasrec_model.pt')
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
model.load_state_dict(torch.load(MODEL_DIR / 'sasrec_model.pt', map_location=device, weights_only=True))
model.eval()

# Item embeddings (L2-normalized)
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

# User embeddings: encode each user's sequence, take last hidden state
user_embs = np.zeros((n_users, EMBEDDING_DIM), dtype=np.float32)

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

        hidden = model.encode_sequence(seq_tensor, mask_tensor)

        for i, user_idx in enumerate(batch_users):
            last_pos = sum(batch_masks[i]) - 1
            emb = hidden[i, last_pos].cpu().numpy()
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            user_embs[user_idx] = emb

print(f'User embeddings: {user_embs.shape}')
print(f'Users with non-zero embeddings: {(np.linalg.norm(user_embs, axis=1) > 0.01).sum():,}')

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
# 6. Evaluate: Recall@K
# ============================================================
print('\nComputing final Recall@K...')

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

    user_vec = user_embs[user_idx].reshape(1, -1).astype(np.float32)
    if np.linalg.norm(user_vec) < 0.01:
        continue

    for k in K_values:
        _, positions = index.search(user_vec, k)
        candidates = set(positions[0].tolist())
        candidates.discard(-1)
        hits = len(candidates & targets)
        recalls[k].append(hits / len(targets))

print(f'\nSASRec Recall@K ({len(sample_users)} users):')
for k in K_values:
    print(f'  Recall@{k:<4}: {np.mean(recalls[k]):.4f}')

print(f'\nBaselines:')
print(f'  Two-Tower  Recall@200: 0.2694')
print(f'  ComiRec    Recall@200: 0.2748')

print('\nDone!')
