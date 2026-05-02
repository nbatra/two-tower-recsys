"""
Approach C: Hard negative mining - fewer but smarter training pairs.

Key idea:
- Keep all 12M positives
- Mine hard negatives using the baseline model's FAISS index:
  items ranked 50-500 for each user (too close for comfort, but not the user's actual positives)
- Also include popularity-weighted negatives (popular items the user skipped)
- Result: ~36M pairs where every negative is informative
- This is half the data of baseline (69M) but much harder for the model
"""
import numpy as np
import pandas as pd
import pickle
import time
import gc
import os
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import faiss

DATA_DIR = Path('data/processed')
MODEL_DIR = Path('models')

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

# ============================================================
# 1. Load data
# ============================================================
with open(DATA_DIR / 'metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

n_users = metadata['n_users']
n_movies = metadata['n_movies']
user_feature_dim = metadata['user_feature_dim']
item_feature_dim = metadata['item_feature_dim']
user2idx = metadata['user2idx']
movie2idx = metadata['movie2idx']
idx2user = metadata['idx2user']
idx2movie = metadata['idx2movie']

user_features_df = pd.read_parquet(DATA_DIR / 'user_features.parquet')
item_features_df = pd.read_parquet(DATA_DIR / 'item_features.parquet')

user_feature_matrix = np.zeros((n_users, user_feature_dim), dtype=np.float32)
item_feature_matrix = np.zeros((n_movies, item_feature_dim), dtype=np.float32)

for user_id, idx in user2idx.items():
    if user_id in user_features_df.index:
        user_feature_matrix[idx] = user_features_df.loc[user_id].values
for movie_id, idx in movie2idx.items():
    if movie_id in item_features_df.index:
        item_feature_matrix[idx] = item_features_df.loc[movie_id].values

del user_features_df, item_features_df
gc.collect()

user_feature_tensor = torch.from_numpy(user_feature_matrix).to(device)
item_feature_tensor = torch.from_numpy(item_feature_matrix).to(device)

# Load positive pairs
train_pos_user_idx = np.load(DATA_DIR / 'tt_pos_user_idx.npy')
train_pos_movie_idx = np.load(DATA_DIR / 'tt_pos_movie_idx.npy')
n_positives = len(train_pos_user_idx)

# Validation
val_user_idx = np.load(DATA_DIR / 'val_user_idx.npy')
val_movie_idx = np.load(DATA_DIR / 'val_movie_idx.npy')
val_labels = np.load(DATA_DIR / 'val_labels.npy')
n_val = len(val_labels)

print(f'Users: {n_users:,}, Movies: {n_movies:,}', flush=True)
print(f'Positives: {n_positives:,}', flush=True)
print(f'Val: {n_val:,}', flush=True)

# ============================================================
# 2. Mine hard negatives using baseline FAISS index
# ============================================================
print('\nMining hard negatives from baseline model...', flush=True)
t0 = time.time()

user_embeddings_baseline = np.load(MODEL_DIR / 'user_embeddings.npy')
item_embeddings_baseline = np.load(MODEL_DIR / 'item_embeddings.npy')

index_baseline = faiss.IndexFlatIP(64)
index_baseline.add(item_embeddings_baseline)

# Build set of positives per user for exclusion
user_positive_sets = {}
for i in range(n_positives):
    uid = train_pos_user_idx[i]
    mid = train_pos_movie_idx[i]
    if uid not in user_positive_sets:
        user_positive_sets[uid] = set()
    user_positive_sets[uid].add(mid)

# Item popularity (for popularity-weighted negatives)
item_counts = np.zeros(n_movies, dtype=np.int32)
for mid in train_pos_movie_idx:
    item_counts[mid] += 1
item_popularity = item_counts / item_counts.sum()

# For each unique user, retrieve items ranked 50-500 as hard negative candidates
unique_users = np.array(list(user_positive_sets.keys()), dtype=np.int32)
n_unique_users = len(unique_users)
print(f'Unique users with positives: {n_unique_users:,}', flush=True)

# We'll generate 2 hard negatives per positive
NEGATIVES_PER_POSITIVE = 2
hard_neg_user = np.zeros(n_positives * NEGATIVES_PER_POSITIVE, dtype=np.int32)
hard_neg_movie = np.zeros(n_positives * NEGATIVES_PER_POSITIVE, dtype=np.int32)

# Process users in batches for FAISS search
SEARCH_BATCH = 1024
SEARCH_K = 500
HARD_START = 30  # Skip top-30 (too close, might be near-positives)

rng = np.random.default_rng(42)
neg_idx = 0

for batch_start in range(0, n_unique_users, SEARCH_BATCH):
    batch_end = min(batch_start + SEARCH_BATCH, n_unique_users)
    batch_users = unique_users[batch_start:batch_end]
    batch_embs = user_embeddings_baseline[batch_users]

    _, positions = index_baseline.search(batch_embs, SEARCH_K)

    for i, uid in enumerate(batch_users):
        pos_set = user_positive_sets[uid]
        # Candidates: items ranked HARD_START to SEARCH_K that are NOT positives
        candidates = []
        for pos in positions[i, HARD_START:]:
            if pos != 0 and pos not in pos_set:
                candidates.append(pos)

        if len(candidates) == 0:
            continue

        # How many negatives do we need for this user?
        n_user_positives = len(pos_set)
        n_negs_needed = n_user_positives * NEGATIVES_PER_POSITIVE

        # Sample from hard candidates (with replacement if needed)
        sampled = rng.choice(candidates, size=min(n_negs_needed, len(candidates) * 3),
                            replace=True if n_negs_needed > len(candidates) else False)

        # Assign to the negative arrays
        user_pos_indices = np.where(train_pos_user_idx == uid)[0]
        for j, pos_idx in enumerate(user_pos_indices):
            for k in range(NEGATIVES_PER_POSITIVE):
                sample_idx = (j * NEGATIVES_PER_POSITIVE + k) % len(sampled)
                if neg_idx < len(hard_neg_user):
                    hard_neg_user[neg_idx] = uid
                    hard_neg_movie[neg_idx] = sampled[sample_idx]
                    neg_idx += 1

    if (batch_start // SEARCH_BATCH) % 20 == 0:
        print(f'  Processed {batch_end:,}/{n_unique_users:,} users, {neg_idx:,} negatives mined', flush=True)

# Trim to actual size
hard_neg_user = hard_neg_user[:neg_idx]
hard_neg_movie = hard_neg_movie[:neg_idx]
print(f'\nHard negatives mined: {neg_idx:,} in {time.time()-t0:.0f}s', flush=True)

del user_embeddings_baseline, item_embeddings_baseline, index_baseline
gc.collect()

# ============================================================
# 3. Combine positives + hard negatives into training set
# ============================================================
train_user_idx = np.concatenate([train_pos_user_idx, hard_neg_user])
train_movie_idx = np.concatenate([train_pos_movie_idx, hard_neg_movie])
train_labels = np.concatenate([
    np.ones(n_positives, dtype=np.float32),
    np.zeros(len(hard_neg_user), dtype=np.float32)
])

n_train = len(train_labels)
print(f'Total training pairs: {n_train:,} ({n_positives:,} pos + {len(hard_neg_user):,} hard neg)', flush=True)
print(f'Positive ratio: {train_labels.mean():.3f}', flush=True)

del hard_neg_user, hard_neg_movie
gc.collect()

# ============================================================
# 4. Model (same architecture as baseline)
# ============================================================
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
OUTPUT_DIM = 64
DROPOUT = 0.2

class UserTower(nn.Module):
    def __init__(self, n_users, user_feature_dim, embedding_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.user_embedding.weight[1:])
        input_dim = embedding_dim + user_feature_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
    def forward(self, user_idx, user_features):
        emb = self.user_embedding(user_idx)
        x = torch.cat([emb, user_features], dim=1)
        x = self.mlp(x)
        return F.normalize(x, p=2, dim=1)

class ItemTower(nn.Module):
    def __init__(self, n_movies, item_feature_dim, embedding_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.item_embedding = nn.Embedding(n_movies, embedding_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.item_embedding.weight[1:])
        input_dim = embedding_dim + item_feature_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
    def forward(self, movie_idx, item_features):
        emb = self.item_embedding(movie_idx)
        x = torch.cat([emb, item_features], dim=1)
        x = self.mlp(x)
        return F.normalize(x, p=2, dim=1)

class TwoTowerModel(nn.Module):
    def __init__(self, n_users, n_movies, user_feature_dim, item_feature_dim,
                 embedding_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.user_tower = UserTower(n_users, user_feature_dim, embedding_dim, hidden_dim, output_dim, dropout)
        self.item_tower = ItemTower(n_movies, item_feature_dim, embedding_dim, hidden_dim, output_dim, dropout)

    def forward(self, user_idx, user_features, movie_idx, item_features):
        user_emb = self.user_tower(user_idx, user_features)
        item_emb = self.item_tower(movie_idx, item_features)
        score = (user_emb * item_emb).sum(dim=1) * np.sqrt(OUTPUT_DIM)
        return score

    def get_user_embeddings(self, user_idx, user_features):
        return self.user_tower(user_idx, user_features)

    def get_item_embeddings(self, movie_idx, item_features):
        return self.item_tower(movie_idx, item_features)

model = TwoTowerModel(n_users, n_movies, user_feature_dim, item_feature_dim,
                      EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, DROPOUT).to(device)
print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}', flush=True)

# ============================================================
# 5. Training
# ============================================================
BATCH_SIZE = 8192
EPOCHS = 8
LOG_EVERY = 1000

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, steps_per_epoch=n_train // BATCH_SIZE, epochs=EPOCHS
)
criterion = nn.BCEWithLogitsLoss()

n_batches_train = n_train // BATCH_SIZE
perm = np.arange(n_train, dtype=np.int32)

print(f'\nBatch size: {BATCH_SIZE:,}, Batches/epoch: {n_batches_train:,}', flush=True)
print(f'Total training pairs: {n_train:,} (vs 69M in baseline)', flush=True)

best_auc = 0.0
best_epoch = 0
history = []

print('\nStarting hard-negative training...', flush=True)
print('=' * 70, flush=True)

for epoch in range(EPOCHS):
    epoch_start = time.time()
    print(f'\nEpoch {epoch+1}/{EPOCHS}', flush=True)

    np.random.shuffle(perm)
    model.train()
    total_loss = 0.0
    n_done = 0

    for batch_idx in range(n_batches_train):
        start = batch_idx * BATCH_SIZE
        end = start + BATCH_SIZE
        indices = perm[start:end]

        u_idx = torch.from_numpy(train_user_idx[indices].astype(np.int64)).to(device)
        m_idx = torch.from_numpy(train_movie_idx[indices].astype(np.int64)).to(device)
        labs = torch.from_numpy(train_labels[indices]).to(device)

        user_feats = user_feature_tensor[u_idx]
        item_feats = item_feature_tensor[m_idx]

        scores = model(u_idx, user_feats, m_idx, item_feats)
        loss = criterion(scores, labs)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_done += 1

        if (batch_idx + 1) % LOG_EVERY == 0:
            avg_loss = total_loss / n_done
            lr = optimizer.param_groups[0]['lr']
            print(f'  Batch {batch_idx+1:,}/{n_batches_train:,} - Loss: {avg_loss:.4f}, LR: {lr:.6f}', flush=True)

    train_loss = total_loss / n_done

    # Validate
    model.eval()
    all_scores = []
    all_labels_list = []
    val_batch_size = BATCH_SIZE * 2
    n_batches_val = (n_val + val_batch_size - 1) // val_batch_size

    with torch.no_grad():
        for batch_idx in range(n_batches_val):
            start = batch_idx * val_batch_size
            end = min(start + val_batch_size, n_val)

            u_idx = torch.from_numpy(val_user_idx[start:end].astype(np.int64)).to(device)
            m_idx = torch.from_numpy(val_movie_idx[start:end].astype(np.int64)).to(device)

            user_feats = user_feature_tensor[u_idx]
            item_feats = item_feature_tensor[m_idx]

            scores = model(u_idx, user_feats, m_idx, item_feats)

            all_scores.append(scores.cpu().numpy())
            all_labels_list.append(val_labels[start:end])

    all_scores_arr = np.concatenate(all_scores)
    all_labels_arr = np.concatenate(all_labels_list)
    val_auc = roc_auc_score(all_labels_arr, all_scores_arr)

    epoch_time = time.time() - epoch_start
    print(f'  Train Loss: {train_loss:.4f}, Val AUC: {val_auc:.4f}, Time: {epoch_time:.0f}s', flush=True)

    history.append({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'val_auc': val_auc,
        'time': epoch_time
    })

    if val_auc > best_auc:
        best_auc = val_auc
        best_epoch = epoch + 1
        torch.save(model.state_dict(), MODEL_DIR / 'two_tower_hard_neg.pt')
        print(f'  --> Best model saved (AUC={best_auc:.4f})', flush=True)

print(f'\nBest Val AUC: {best_auc:.4f} at epoch {best_epoch}', flush=True)

with open(MODEL_DIR / 'history_hard_neg.pkl', 'wb') as f:
    pickle.dump(history, f)

# ============================================================
# 6. Extract embeddings and build FAISS
# ============================================================
model.load_state_dict(torch.load(MODEL_DIR / 'two_tower_hard_neg.pt', map_location=device))
model.eval()

@torch.no_grad()
def extract_all_embeddings(model, feature_tensor, n_entities, tower='item', batch_size=2048):
    embeddings = np.zeros((n_entities, OUTPUT_DIM), dtype=np.float32)
    for start in range(1, n_entities, batch_size):
        end = min(start + batch_size, n_entities)
        idx = torch.arange(start, end, device=device)
        feats = feature_tensor[idx]
        if tower == 'item':
            emb = model.get_item_embeddings(idx, feats)
        else:
            emb = model.get_user_embeddings(idx, feats)
        embeddings[start:end] = emb.cpu().numpy()
    return embeddings

print('\nExtracting embeddings...', flush=True)
item_embeddings = extract_all_embeddings(model, item_feature_tensor, n_movies, tower='item')
user_embeddings = extract_all_embeddings(model, user_feature_tensor, n_users, tower='user')
print(f'  Item norms: {np.linalg.norm(item_embeddings[1:], axis=1).mean():.4f}', flush=True)
print(f'  User norms: {np.linalg.norm(user_embeddings[1:], axis=1).mean():.4f}', flush=True)

np.save(MODEL_DIR / 'item_embeddings_hard_neg.npy', item_embeddings)
np.save(MODEL_DIR / 'user_embeddings_hard_neg.npy', user_embeddings)

index = faiss.IndexFlatIP(OUTPUT_DIM)
index.add(item_embeddings)
faiss.write_index(index, str(MODEL_DIR / 'faiss_index_hard_neg.bin'))

# ============================================================
# 7. Recall@K
# ============================================================
val_df = pd.read_parquet(DATA_DIR / 'val_set.parquet')
val_positives = val_df[val_df['label'] == 1].groupby('user_idx')['movie_idx'].apply(set).to_dict()
sample_users = list(val_positives.keys())[:5000]

K_values = [10, 50, 100, 200, 500]
recalls = {k: [] for k in K_values}
for i in range(0, len(sample_users), 256):
    batch_users = sample_users[i:i+256]
    batch_embs = user_embeddings[batch_users]
    _, batch_positions = index.search(batch_embs, 500)
    for j, user_idx in enumerate(batch_users):
        positives = val_positives.get(user_idx, set())
        if len(positives) == 0:
            continue
        for k in K_values:
            top_k = set(batch_positions[j][:k].tolist())
            recalls[k].append(len(top_k & positives) / len(positives))

print(f'\nHard Negative Recall@K:', flush=True)
for k in K_values:
    print(f'  Recall@{k:<4}: {np.mean(recalls[k]):.4f}', flush=True)

# Item similarity check
movies_df = pd.read_csv('data/ml-25m/movies.csv')
movie_titles = dict(zip(movies_df['movieId'], movies_df['title']))

for test_mid in [1, 296, 356]:
    test_midx = movie2idx.get(test_mid)
    if test_midx is None:
        continue
    query = item_embeddings[test_midx:test_midx+1]
    s, p = index.search(query, 6)
    print(f'\nSimilar to "{movie_titles[test_mid]}":', flush=True)
    for pos, score in zip(p[0], s[0]):
        if pos == test_midx or pos == 0:
            continue
        mid = idx2movie[pos]
        title = movie_titles.get(mid, f'id={mid}')
        print(f'  {title[:45]:45s} {score:.4f}', flush=True)

print('\nDone! (Hard negative model)', flush=True)
