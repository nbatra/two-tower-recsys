"""
Approach D: Fine-tune the baseline model with mixed negatives.

Key insight from failed approaches:
- Pure random negatives (baseline): AUC 0.60, Recall@100 0.16 -- decent but ceiling is low
- Pure hard negatives: model can't generalize, AUC < 0.5
- InfoNCE: temperature collapse, no ranking signal

New strategy:
1. Start from the PRETRAINED baseline model (already learned general preferences)
2. Fine-tune with a MIX of negatives:
   - 50% hard negatives (items ranked 30-200 from baseline FAISS)
   - 50% random negatives (maintain calibration)
3. Use a smaller dataset (fewer pairs) but more informative
4. Lower LR (1e-4) since we're fine-tuning, not training from scratch
5. Short training (3-4 epochs) to avoid overfitting the hard negatives
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

# ============================================================
# 2. Mine hard negatives (items ranked 30-200 per user)
# ============================================================
print('\nMining hard negatives...', flush=True)
t0 = time.time()

user_embeddings_baseline = np.load(MODEL_DIR / 'user_embeddings.npy')
item_embeddings_baseline = np.load(MODEL_DIR / 'item_embeddings.npy')

index_baseline = faiss.IndexFlatIP(64)
index_baseline.add(item_embeddings_baseline)

# Build positive sets per user
user_positive_sets = {}
for i in range(n_positives):
    uid = int(train_pos_user_idx[i])
    mid = int(train_pos_movie_idx[i])
    if uid not in user_positive_sets:
        user_positive_sets[uid] = set()
    user_positive_sets[uid].add(mid)

unique_users = np.array(list(user_positive_sets.keys()), dtype=np.int32)
n_unique_users = len(unique_users)

# Generate 1 hard negative per positive
rng = np.random.default_rng(123)
hard_neg_user = []
hard_neg_movie = []

SEARCH_K = 200
HARD_START = 30
BATCH = 2048

for batch_start in range(0, n_unique_users, BATCH):
    batch_end = min(batch_start + BATCH, n_unique_users)
    batch_users = unique_users[batch_start:batch_end]
    batch_embs = user_embeddings_baseline[batch_users]

    _, positions = index_baseline.search(batch_embs, SEARCH_K)

    for i, uid in enumerate(batch_users):
        pos_set = user_positive_sets[uid]
        candidates = [int(p) for p in positions[i, HARD_START:] if p != 0 and int(p) not in pos_set]

        if len(candidates) == 0:
            continue

        n_user_pos = len(pos_set)
        sampled = rng.choice(candidates, size=min(n_user_pos, len(candidates)), replace=len(candidates) < n_user_pos)

        for neg_mid in sampled:
            hard_neg_user.append(uid)
            hard_neg_movie.append(neg_mid)

    if batch_start % (BATCH * 10) == 0:
        print(f'  {batch_end:,}/{n_unique_users:,} users processed', flush=True)

hard_neg_user = np.array(hard_neg_user, dtype=np.int32)
hard_neg_movie = np.array(hard_neg_movie, dtype=np.int32)
n_hard = len(hard_neg_user)
print(f'Hard negatives: {n_hard:,} in {time.time()-t0:.0f}s', flush=True)

del user_embeddings_baseline, item_embeddings_baseline, index_baseline
gc.collect()

# ============================================================
# 3. Generate random negatives (same count as hard)
# ============================================================
print('Generating random negatives...', flush=True)
rand_neg_user = np.zeros(n_hard, dtype=np.int32)
rand_neg_movie = np.zeros(n_hard, dtype=np.int32)

# Sample random items for random users from our positive set
rand_user_indices = rng.integers(0, n_positives, size=n_hard)
rand_neg_user[:] = train_pos_user_idx[rand_user_indices]
rand_neg_movie[:] = rng.integers(1, n_movies, size=n_hard).astype(np.int32)

print(f'Random negatives: {n_hard:,}', flush=True)

# ============================================================
# 4. Combine: positives + hard negatives + random negatives
# ============================================================
train_user_idx = np.concatenate([train_pos_user_idx, hard_neg_user, rand_neg_user])
train_movie_idx = np.concatenate([train_pos_movie_idx, hard_neg_movie, rand_neg_movie])
train_labels = np.concatenate([
    np.ones(n_positives, dtype=np.float32),
    np.zeros(n_hard, dtype=np.float32),
    np.zeros(n_hard, dtype=np.float32)
])

n_train = len(train_labels)
print(f'\nTotal training set: {n_train:,}', flush=True)
print(f'  Positives: {n_positives:,} ({n_positives/n_train:.1%})', flush=True)
print(f'  Hard negatives: {n_hard:,} ({n_hard/n_train:.1%})', flush=True)
print(f'  Random negatives: {n_hard:,} ({n_hard/n_train:.1%})', flush=True)

del hard_neg_user, hard_neg_movie, rand_neg_user, rand_neg_movie
gc.collect()

# ============================================================
# 5. Model (same architecture, load pretrained baseline)
# ============================================================
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
OUTPUT_DIM = 64
DROPOUT = 0.2

class UserTower(nn.Module):
    def __init__(self, n_users, user_feature_dim, embedding_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim, padding_idx=0)
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

# Load pretrained baseline weights
model.load_state_dict(torch.load(MODEL_DIR / 'two_tower_model.pt', map_location=device))
print(f'Loaded pretrained baseline model (AUC=0.60)', flush=True)
print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}', flush=True)

# ============================================================
# 6. Fine-tune with low LR
# ============================================================
BATCH_SIZE = 8192
EPOCHS = 5
LOG_EVERY = 1000

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.BCEWithLogitsLoss()

n_batches_train = n_train // BATCH_SIZE
perm = np.arange(n_train, dtype=np.int32)

print(f'\nBatch size: {BATCH_SIZE:,}, Batches/epoch: {n_batches_train:,}', flush=True)
print(f'LR: 1e-4 (fine-tuning from pretrained)', flush=True)

best_auc = 0.0
best_epoch = 0
history = []

print('\nStarting fine-tuning with mixed negatives...', flush=True)
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

        total_loss += loss.item()
        n_done += 1

        if (batch_idx + 1) % LOG_EVERY == 0:
            avg_loss = total_loss / n_done
            lr = optimizer.param_groups[0]['lr']
            print(f'  Batch {batch_idx+1:,}/{n_batches_train:,} - Loss: {avg_loss:.4f}, LR: {lr:.6f}', flush=True)

    train_loss = total_loss / n_done
    scheduler.step()

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
        torch.save(model.state_dict(), MODEL_DIR / 'two_tower_finetune.pt')
        print(f'  --> Best model saved (AUC={best_auc:.4f})', flush=True)

print(f'\nBest Val AUC: {best_auc:.4f} at epoch {best_epoch}', flush=True)
print(f'Baseline was: 0.5997', flush=True)

with open(MODEL_DIR / 'history_finetune.pkl', 'wb') as f:
    pickle.dump(history, f)

# ============================================================
# 7. Extract embeddings and evaluate
# ============================================================
model.load_state_dict(torch.load(MODEL_DIR / 'two_tower_finetune.pt', map_location=device))
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

np.save(MODEL_DIR / 'item_embeddings_finetune.npy', item_embeddings)
np.save(MODEL_DIR / 'user_embeddings_finetune.npy', user_embeddings)

index = faiss.IndexFlatIP(OUTPUT_DIM)
index.add(item_embeddings)
faiss.write_index(index, str(MODEL_DIR / 'faiss_index_finetune.bin'))

# Recall@K
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

print(f'\nFine-tuned Recall@K:', flush=True)
for k in K_values:
    print(f'  Recall@{k:<4}: {np.mean(recalls[k]):.4f}', flush=True)

# Compare to baseline
print(f'\nBaseline Recall@K (for reference):', flush=True)
print(f'  Recall@10  : 0.0171', flush=True)
print(f'  Recall@50  : 0.0874', flush=True)
print(f'  Recall@100 : 0.1627', flush=True)
print(f'  Recall@200 : 0.2682', flush=True)
print(f'  Recall@500 : 0.4667', flush=True)

# Item similarity
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

print('\nDone! (Fine-tuned model)', flush=True)
