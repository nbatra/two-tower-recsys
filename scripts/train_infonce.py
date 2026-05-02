"""
Approach A: InfoNCE loss with in-batch negatives.

Key differences from baseline:
- Train on 12M positive pairs only (not 69M with random negatives)
- Each batch of 8192 positives: each item becomes a negative for all other users
- The loss forces the model to distinguish a user's actual movie from 8191 other movies
- Much harder task -> learns finer-grained personalization
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

# Load positive-only training pairs
train_user_idx = np.load(DATA_DIR / 'tt_pos_user_idx.npy')
train_movie_idx = np.load(DATA_DIR / 'tt_pos_movie_idx.npy')
n_train = len(train_user_idx)

# Validation (keep original for AUC comparison)
val_user_idx = np.load(DATA_DIR / 'val_user_idx.npy')
val_movie_idx = np.load(DATA_DIR / 'val_movie_idx.npy')
val_labels = np.load(DATA_DIR / 'val_labels.npy')
n_val = len(val_labels)

print(f'Users: {n_users:,}, Movies: {n_movies:,}', flush=True)
print(f'Train (positives only): {n_train:,}', flush=True)
print(f'Val: {n_val:,}', flush=True)

# ============================================================
# 2. Model
# ============================================================
EMBEDDING_DIM = 32
HIDDEN_DIM = 128
OUTPUT_DIM = 64
DROPOUT = 0.1

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
        self.temperature = nn.Parameter(torch.tensor(0.07).log())

    def forward(self, user_idx, user_features, movie_idx, item_features):
        user_emb = self.user_tower(user_idx, user_features)
        item_emb = self.item_tower(movie_idx, item_features)
        return user_emb, item_emb

    def get_user_embeddings(self, user_idx, user_features):
        return self.user_tower(user_idx, user_features)

    def get_item_embeddings(self, movie_idx, item_features):
        return self.item_tower(movie_idx, item_features)

model = TwoTowerModel(n_users, n_movies, user_feature_dim, item_feature_dim,
                      EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, DROPOUT).to(device)
print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}', flush=True)

# ============================================================
# 3. Training with InfoNCE
# ============================================================
BATCH_SIZE = 4096
EPOCHS = 10
LOG_EVERY = 500

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

n_batches_train = n_train // BATCH_SIZE
perm = np.arange(n_train, dtype=np.int32)

print(f'\nBatch size: {BATCH_SIZE:,}, Batches/epoch: {n_batches_train:,}', flush=True)
print(f'In-batch negatives per positive: {BATCH_SIZE - 1}', flush=True)

best_auc = 0.0
best_epoch = 0
history = []

print('\nStarting InfoNCE training...', flush=True)
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

        user_feats = user_feature_tensor[u_idx]
        item_feats = item_feature_tensor[m_idx]

        user_emb, item_emb = model(u_idx, user_feats, m_idx, item_feats)

        # InfoNCE: similarity matrix [B x B], labels are diagonal
        temp = model.temperature.exp().clamp(min=0.01, max=1.0)
        logits = torch.mm(user_emb, item_emb.t()) / temp  # [B, B]
        labels = torch.arange(BATCH_SIZE, device=device)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_done += 1

        if (batch_idx + 1) % LOG_EVERY == 0:
            avg_loss = total_loss / n_done
            temp_val = model.temperature.exp().item()
            print(f'  Batch {batch_idx+1:,}/{n_batches_train:,} - Loss: {avg_loss:.4f}, temp: {temp_val:.4f}', flush=True)

    train_loss = total_loss / n_done
    scheduler.step()

    # Validate with AUC (using dot product as score)
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

            user_emb, item_emb = model(u_idx, user_feats, m_idx, item_feats)
            scores = (user_emb * item_emb).sum(dim=1)

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
        torch.save(model.state_dict(), MODEL_DIR / 'two_tower_infonce.pt')
        print(f'  --> Best model saved (AUC={best_auc:.4f})', flush=True)

print(f'\nBest Val AUC: {best_auc:.4f} at epoch {best_epoch}', flush=True)

with open(MODEL_DIR / 'history_infonce.pkl', 'wb') as f:
    pickle.dump(history, f)

# ============================================================
# 4. Extract embeddings and build FAISS
# ============================================================
model.load_state_dict(torch.load(MODEL_DIR / 'two_tower_infonce.pt', map_location=device))
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

np.save(MODEL_DIR / 'item_embeddings_infonce.npy', item_embeddings)
np.save(MODEL_DIR / 'user_embeddings_infonce.npy', user_embeddings)

index = faiss.IndexFlatIP(OUTPUT_DIM)
index.add(item_embeddings)
faiss.write_index(index, str(MODEL_DIR / 'faiss_index_infonce.bin'))

# ============================================================
# 5. Recall@K
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

print(f'\nInfoNCE Recall@K:', flush=True)
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

print('\nDone! (InfoNCE model)', flush=True)
