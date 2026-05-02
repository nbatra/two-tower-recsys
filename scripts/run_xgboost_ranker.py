"""Train XGBoost re-ranker on Two-Tower embeddings + features."""
import numpy as np
import pandas as pd
import pickle
import time
import os
import gc
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['MPLBACKEND'] = 'Agg'

import xgboost as xgb
from sklearn.metrics import roc_auc_score

DATA_DIR = Path('data/processed')
MODEL_DIR = Path('models')

print(f'XGBoost version: {xgb.__version__}', flush=True)

# ============================================================
# 1. Load embeddings and feature matrices
# ============================================================
with open(DATA_DIR / 'metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

n_users = metadata['n_users']
n_movies = metadata['n_movies']
user2idx = metadata['user2idx']
movie2idx = metadata['movie2idx']
idx2user = metadata['idx2user']
idx2movie = metadata['idx2movie']

user_embeddings = np.load(MODEL_DIR / 'user_embeddings_128dim.npy')
item_embeddings = np.load(MODEL_DIR / 'item_embeddings_128dim.npy')
print(f'Embeddings: user={user_embeddings.shape}, item={item_embeddings.shape}', flush=True)

user_features_df = pd.read_parquet(DATA_DIR / 'user_features.parquet')
item_features_df = pd.read_parquet(DATA_DIR / 'item_features.parquet')
user_feat_cols = user_features_df.columns.tolist()
item_feat_cols = item_features_df.columns.tolist()

user_feat_matrix = np.zeros((n_users, len(user_feat_cols)), dtype=np.float32)
for uid, uidx in user2idx.items():
    if uid in user_features_df.index:
        user_feat_matrix[uidx] = user_features_df.loc[uid].values

item_feat_matrix = np.zeros((n_movies, len(item_feat_cols)), dtype=np.float32)
for mid, midx in movie2idx.items():
    if mid in item_features_df.index:
        item_feat_matrix[midx] = item_features_df.loc[mid].values

del user_features_df, item_features_df
gc.collect()
print(f'Feature matrices: user={user_feat_matrix.shape}, item={item_feat_matrix.shape}', flush=True)

# ============================================================
# 2. Load and subsample training data
# ============================================================
train_df = pd.read_parquet(DATA_DIR / 'train_set.parquet')
val_df = pd.read_parquet(DATA_DIR / 'val_set.parquet')
train_interaction_feats = pd.read_parquet(DATA_DIR / 'train_interaction_features.parquet')
val_interaction_feats = pd.read_parquet(DATA_DIR / 'val_interaction_features.parquet')

# Filter padding from val
valid_val_mask = val_df['user_idx'] > 0
val_df = val_df[valid_val_mask].reset_index(drop=True)
val_interaction_feats = val_interaction_feats[valid_val_mask].reset_index(drop=True)
print(f'Train: {len(train_df):,}, Val: {len(val_df):,}', flush=True)

# Subsample train to 3M rows
TRAIN_SAMPLE_SIZE = 3_000_000
np.random.seed(42)
user_groups = train_df.groupby('user_idx').size()
users_shuffled = user_groups.index.values.copy()
np.random.shuffle(users_shuffled)

cumulative = 0
selected_users = []
for u in users_shuffled:
    selected_users.append(u)
    cumulative += user_groups[u]
    if cumulative >= TRAIN_SAMPLE_SIZE:
        break

train_mask = train_df['user_idx'].isin(set(selected_users))
train_user_idxs = train_df.loc[train_mask, 'user_idx'].values.copy()
train_movie_idxs = train_df.loc[train_mask, 'movie_idx'].values.copy()
y_train = train_df.loc[train_mask, 'label'].values.astype(np.float32)
train_cross_arr = train_interaction_feats.loc[train_mask].values.astype(np.float32)
print(f'Subsampled: {len(y_train):,} rows from {len(selected_users):,} users', flush=True)

del train_df, train_interaction_feats
gc.collect()

# Val arrays
val_user_idxs = val_df['user_idx'].values.copy()
val_movie_idxs = val_df['movie_idx'].values.copy()
y_val = val_df['label'].values.astype(np.float32)
val_cross_arr = val_interaction_feats.values.astype(np.float32)
del val_df, val_interaction_feats
gc.collect()

# Feature names
feature_names = (
    ['retrieval_score'] +
    [f'user_{c}' for c in user_feat_cols] +
    [f'item_{c}' for c in item_feat_cols] +
    [f'cross_{c}' for c in ['genre_match_score', 'popularity_gap', 'movie_age_at_rating',
                             'dow_sin', 'dow_cos', 'hour_sin', 'hour_cos']]
)
print(f'Features: {len(feature_names)}', flush=True)

# ============================================================
# 3. Build feature matrices (chunked)
# ============================================================
def build_features_chunked(user_idxs, movie_idxs, cross_arr, chunk_size=500_000):
    n = len(user_idxs)
    n_features = 1 + len(user_feat_cols) + len(item_feat_cols) + 7
    features = np.empty((n, n_features), dtype=np.float32)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        u_idx = user_idxs[start:end]
        m_idx = movie_idxs[start:end]
        features[start:end, 0] = np.sum(user_embeddings[u_idx] * item_embeddings[m_idx], axis=1)
        features[start:end, 1:1+len(user_feat_cols)] = user_feat_matrix[u_idx]
        features[start:end, 1+len(user_feat_cols):1+len(user_feat_cols)+len(item_feat_cols)] = item_feat_matrix[m_idx]
        features[start:end, -7:] = cross_arr[start:end]
    return features

print('Building training features...', flush=True)
t0 = time.time()
X_train = build_features_chunked(train_user_idxs, train_movie_idxs, train_cross_arr)
del train_cross_arr
gc.collect()
print(f'  X_train: {X_train.shape}, {X_train.nbytes/1e9:.2f} GB, {time.time()-t0:.1f}s', flush=True)

print('Building validation features...', flush=True)
t0 = time.time()
X_val = build_features_chunked(val_user_idxs, val_movie_idxs, val_cross_arr)
del val_cross_arr
gc.collect()
print(f'  X_val: {X_val.shape}, {X_val.nbytes/1e9:.2f} GB, {time.time()-t0:.1f}s', flush=True)

# ============================================================
# 4. Sort and create groups
# ============================================================
print('Sorting for group construction...', flush=True)
train_sort_idx = np.argsort(train_user_idxs, kind='stable')
X_train = X_train[train_sort_idx]
y_train = y_train[train_sort_idx]
train_user_sorted = train_user_idxs[train_sort_idx]

val_sort_idx = np.argsort(val_user_idxs, kind='stable')
X_val = X_val[val_sort_idx]
y_val = y_val[val_sort_idx]
val_user_sorted = val_user_idxs[val_sort_idx]

del train_user_idxs, train_movie_idxs, val_user_idxs, val_movie_idxs
gc.collect()

_, train_group_counts = np.unique(train_user_sorted, return_counts=True)
_, val_group_counts = np.unique(val_user_sorted, return_counts=True)
train_groups = train_group_counts.tolist()
val_groups = val_group_counts.tolist()
print(f'Groups: train={len(train_groups):,} users, val={len(val_groups):,} users', flush=True)

# ============================================================
# 5. Create DMatrix and train
# ============================================================
print('Creating DMatrix...', flush=True)
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
dtrain.set_group(train_groups)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
dval.set_group(val_groups)
del X_train
gc.collect()
print(f'DMatrix: train={dtrain.num_row():,}, val={dval.num_row():,}', flush=True)

params = {
    'objective': 'rank:ndcg',
    'eval_metric': 'ndcg@10',
    'tree_method': 'hist',
    'max_depth': 8,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 50,
    'gamma': 1.0,
    'reg_lambda': 1.0,
    'nthread': 4,
    'seed': 42,
    'verbosity': 1,
}

print('\nTraining XGBoost ranker (LambdaMART)...', flush=True)
t0 = time.time()
evals_result = {}
model = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    evals=[(dtrain, 'train'), (dval, 'val')],
    evals_result=evals_result,
    early_stopping_rounds=30,
    verbose_eval=50
)
train_time = time.time() - t0
print(f'\nTraining complete in {train_time:.0f}s', flush=True)
print(f'Best iteration: {model.best_iteration}', flush=True)
print(f'Best val NDCG@10: {model.best_score:.4f}', flush=True)

# ============================================================
# 6. Evaluate
# ============================================================
val_scores = model.predict(dval)
val_auc = roc_auc_score(y_val, val_scores)
retrieval_auc = roc_auc_score(y_val, X_val[:, 0])

print(f'\nTwo-Tower retrieval AUC: {retrieval_auc:.4f}', flush=True)
print(f'XGBoost ranker AUC:      {val_auc:.4f}', flush=True)
print(f'Improvement:             +{val_auc - retrieval_auc:.4f}', flush=True)

# Feature importance top 10
importance = model.get_score(importance_type='gain')
importance_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)
print(f'\nTop 10 features by gain:', flush=True)
for i, (feat, gain) in enumerate(importance_sorted[:10], 1):
    print(f'  {i}. {feat}: {gain:.1f}', flush=True)

# Per-user ranking metrics
def compute_ranking_metrics(scores, labels, group_sizes, K_values=[5, 10, 20]):
    metrics = {f'ndcg@{k}': [] for k in K_values}
    metrics.update({f'precision@{k}': [] for k in K_values})
    metrics['mrr'] = []
    offset = 0
    for group_size in group_sizes:
        group_labels = labels[offset:offset + group_size]
        group_scores = scores[offset:offset + group_size]
        offset += group_size
        if group_labels.sum() == 0 or group_size < 2:
            continue
        rank_order = np.argsort(group_scores)[::-1]
        ranked_labels = group_labels[rank_order]
        first_pos = np.where(ranked_labels == 1)[0]
        metrics['mrr'].append(1.0 / (first_pos[0] + 1) if len(first_pos) > 0 else 0.0)
        for k in K_values:
            actual_k = min(k, len(ranked_labels))
            top_k = ranked_labels[:actual_k]
            metrics[f'precision@{k}'].append(top_k.sum() / k)
            dcg = np.sum(top_k / np.log2(np.arange(2, actual_k + 2)))
            ideal = np.sort(group_labels)[::-1][:actual_k]
            idcg = np.sum(ideal / np.log2(np.arange(2, actual_k + 2)))
            metrics[f'ndcg@{k}'].append(dcg / idcg if idcg > 0 else 0.0)
    return {k: np.mean(v) for k, v in metrics.items()}

max_groups = 3000
eval_groups = val_groups[:max_groups]
eval_size = sum(eval_groups)

ranker_metrics = compute_ranking_metrics(val_scores[:eval_size], y_val[:eval_size], eval_groups)
retrieval_metrics = compute_ranking_metrics(X_val[:eval_size, 0], y_val[:eval_size], eval_groups)

print(f'\n{"Metric":<15}{"Two-Tower":<15}{"XGBoost":<15}{"Improvement":<15}', flush=True)
print('-' * 60, flush=True)
for metric in ['ndcg@5', 'ndcg@10', 'ndcg@20', 'precision@5', 'precision@10', 'mrr']:
    r = ranker_metrics[metric]
    t = retrieval_metrics[metric]
    print(f'{metric:<15}{t:<15.4f}{r:<15.4f}{r-t:+.4f}', flush=True)

# ============================================================
# 7. Save model
# ============================================================
model.save_model(str(MODEL_DIR / 'xgboost_ranker.json'))
with open(MODEL_DIR / 'ranker_feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
with open(MODEL_DIR / 'xgboost_evals_result.pkl', 'wb') as f:
    pickle.dump(evals_result, f)
print(f'\nModel saved: {MODEL_DIR / "xgboost_ranker.json"}', flush=True)
print(f'Trees: {model.best_iteration + 1}', flush=True)

# ============================================================
# 8. End-to-end demo
# ============================================================
import faiss
index = faiss.read_index(str(MODEL_DIR / 'faiss_index_128dim.bin'))
movies_df = pd.read_csv('data/ml-25m/movies.csv')
movie_titles = dict(zip(movies_df['movieId'], movies_df['title']))
movie_genres = dict(zip(movies_df['movieId'], movies_df['genres']))

for user_id in [1, 100, 1000]:
    user_idx = user2idx.get(user_id)
    if user_idx is None:
        continue
    user_emb = user_embeddings[user_idx:user_idx+1]
    scores, positions = index.search(user_emb, 200)
    candidate_idxs = positions[0]
    retrieval_scores_demo = scores[0]

    n_cands = len(candidate_idxs)
    n_f = 1 + len(user_feat_cols) + len(item_feat_cols) + 7
    X_cand = np.zeros((n_cands, n_f), dtype=np.float32)
    X_cand[:, 0] = retrieval_scores_demo
    X_cand[:, 1:1+len(user_feat_cols)] = user_feat_matrix[user_idx]
    X_cand[:, 1+len(user_feat_cols):1+len(user_feat_cols)+len(item_feat_cols)] = item_feat_matrix[candidate_idxs]
    user_genre_prefs = user_feat_matrix[user_idx, 4:23]
    X_cand[:, -7] = item_feat_matrix[candidate_idxs, :19] @ user_genre_prefs
    X_cand[:, -6] = item_feat_matrix[candidate_idxs, 20] - user_feat_matrix[user_idx, 23]

    dcand = xgb.DMatrix(X_cand, feature_names=feature_names)
    ranker_scores_demo = model.predict(dcand)
    top_indices = np.argsort(ranker_scores_demo)[::-1][:10]

    print(f'\nRecommendations for User {user_id}:', flush=True)
    print(f'{"Rank":<6}{"Title":<50}{"Genres":<30}{"Score":<8}', flush=True)
    print('-' * 94, flush=True)
    for rank, idx in enumerate(top_indices, 1):
        midx = candidate_idxs[idx]
        if midx == 0:
            continue
        mid = idx2movie[midx]
        title = movie_titles.get(mid, f'id={mid}')[:49]
        genres = movie_genres.get(mid, '')[:29]
        print(f'{rank:<6}{title:<50}{genres:<30}{ranker_scores_demo[idx]:<8.3f}', flush=True)

print('\nDone!', flush=True)
