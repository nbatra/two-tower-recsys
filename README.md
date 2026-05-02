# Two-Tower Recommendation Architecture: From Theory to Production

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-LambdaMART-blue)](https://xgboost.readthedocs.io)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-4285F4)](https://github.com/facebookresearch/faiss)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-Educational-green)](LICENSE)

**Keywords:** `Two-Tower Model` `Dual Encoder` `Recommendation System` `Collaborative Filtering` `ComiRec` `SASRec` `Transformer` `Capsule Network` `Multi-Interest Retrieval` `Sequential Recommendation` `FAISS` `Approximate Nearest Neighbor` `XGBoost LambdaMART` `Learning to Rank` `NDCG` `A/B Testing` `MovieLens` `Candidate Generation` `Re-Ranking` `Feature Store` `Embedding Retrieval` `Cold Start` `Production ML` `RecSys`

> Developed and trained entirely on a MacBook M4 Max (64GB RAM). The compute constraint was intentional -- it forced us to make the same trade-offs a production team faces when deploying retrieval systems at scale: choosing embedding dimensions, model complexity, and index strategies that balance quality against resource budgets.

## What This Project Is

This is an implementation of the **two-tower (dual encoder) architecture** -- the dominant paradigm behind recommendation systems at most modern platforms that serve personalized content at scale. The core idea: encode users and items into a shared embedding space, then use approximate nearest-neighbor search to retrieve candidates in sub-millisecond time.

We chose [**MovieLens 25M**](https://grouplens.org/datasets/movielens/25m/) (25 million ratings, 138K users, 21K movies) as the demonstration dataset for two reasons:
1. It is large enough to expose real challenges (sparse interactions, long-tail items, cold-start users) while fitting in laptop memory
2. It has rich metadata (tag genome with 1,128 relevance scores per movie) that lets us engineer meaningful content features -- critical for the ranking stage

The project goes beyond a single model. After implementing the baseline Two-Tower retriever, we asked: **what are the known failure modes of single-embedding retrieval, and can alternative architectures address them?** This led to implementing two additional retrieval models, each targeting a specific limitation.

## The Problem with One Embedding Per User

The standard Two-Tower model compresses a user's entire preference history into a single 128-dimensional vector. This works well for users with coherent tastes (someone who watches only horror films gets an embedding deep in the horror cluster). But it fails for **eclectic users** -- someone who watches sci-fi, cooking shows, and French arthouse gets an embedding that points to the centroid of those three regions, which is a neighborhood containing none of them.

This is not a theoretical concern. In our evaluation, Two-Tower achieves the highest raw retrieval recall (0.305 at K=200) but the lowest end-to-end ranking quality (NDCG@10 = 0.032) precisely because its candidate pool is homogeneous.

## Three Retrieval Models

| Model | Paper | Core Idea | What It Fixes |
|-------|-------|-----------|---------------|
| **Two-Tower** | Covington et al. (2016) | Single user embedding from profile features; single FAISS query | Baseline. Fast, robust, good cold-start handling |
| **ComiRec** | Cen et al. (KDD 2020) | Multiple interest embeddings via dynamic capsule routing; one FAISS query per interest | Fixes the single-embedding bottleneck. Eclectic users get separate representations for each taste facet |
| **SASRec** | Kang & McAuley (ICDM 2018) | Transformer self-attention over interaction sequence; embedding reflects recent context | Fixes temporal blindness. Captures "what are you in the mood for right now" vs. lifetime average |

### Other models considered but not implemented

- **MIND (Multi-Interest Network with Dynamic Routing)** -- Similar multi-interest approach to ComiRec but uses a different routing mechanism. ComiRec's capsule routing gave cleaner interest separation in our experiments, and MIND's label-aware attention requires impression logs we don't have.
- **BERT4Rec** -- Bidirectional Transformer for sequences. Theoretically stronger than SASRec's unidirectional attention, but requires masked-item training (computationally expensive on M4 Max for 21K vocab) and the bidirectional signal adds marginal value when we ultimately compress to one embedding for FAISS.
- **DCN v2 (Deep & Cross Network)** -- A ranking model, not a retrieval model. It would replace XGBoost in stage 2 rather than the retrieval stage. On 200 candidates with 105 features, XGBoost LambdaMART already achieves NDCG@10 of 0.87 -- neural rankers struggle to beat gradient-boosted trees at this scale without significantly more training data.
- **GNN-based models (PinSage, LightGCN)** -- Require constructing and training on the full user-item interaction graph. Memory-intensive (the 25M-edge graph explodes the adjacency matrix) and slow to iterate on. Better suited for cluster environments.

## The Two-Stage Architecture

All three retrieval models feed into the same downstream pipeline. This is the industry standard because it decouples the "what to consider" decision from the "how to rank" decision:

```
Stage 1: Retrieval (this is where the three models differ)
    Goal: Narrow 21K items to ~200 candidates in <2ms
    Method: FAISS inner-product search on pre-computed embeddings
    Trade-off: Speed over precision (approximate relevance is fine)

Stage 2: Ranking (identical for all three models)
    Goal: Score 200 candidates with full feature richness
    Method: XGBoost LambdaMART on 105-109 features
    Features: retrieval score + user profile (24) + item profile (73) + cross features (7)
    Trade-off: Precision over speed (2ms for 200 candidates is acceptable)

Stage 3: Post-processing
    Goal: Ensure diversity and filter already-seen items
    Method: Maximal Marginal Relevance (MMR) re-ranking
```

The key architectural insight: the retrieval model does not need to perfectly rank items. Its job is to produce a **diverse, high-quality candidate pool**. The ranker handles fine-grained ordering using features the retrieval model cannot access (cross-features violate tower independence).

## Results

### End-to-End Performance

| Model | Retrieval Recall@200 | End-to-End NDCG@10 | Diversity (ILD) | Coverage |
|-------|---------------------|-------------------|-----------------|----------|
| Two-Tower | **0.305** (best) | 0.032 | 0.28 | baseline |
| ComiRec | 0.214 | **0.036** (best) | **0.54** (+91%) | +59% |
| SASRec | 0.264 | 0.033 | 0.41 (+46%) | +31% |

### Ranking Stage Performance (how well XGBoost orders given candidates)

| Model | Val NDCG@10 | Test NDCG@10 | Trees |
|-------|------------|-------------|-------|
| Two-Tower | 0.8814 | 0.8679 | 174 |
| ComiRec | 0.8755 | 0.8593 | 329 |
| SASRec | 0.8775 | 0.8601 | 283 |

### Production Latency (simulated, single-threaded Python)

| Percentile | Latency |
|-----------|---------|
| P50 | 3.31 ms |
| P95 | 4.47 ms |
| P99 | 4.70 ms |
| Throughput | 305 req/sec |

### A/B Test Simulation Results

- CTR differences between models are not statistically significant at n=1000/group (power analysis shows ~122K users per group needed to detect 2% CTR lift)
- Diversity improvement from ComiRec (+92% ILD) is highly significant (p < 0.001)
- SASRec fails the engagement guardrail (-12% users with clicks vs. control)
- Decision: ComiRec recommended for segmented rollout to eclectic users; Two-Tower remains default

## Why These Results Make Sense

**ComiRec wins end-to-end despite lowest recall** because recommendation quality is not just about finding relevant items -- it's about finding *diverse* relevant items. A user's test positives span multiple genres. ComiRec's multi-probe retrieval surfaces candidates from each genre independently, so the ranker has diverse material to promote. Two-Tower's single probe returns 200 items from one taste neighborhood -- even perfect ranking cannot surface a drama if only sci-fi was retrieved.

**XGBoost dominates regardless of retrieval model.** The retrieval_score feature (dot product from FAISS) ranks only #10-15 in XGBoost's feature importance. The top features are item_avg_rating, genre_match, and genome PCA dimensions. This means the retrieval model's ranking is largely overridden -- what matters is which items made it into the candidate pool at all.

**SASRec matches Two-Tower on latency but not quality.** The Transformer's context-aware embedding should capture "recent mood" -- but MovieLens ratings are sparse (median user has ~30 ratings over years). Sequential signal is weak compared to dense-interaction domains (music, news, e-commerce). SASRec would likely outperform on datasets with session-level granularity.

## Hardware Constraints and Design Decisions

The MacBook M4 Max (64GB unified memory, no discrete GPU) imposed constraints that mirror production realities:

| Decision | Why This Constraint Matters | Choice |
|----------|---------------------------|--------|
| Embedding dim = 128 | 138K users x 128 x 4 bytes = 70MB per model. At 256-dim, three models + FAISS indices exceed comfortable memory | 128-dim (standard in literature) |
| ComiRec K=4 interests | 4 FAISS probes must complete in <2ms for P95 SLA | 4 (not 8 or 16 interests) |
| SASRec: 2 layers, 2 heads | Training on MPS must complete in <2 hours; attention O(n^2) on sequence length | Minimal Transformer that still learns attention patterns |
| Sequence length = 50 | Attention matrix is 50x50 -- fits in L1 cache | 50 (vs. 200+ in paper) |
| FAISS IndexFlatIP | 21K items is small enough for exact brute-force search | Exact search (no recall loss from approximation) |
| XGBoost ~300 trees | DMatrix for 200K training rows x 109 features fits in memory | Early stopping prevents overfitting anyway |
| MovieLens 25M dataset | Largest MovieLens variant that fits in RAM after feature engineering (~4GB processed) | Could not use full industry-scale datasets (100M+ interactions) |

These constraints are representative. A production team deploying a new model at a mid-size company (10M users, 100K items) faces similar memory and latency budgets on inference servers.

## Project Structure

```
notebooks/
    00_architecture_overview.ipynb    # System design reference (no code)
    01_eda.ipynb                      # Data exploration, distribution analysis
    02_feature_engineering.ipynb      # Feature computation pipeline
    03_two_tower_training.ipynb       # Two-Tower model + embedding extraction
    04_two_tower_xgboost_ranker.ipynb # XGBoost LambdaMART for Two-Tower
    05_two_tower_evaluation.ipynb     # Evaluation: retrieval + ranking metrics
    06_comirec_training.ipynb         # ComiRec capsule network + multi-interest
    07_comirec_xgboost_ranker.ipynb   # XGBoost LambdaMART for ComiRec
    08_comirec_evaluation.ipynb       # Two-way comparison (TT vs ComiRec)
    09_sasrec_training.ipynb          # SASRec Transformer + sequence embeddings
    10_sasrec_xgboost_ranker.ipynb    # XGBoost LambdaMART for SASRec
    11_sasrec_evaluation.ipynb        # Three-way comparison (all 3 models)
    12_production_inference_simulation.ipynb  # Production service simulation
    13_ab_testing.ipynb               # A/B testing with statistical rigor

models/                              # Saved artifacts (FAISS indices, XGBoost, embeddings)
data/processed/                      # Engineered features, train/val/test splits
```

## Technical Stack

- **PyTorch** (MPS backend) -- Two-Tower, ComiRec capsule routing, SASRec Transformer
- **FAISS** -- Exact inner-product search on 128-dim embeddings
- **XGBoost** -- LambdaMART with rank:ndcg objective, 105-109 features
- **Feature engineering** -- 24-dim user profiles, 73-dim item profiles (19 genres + popularity + quality + 50 genome PCA dims), 7-dim cross features (genre match, popularity gap, temporal)
- **Evaluation** -- NDCG@K, Precision@K, Recall@K, Hit Rate, Intra-List Diversity, Catalog Coverage
- **A/B testing** -- Hash-based deterministic assignment, Welch's t-test, Bonferroni correction, power analysis, guardrail metrics

## How to Run

```bash
# Clone the repository
git clone https://github.com/nbatra/two-tower-recsys.git
cd two-tower-recsys

# Run setup script (creates venv, installs dependencies, downloads MovieLens 25M)
./setup.sh

# Launch Jupyter and run notebooks in order (00 is reference-only, start from 01)
.venv/bin/jupyter lab notebooks/
```

The setup script downloads the MovieLens 25M dataset (~250MB compressed, ~1GB extracted) from [grouplens.org](https://grouplens.org/datasets/movielens/25m/). Model artifacts and processed features are generated by running the notebooks sequentially. Full pipeline takes approximately 2-3 hours on an M4 Max.

## Key Takeaways

1. **The retrieval model's job is candidate diversity, not ranking precision.** The downstream ranker has richer features and dominates final ordering. Retrieval just needs to get the right items into the room.

2. **Multi-interest models win for eclectic users.** ComiRec's per-interest FAISS probes surface candidates that a single-embedding model structurally cannot reach, leading to +91% diversity and higher end-to-end NDCG.

3. **Sequential models need dense interaction data.** SASRec's Transformer architecture is powerful but MovieLens's sparse rating data (30 ratings over years) provides weak sequential signal. This model would shine on streaming/e-commerce data with session-level granularity.

4. **Production deployment is a solved problem at this scale.** Sub-5ms latency, 300 req/sec throughput, graceful cold-start degradation, and model routing -- all achievable on a laptop with no GPU at inference time. The architecture scales linearly with hardware.

5. **A/B testing requires massive sample sizes for engagement metrics.** Power analysis shows 122K users per group to detect a 2% CTR lift. Diversity differences are detectable at 1K users because variance is lower. This explains why industry experiments run for weeks on full traffic.

---

## Author

Built by **Nipun Batra**

[![GitHub](https://img.shields.io/badge/GitHub-nbatra-181717?logo=github)](https://github.com/nbatra)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-nipunbatra-0A66C2?logo=linkedin)](https://www.linkedin.com/in/nipunbatra/)

---

## License

This project is released for educational and portfolio purposes. The MovieLens 25M dataset is provided by [GroupLens Research](https://grouplens.org/) under their own terms of use.
