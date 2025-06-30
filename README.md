# Cryptopond Model Submissions 🏗️🪙

A curated collection of **Omniacs.DAO** data-science & ML workflows that competed on the [Pond](https://cryptopond.xyz) platform and adjacent grant contests.  
Each sub-folder is a *self-contained* repo with its own README, dataset layout and scripts; this top-level file simply gives you the “grand tour” so you can pick the project you’re after in seconds.

---

## 📂 Project index

| Folder | Contest / Challenge | What the model does | Primary tech |
|--------|--------------------|---------------------|--------------|
| [`GG23-predictive-funding-challenge`](GG23-predictive-funding-challenge/) | **Gitcoin Grants 23 – Predictive Funding** | Embeds every grant description, adds historical round stats and learns two **XGBoost** regressors to forecast both community contributions *and* matching-pool share. Exports three ready-to-upload CSV submissions. | R (`tidyverse`, `ollama`) + Python (`xgboost`, `scikit-learn`) :contentReference[oaicite:0]{index=0} |
| [`deep-funding-mini-contest-pipeline`](deep-funding-mini-contest-pipeline/) | **Cardano Deep Funding Mini-Contest** | Scrapes GitHub & OSO BigQuery stats for all repos, generates >1 000 engineered features + doc-embeddings, then trains a *stacked* ensemble (XGB + RF + GB + SVR → XGB meta-learner). Also ships a UMAP viewer dataset and a local **Shiny** dashboard for exploratory analysis. | Python (`pandas`, `xgboost`, `umap-learn`), R (Shiny) :contentReference[oaicite:1]{index=1} |
| [`sybil-detection-with-human-passport-and-octant`](sybil-detection-with-human-passport-and-octant/) | **Holonym × Octant Sybil-Wallet Detection** | Crafts on-chain behaviour & graph-embedding features for wallets on Ethereum + Base, then tunes a **LightGBM** GBDT with Optuna. Three-command pipeline: create venv → generate features → train & predict. | Python (`networkx`, `node2vec`, `lightgbm`) :contentReference[oaicite:2]{index=2} |

> **Tip:** each sub-README includes exact CLI commands, dependency lists and data expectations – follow them verbatim for reproducibility.

---

## 🔧 Quick start for *any* submission

```bash
git clone https://github.com/OmniacsDAO/CryptopondSubmissions.git
cd CryptopondSubmissions/<project-folder>
# now follow the 1-2-3 in that folder’s README
