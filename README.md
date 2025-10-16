# Cryptopond Model Submissions 🏗️

*Data-science & ML workflows created by **Omniacs.DAO** for the [Cryptopond](https://cryptopond.xyz) platform and allied grant challenges.*

Each sub-folder is a **stand-alone, reproducible project** with its own README, dataset layout, and scripts.  
This top-level file is just the “grand tour” so you can jump straight to the one you need.

---

## 📂 Project index

| Folder | Contest / Challenge | At a glance | Main stack |
|--------|--------------------|-------------|------------|
| [`GG23-predictive-funding-challenge`](GG23-predictive-funding-challenge/) | **Gitcoin Grants 23 – Predictive Funding** | Embeds every grant description, adds round stats, and trains two **XGBoost** regressors to forecast community contributions **and** matching-pool share. Exports three ready-to-upload CSV submissions. | R (`tidyverse`, `ollama`) · Python (`xgboost`, `scikit-learn`) |
| [`deep-funding-mini-contest-pipeline`](deep-funding-mini-contest-pipeline/) | **Improve the Funding Mechanism for Ethereum Projects – Mini-Contest** | Pulls GitHub & BigQuery OSS metrics, generates >1 000 engineered features + doc embeddings, then fits a *stacked* ensemble (XGB + RF + GB + SVR → XGB meta-learner). Ships a UMAP viewer dataset **and** a local Shiny dashboard. | Python (`pandas`, `xgboost`, `umap-learn`) · R (`shiny`) |
| [`sybil-detection-with-human-passport-and-octant`](sybil-detection-with-human-passport-and-octant/) | **Human Passport × Octant – Sybil Wallet Detection** | Crafts on-chain behaviour & graph-embedding features for wallets on Ethereum + Base, then tunes a **LightGBM** model with Optuna. Three commands: create venv → generate features → train & predict. | Python (`networkx`, `node2vec`, `lightgbm`) |
| [`ethereum-open-source-contrib-quantifier`](https://github.com/OmniacsDAO/CryptopondSubmissions/tree/main/ethereum-open-source-contrib-quantifier) | **Quantifying Contributions of Open Source Projects to the Ethereum Universe** | Describes the process of fitting various regression, ML and scoring models to the pairwise juror data. Discusses overfitting to the public leaderboard and the results from that effort. | Python · JMP  · R (`DirichletReg`) · ChatGPT 5 |

---

## 🔧 Quick start

```bash
# 1. Clone the repo
git clone https://github.com/OmniacsDAO/CryptopondSubmissions.git
cd CryptopondSubmissions

# 2. Pick a project and follow its README
cd GG23-predictive-funding-challenge   # (or another folder)
```

> **Tip:** All projects were built on Linux with Python 3.11 (and, where noted, R ≥ 4.3).
> Use a fresh `virtualenv`/`conda` environment *per project* to avoid version clashes.

---

## ✨ Guiding principles

1. **Public-goods first** – every model tackles an open-source funding or Web3 governance pain-point.
2. **End-to-end reproducibility** – raw data acquisition, feature engineering, and training are *fully scripted*.
3. **Interactivity where it helps** – dashboards or viewer datasets are provided so non-technical DAO members can explore results.

Feel free to fork, adapt, or just peek under the hood.
If this work helps your own public-goods efforts, a ⭐ on the repo (and maybe a few \$IACS tokens 😉) is always welcome!

---

*Maintained with ❤️ by **Omniacs.DAO** – accelerating digital public goods through data.*

---
🛠️ Keep public infrastructure thriving. Buy [$IACS](http://dexscreener.com/base/0xd4d742cc8f54083f914a37e6b0c7b68c6005a024) on Base — CA: 0x46e69Fa9059C3D5F8933CA5E993158568DC80EBf
