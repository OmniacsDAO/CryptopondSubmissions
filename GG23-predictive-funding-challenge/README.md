# GG23 Predictive Funding Challenge

End-to-end workflow for our **GG23 Predictive Funding Challenge** entry on the Pond platform (model page 🔗: [https://cryptopond.xyz/modelfactory/detail/5249184](https://cryptopond.xyz/modelfactory/detail/5249184)).

Minimal R + Python pipeline that builds description embeddings, trains XGBoost regressors, and produces three CSV submissions predicting final funding for every project in **Gitcoin Grants Round 23**.

---

## Pipeline Overview

| Step | Script                | Exact actions in code                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ---- | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | `1_prep_train_data.R` | • Load **`GG Allocation Since GG18.csv`**.<br>• Build table with `Round`, `ProjID`, `NumContributors`, `Amt`, `MatchingAmt`, `TotalAmt`, `MatchingPool`.<br>• Query Gitcoin GraphQL (`get_gitcoin_project`) for each project’s `description`.<br>• Embed descriptions to 768-D vectors via `ollama::embeddings(model=\"nomic-embed-text:v1.5\")` at `http://192.168.11.98:9000`.<br>• Aggregate to per-project rows with `Amt` (mean) and `MatchingPoolPct` (share of round pool).<br>• Save **`train.csv`**.                                                                                                                                                                                      |
| 2    | `2_prep_test_data.R`  | • Read **`manualmatches2.csv`** with `ROUND`, `PROJECT_ID`, `Description`.<br>• Fill any missing descriptions through the same GraphQL call.<br>• Embed each description to 768-D vectors (same model/endpoint).<br>• Save **`test.csv`**.                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| 3    | `3_model.py`          | • Load `train.csv`, `test.csv`, and official **`projects_Apr_1.csv`**.<br>• Train two `XGBRegressor`s (500 trees, depth 6, lr 0.05):<br>  – **Target 1** `Amt` (community contributions).<br>  – **Target 2** `MatchingPoolPct`.<br>• Add per-round pool size (`MPOOL`: 200 k USD, or 600 k for *Mature Builders*).<br>• Zero predictions where `Live == 0`.<br>• Convert `MatchingPoolPct` to dollar amounts (`MPOOLPCT1`, `MPOOLPCT12`).<br>• Write three submission files:<br>  1. **`submission1.csv`** – `CONTRIBUTION + MPOOLPCT1`<br>  2. **`submission2.csv`** – `MPOOLPCT12` only<br>  3. **`submission3.csv`** – `CONTRIBUTION + MPOOLPCT1`, contributions zeroed for *Mature Builders*. |

*Each model outputs train/validation RMSE to stdout.*

---

## Requirements

**R:** `httr`, `jsonlite`, `readr`, `ollamar`
**Python:** `pandas`, `scikit-learn`, `xgboost`, `matplotlib`, `shap`
Embedding server reachable at **`http://192.168.11.98:9000`**

---

## Reproduce Results

```bash
# Build training data with embeddings
Rscript 1_prep_train_data.R

# Build test data with embeddings
Rscript 2_prep_test_data.R

# Train models and create GG23 submission CSVs
python 3_model.py
```

Three ready-to-upload CSVs will appear in `dataset/pred/`.
