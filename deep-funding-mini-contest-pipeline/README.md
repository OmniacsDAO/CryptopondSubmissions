# deep-funding-mini-contest-pipeline

End-to-end workflow for our **Deep Funding mini-contest** entry on the Pond platform (model page 🔗: [https://cryptopond.xyz/modelfactory/detail/306250](https://cryptopond.xyz/modelfactory/detail/306250)).
The code pulls public OSS metrics, builds rich feature sets, trains a stacked model, emits a submission CSV **and** ships two visual layers:

1. A **UMAP viewer dataset** (`6_umap.py`) used in our write-up ([https://research.allo.capital/t/submission-of-entries-to-the-deep-funding-mini-contest/22/11](https://research.allo.capital/t/submission-of-entries-to-the-deep-funding-mini-contest/22/11)).
2. A **local Shiny dashboard** (`app.R`) that lets you explore the same data interactively.

Everything below reflects the actual code—no extra features are described.

---

## 1 · Solution at a glance

| Phase               | What happens                                                                                              |
| ------------------- | --------------------------------------------------------------------------------------------------------- |
| **Data collection** | GitHub REST + OSO BigQuery metrics for every repo in *train ∪ test*.                                      |
| **Embeddings**      | Shallow-clone each repo, embed docs with **Ollama / nomic-embed-text**.                                   |
| **Features**        | >1 000 engineered features: ratios, logs, one-hots, embedding slices.                                     |
| **Model**           | Stacked ensemble (XGB + RF + GB + SVR → XGB meta-learner) predicting `weight_a`.                          |
| **Visuals**         | `6_umap.py` builds `UMAPData.csv`; `app.R` renders an interactive UMAP plot with percentile highlighting. |

Compliance ✓: no external funding labels—only public metadata and embeddings feed the model.

---

## 2 · Repository layout

| File                        | Purpose                                                                        |
| --------------------------- | ------------------------------------------------------------------------------ |
| `0_get_github_stats_oso.py` | Query OSO BigQuery → `repostatsoso_df.csv`.                                    |
| `1_get_github_stats.py`     | GitHub REST API → `repostats_df.csv`.                                          |
| `2_get_github_repo.py`      | Clone + embed → `repoemb_df.csv`.                                              |
| `3_create_features.py`      | Merge & engineer features → `trainfeatures.csv`, `testfeatures.csv`.           |
| `5_fit_model.py`            | Train stacked model → **`sub_stacked.csv`** (contest submission).              |
| `6_umap.py`                 | Build **`HuggingFaceData/UMAPData.csv`** (input for external UMAP demo).       |
| **`app.R`**                 | Shiny dashboard for local interactive exploration of **`FundingMapData.csv`**. |

### Generated folders

```
CryptoPondData/
├── dataset.csv               ← contest train pairs (add manually)
├── test.csv                  ← contest test  pairs (add manually)
├── repos/                    ← git clones
├── repostats/                ← GitHub API JSON
├── repostats_df.csv
├── repostatsoso_df.csv
├── repoemb_df.csv
├── trainfeatures.csv
├── testfeatures.csv
└── sub_stacked.csv
HuggingFaceData/
└── UMAPData.csv              ← via 6_umap.py
shiny/
└── www/server_Icon1.png      ← logo used by app.R (create folder if absent)
```

---

## 3 · Quick-start (Python pipeline)

```bash
# 1 · Install Python deps
pip install -r requirements.txt
#    (needs git, ollama running on :9000, and gcloud SDK or a BigQuery
#     service-account JSON for OSO queries)

# 2 · Credentials
export GITHUB_TOKEN=ghp_yourtoken
export GOOGLE_APPLICATION_CREDENTIALS=oso_gcp_credentials.json

# 3 · Run end-to-end
python 0_get_github_stats_oso.py
python 1_get_github_stats.py
python 2_get_github_repo.py
python 3_create_features.py
python 5_fit_model.py        # → CryptoPondData/sub_stacked.csv

# 4 · (Optional) build the UMAP dataset for external demo
python 6_umap.py             # → HuggingFaceData/UMAPData.csv
```

---

## 4 · Shiny dashboard (`app.R`)

### What it does

* Loads **`FundingMapData.csv`** (same shape as `UMAPData.csv` but named for the app).
* Lets you choose **any two numeric columns** (defaults `UMAP1/UMAP2`) for the axes.
* Lets you highlight points above/below a selectable **percentile range** on a third metric.
* Uses **ggplot2 + plotly** for an interactive scatter; hovering shows repo name & the chosen metric value.
* Overlays a logo image (`www/server_Icon1.png`) in the bottom-right corner of the plot.

### Run it

```bash
# Install R packages once
R -e "install.packages(c('shiny','ggplot2','dplyr','plotly','RCurl'))"

# Ensure dataset & logo exist:
#   FundingMapData.csv             ← in the same working directory as app.R
#   shiny/www/server_Icon1.png     ← create this path and drop your PNG

# Launch the app
R -e "shiny::runApp('app.R', launch.browser = TRUE)"
```

### File expectations

* **`FundingMapData.csv`** must contain at least the columns referenced in the code:
  `SumWeights`, `UMAP1`, `UMAP2`, `Size`, `StarCount`, `Forks`, `IssueCount`.
  (You can symlink or copy `HuggingFaceData/UMAPData.csv` if you prefer.)
* **Logo**: any 100 × 100 px PNG is fine; the app base-64 encodes it for Plotly.

---

## 5 · UMAP dataset builder (`6_umap.py`)

1. Enumerates every distinct repo in train ∪ test.
2. Creates all pairwise feature rows and scores them with the trained XGB backbone.
3. Sums predicted weights per repo (`SumWeights`) and appends to embeddings.
4. Writes `HuggingFaceData/UMAPData.csv`—ready for Shiny, HF Spaces, or any UMAP/TSNE viewer.

---

## 6 · Model highlights

* **Transitivity-safe**: per-repo scoring → derive pairwise weights, so `A<B<B<C ⇒ A<C`.
* **Rich features**: ratios (A/B), logs, categorical encodings, embedding slices.
* **Stacked ensemble** consistently beats single models in 5-fold CV (MSE).
* **Reproducible**: cached artefacts allow fast re-runs; scripts are idempotent.

For further discussion, ablation tables, and leaderboard scores, see the forum write-up.

