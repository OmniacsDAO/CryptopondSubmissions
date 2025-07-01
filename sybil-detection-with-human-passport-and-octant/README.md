# Sybil Detection with Human Passport and Octant 🛡️🛂

> **Sniffing out Sybil wallets on Ethereum & Base for the Holonym × Octant challenge**
> 📂 Project page: [https://cryptopond.xyz/modelfactory/detail/4712551](https://cryptopond.xyz/modelfactory/detail/4712551)

---

## 🚀 TL;DR (3-step run)

```bash
# 1️⃣  Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2️⃣  Generate features (graph embeddings & all)
python features.py          # writes features_out/features.parquet

# 3️⃣  Train + predict
python model.py             # creates features_out/submission.csv
```

Upload `submission.csv` to the leaderboard, drop your write-up on the Octant forum by **31 May 2025**, and you’re in the prize race. 🏆

---

## 🔬 Under the hood

| Layer             | Highlights                                                                                                                                                                                                                                                               |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Feature Forge** | • Tx / token / swap counts<br>• Ether & USD value stats<br>• Activity cadence, burstiness, night-owl ratio<br>• Gas-price CV & low-gas flags<br>• Counter-party diversity & ping-pong motifs<br>• 128-dim Node2Vec graph embeddings<br>• Cross-chain (ETH : Base) ratios |
| **Model Lab**     | LightGBM-GBDT • Optuna TPE (40 trials × 7 folds) • undersampled negatives • median-pooled hyper-params • full retrain for inference                                                                                                                                      |

---

## 🗂️ Repo map

```
.
├── utils.py          helper functions / aggregations
├── features.py       builds the big feature matrix
├── model.py          tuning, CV, final submission
├── competition_*     organiser parquet data  ← add locally
└── features_out/     auto-generated (features, model, submission)
```

*Datasets aren’t committed; copy them to `competition_4712551_{ethereum,base}/` or tweak the paths.*

---

## 🛠️ Tips & tricks

* **Low-RAM?** Reduce `walks_per_node` or `dims` in `graph_embeddings`.
* **GPU LightGBM?** Add `"device_type":"gpu"` to the params.
* **Want ensembles?** `features_out/feature_importance.csv` shows which columns matter most.

---

💚 Like this project? Support more like it with [$IACS](http://dexscreener.com/base/0xd4d742cc8f54083f914a37e6b0c7b68c6005a024) on Base — CA: 0x46e69Fa9059C3D5F8933CA5E993158568DC80EBf
