import os, gc, argparse, pickle, joblib, warnings
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from datetime import timedelta
from tqdm import tqdm
from utils import *


# ──────────────────────────────────────────────
# 10.  main
# ──────────────────────────────────────────────
print("▶  loading parquet …")
# Ethereum
eth_dir = "competition_4712551_ethereum"
train_e, test_e, tx_e, tok_e, sw_e = (
        read_parquet(os.path.join(eth_dir, f"{fn}.parquet"))
        for fn in ["train_addresses", "test_addresses", "transactions",
                   "token_transfers", "dex_swaps"]
)
# Base
base_dir = "competition_4712551_base"
train_b, test_b, tx_b, tok_b, sw_b = (
        read_parquet(os.path.join(base_dir, f"{fn}.parquet"))
        for fn in ["train_addresses", "test_addresses", "transactions",
                   "token_transfers", "dex_swaps"]
)

all_addrs = pd.Index(
        pd.concat([
            train_e.ADDRESS, test_e.ADDRESS,
            train_b.ADDRESS, test_b.ADDRESS
        ]).unique(), name="ADDRESS"
)
features = pd.DataFrame(index=all_addrs)

print("▶  building Ethereum feature block …")
feat_eth = build_chain_features(tx_e, tok_e, sw_e,all_addrs, "eth")
features = safe_join(features, feat_eth, all_addrs); del feat_eth; gc.collect()

print("▶  building Base feature block …")
feat_base = build_chain_features(tx_b, tok_b, sw_b,all_addrs, "base")
features = safe_join(features, feat_base, all_addrs); del feat_base; gc.collect()

# chain ratio features
for col in [c for c in features.columns if c.startswith("eth")]:
    cbase = col.replace("eth_", "base_")
    if cbase in features.columns:
        if is_datetime64_any_dtype(features[col]) or is_datetime64_any_dtype(features[cbase]):
            continue
        features[f"{col}_ratio"] = features[col] / (features[cbase] + 1e-9)


print("▶  computing graph embeddings (Node2Vec)… filtered to train/test addresses")
tx = pd.concat([tx_e, tx_b]);token = pd.concat([tok_e, tok_b]);swaps=pd.concat([sw_e, sw_b]);dims=128;walk_length=20;context_size=10;walks_per_node=10
emb = graph_embeddings(pd.concat([tx_e, tx_b]),pd.concat([tok_e, tok_b]),pd.concat([sw_e, sw_b]),)
features = safe_join(features, emb)

# housekeeping
features.replace([np.inf, -np.inf], np.nan, inplace=True)
features.fillna(0, inplace=True)

print("▶  saving parquet & metadata")
out_dir = "features_out"
features.to_parquet(os.path.join(out_dir, "features.parquet"))
meta = {"n_rows": len(features),"n_cols": len(features.columns),"build_graph": True,}
with open(os.path.join(out_dir, "meta.json"), "w") as fh:
    import json; json.dump(meta, fh, indent=2)

print("✓ done – matrix shape:", features.shape)
