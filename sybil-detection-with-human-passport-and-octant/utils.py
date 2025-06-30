import os, gc, argparse, pickle, joblib, warnings
import numpy as np
import pandas as pd
from datetime import timedelta
from tqdm import tqdm

# ──────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────
def read_parquet(fp, cols=None):
    return pd.read_parquet(fp, columns=cols)

def safe_join(base, block, addr_index):
    block = block.reindex(addr_index)      # keep only wanted rows
    if base.empty:
        # initialise with full index once
        return pd.DataFrame(index=addr_index).join(block, how="left")
    return base.join(block, how="left")

def nunique(series):
    return series.nunique(dropna=True)

def entropy(x):
    p = x.value_counts(normalize=True)
    return -(p * np.log2(p)).sum()

def qcut_bin(s, q=10):
    return pd.qcut(s, q, duplicates="drop", labels=False)

def agg(df, key, agg_dict, prefix):
    g = df.groupby(key, sort=False).agg(agg_dict)

    if isinstance(g.columns, pd.MultiIndex):
        # flatten ("VALUE","sum") → "VALUE_sum"
        g.columns = [f"{c[0]}_{c[1]}" for c in g.columns]
    else:
        g.columns = g.columns.astype(str)          # safety

    g.columns = [f"{prefix}_{c}" for c in g.columns]
    return g

# ──────────────────────────────────────────────
# 1.  basic activity counts
# ──────────────────────────────────────────────
# utils.py  – overwrite the originals
def tx_basic_counts(tx, from_col, to_col, prefix):
    sent = agg(tx, from_col, {"TX_HASH": nunique}, f"{prefix}_tx_sent_cnt")
    recv = agg(tx, to_col,   {"TX_HASH": nunique}, f"{prefix}_tx_recv_cnt")
    return sent.join(recv, how="outer")       # <-- no safe_join here

def token_basic_counts(tok, from_col, to_col, prefix):
    sent = agg(tok, from_col, {"TX_HASH": nunique}, f"{prefix}_trf_sent_cnt")
    recv = agg(tok, to_col,   {"TX_HASH": nunique}, f"{prefix}_trf_recv_cnt")
    uniq_sent = agg(tok, from_col, {"CONTRACT_ADDRESS": nunique},
                    f"{prefix}_uniq_token_sent")
    uniq_recv = agg(tok, to_col,   {"CONTRACT_ADDRESS": nunique},
                    f"{prefix}_uniq_token_recv")
    block = sent.join(recv, how="outer")
    return block.join(uniq_sent, how="outer").join(uniq_recv, how="outer")

def swap_basic_counts(sw, from_col, prefix):
    return agg(sw, from_col, {"TX_HASH": nunique}, f"{prefix}_swap_cnt")

# ──────────────────────────────────────────────
# 2.  monetary aggregates
# ──────────────────────────────────────────────
def eth_value_stats(tx, addr_col, prfx):
    feats = agg(
        tx.loc[tx["VALUE"] > 0, :],
        addr_col,
        {
            "VALUE": ["sum", "mean", "std", "max", "median"],
            "TX_FEE": ["sum", "mean"],
        },
        prfx
    )
    feats[f"{prfx}_eth_val_cv"] = feats[f"{prfx}_VALUE_std"] / (feats[f"{prfx}_VALUE_mean"] + 1e-9)
    feats[f"{prfx}_fee_ratio"] = feats[f"{prfx}_TX_FEE_sum"] / (feats[f"{prfx}_VALUE_sum"] + 1e-9)
    feats.drop(columns=[c for c in feats.columns if c.endswith("_std")], inplace=True)
    return feats

def token_value_stats(tok, addr_col, prfx):
    usd = tok.copy()
    usd["AMT_USD_SENT"] = np.where(usd["FROM_ADDRESS"] == tok[addr_col],
                                   usd["AMOUNT_USD"], np.nan)
    feats = agg(tok, addr_col, {"AMOUNT_USD": ["sum", "mean", "max"]}, prfx)
    return feats

# ──────────────────────────────────────────────
# 3.  temporal cadence
# ──────────────────────────────────────────────
def cadence_features(tx, addr_col, prfx):
    df = tx[[addr_col, "BLOCK_TIMESTAMP"]].sort_values("BLOCK_TIMESTAMP")
    g = df.groupby(addr_col)["BLOCK_TIMESTAMP"]
    first = g.min()
    last = g.max()
    today = pd.Timestamp.now(tz=first.dt.tz)  # match timezone awareness
    out = pd.DataFrame({
        f"{prfx}_first_seen": (today - first).dt.days,
        f"{prfx}_last_seen": (today - last).dt.days,
        f"{prfx}_active_days": g.apply(lambda x: nunique(x.dt.date)),
    })
    out[f"{prfx}_lifetime_days"] = out[f"{prfx}_last_seen"] - out[f"{prfx}_first_seen"]
    # inter-tx seconds stats
    inter = g.apply(lambda x: x.diff().dt.total_seconds().dropna())
    out[f"{prfx}_mean_inter_sec"] = inter.groupby(level=0).mean()
    out[f"{prfx}_std_inter_sec"]  = inter.groupby(level=0).std()
    out[f"{prfx}_burst_z"] = (
        out[f"{prfx}_std_inter_sec"] - out[f"{prfx}_mean_inter_sec"]
    ) / (out[f"{prfx}_std_inter_sec"] + out[f"{prfx}_mean_inter_sec"] + 1e-9)
    # entropy
    out[f"{prfx}_weekday_entropy"] = g.apply(lambda x: entropy(x.dt.dayofweek))
    out[f"{prfx}_hour_entropy"]    = g.apply(lambda x: entropy(x.dt.hour))
    # pct night
    out[f"{prfx}_night_ratio"] = g.apply(
        lambda x: (x.dt.hour.isin([0,1,2,3,4,5]).mean())
    )
    return out

# ──────────────────────────────────────────────
# 4.  gas / fee behaviour
# ──────────────────────────────────────────────
def gas_features(tx, from_col, prfx):
    feats = agg(
        tx,
        from_col,
        {
            "GAS_PRICE": ["mean", "median", "std"],
            "EFFECTIVE_GAS_PRICE": ["mean"],
            "GAS_USED": ["mean"],
        },
        prfx
    )
    feats[f"{prfx}_gas_price_cv"] = feats[f"{prfx}_GAS_PRICE_std"] / (
        feats[f"{prfx}_GAS_PRICE_mean"] + 1e-9
    )
    # low-gas counter
    low = tx.copy()
    low["is_low"] = low.groupby("BLOCK_NUMBER")["GAS_PRICE"].transform(
        lambda s: s < s.quantile(0.10)
    )
    low_flag = agg(
        low[low.is_low],
        from_col,
        {"TX_HASH": "count"},
        f"{prfx}_lowgas_cnt"
    )
    return feats.join(low_flag, how="outer")


# ──────────────────────────────────────────────
# 5.  counterparty diversity
# ──────────────────────────────────────────────
def counterparty_feats(tx, prfx, addr_index):
    sent = agg(tx, "FROM_ADDRESS", {"TO_ADDRESS": nunique}, f"{prfx}_uniq_ctr_ptys")
    recv = agg(tx, "TO_ADDRESS", {"FROM_ADDRESS": nunique}, f"{prfx}_uniq_src_ptys")
    return safe_join(sent, recv, addr_index)

# ──────────────────────────────────────────────
# 6.  DEX swap patterns
# ──────────────────────────────────────────────
def swap_feats(sw, prfx):
    g = sw.groupby("ORIGIN_FROM_ADDRESS")
    out = pd.DataFrame({
        f"{prfx}_swap_cnt": g["TX_HASH"].nunique(),
        f"{prfx}_swap_pair_div": g.apply(
            lambda x: pd.Series(list(zip(x["TOKEN_IN"], x["TOKEN_OUT"]))).nunique()
        ),
        f"{prfx}_swap_amt_usd": g["AMOUNT_IN_USD"].sum(min_count=1),
    })
    return out

# ──────────────────────────────────────────────
# 7.  Temporal motifs / ping-pong (simple version)
# ──────────────────────────────────────────────
def pingpong(tx, window_min=10):
    """count back-and-forth pairs within a window."""
    df = tx[["FROM_ADDRESS", "TO_ADDRESS", "BLOCK_TIMESTAMP"]].copy()
    df.sort_values("BLOCK_TIMESTAMP", inplace=True)
    df["key"] = list(zip(df.FROM_ADDRESS, df.TO_ADDRESS))
    lookup = df.set_index("key")["BLOCK_TIMESTAMP"].to_dict()
    pp = []
    for (a, b), t in lookup.items():
        if (b, a) in lookup:
            if abs((t - lookup[(b,a)]).total_seconds()) <= window_min*60:
                pp.append((a, b))
    return pd.Series({k[0]: 1 for k in pp})

# ──────────────────────────────────────────────
# 8.  Graph Embedding
# ──────────────────────────────────────────────
import pandas as pd
import networkx as nx
from nodevectors import Node2Vec

def graph_embeddings(tx, token, swaps, dims=128, walk_length=20, context_size=10, walks_per_node=10):
    # Create edge list
    edges = pd.concat([
        tx[["FROM_ADDRESS", "TO_ADDRESS"]],
        token[["FROM_ADDRESS", "TO_ADDRESS"]],
        swaps[["ORIGIN_FROM_ADDRESS", "TX_TO"]].rename(
            columns={"ORIGIN_FROM_ADDRESS": "FROM_ADDRESS", "TX_TO": "TO_ADDRESS"}
        ),
    ], ignore_index=True).dropna()

    # Keep only string addresses
    edges["FROM_ADDRESS"] = edges["FROM_ADDRESS"].astype(str)
    edges["TO_ADDRESS"] = edges["TO_ADDRESS"].astype(str)

    # Build undirected NetworkX graph
    G = nx.from_pandas_edgelist(edges, "FROM_ADDRESS", "TO_ADDRESS", create_using=nx.Graph())

    # Train Node2Vec using nodevectors (gensim-based, fast)
    n2v = Node2Vec(
        n_components=dims,
        walklen=walk_length,
        epochs=5,
        return_weight=1.0,
        neighbor_weight=1.0,
        threads=4,
        w2vparams={
            "window": context_size,
            "negative": 5,
            "min_count": 1,
            "batch_words": 4_000
        }
    )
    print("▶ Training Node2Vec embeddings…")
    n2v.fit(G)

    # Extract embeddings
    emb_matrix = n2v.model.wv.vectors
    node_ids = n2v.model.wv.index_to_key

    emb_df = pd.DataFrame(emb_matrix, index=node_ids)
    emb_df.index.name = "ADDRESS"
    emb_df.columns = [f"emb_{i}" for i in range(dims)]

    return emb_df


# ──────────────────────────────────────────────
# 9.  master feature builder per chain
# ──────────────────────────────────────────────
def build_chain_features(tx, token, swaps, addr_index, chain_tag):
    out = pd.DataFrame(index=addr_index)        # ← seed with fixed index

    # ─ basic counts
    out = safe_join(out,
                    tx_basic_counts(tx, "FROM_ADDRESS", "TO_ADDRESS", chain_tag),
                    addr_index)

    out = safe_join(out,
                    token_basic_counts(token, "FROM_ADDRESS", "TO_ADDRESS", chain_tag),
                    addr_index)

    out = safe_join(out,
                    swap_basic_counts(swaps, "ORIGIN_FROM_ADDRESS", chain_tag),
                    addr_index)

    # ─ money
    out = safe_join(out,
                    eth_value_stats(tx, "FROM_ADDRESS", f"{chain_tag}_out"),
                    addr_index)

    out = safe_join(out,
                    eth_value_stats(tx, "TO_ADDRESS",   f"{chain_tag}_in"),
                    addr_index)

    # ─ cadence
    out = safe_join(out,
                    cadence_features(tx, "FROM_ADDRESS", f"{chain_tag}_cad"),
                    addr_index)

    # ─ gas
    out = safe_join(out,
                    gas_features(tx, "FROM_ADDRESS", f"{chain_tag}_gas"),
                    addr_index)

    # ─ counterparties
    out = safe_join(out,
                counterparty_feats(tx, chain_tag, addr_index),
                addr_index)

    # ─ swaps
    out = safe_join(out,
                    swap_feats(swaps, chain_tag),
                    addr_index)

    # ─ ping-pong flag
    pp = pingpong(tx)
    out = safe_join(out,
                    pp.to_frame(f"{chain_tag}_pingpong_flag"),
                    addr_index)

    return out
