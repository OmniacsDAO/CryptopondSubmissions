import os, json, argparse, joblib, warnings
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
import optuna

warnings.simplefilter("ignore")

# ─────────────────────────────────────────────────────────
# Params
# ─────────────────────────────────────────────────────────
seed=42
rng = np.random.RandomState(seed)

# ─────────────────────────────────────────────────────────
# data
# ─────────────────────────────────────────────────────────
print("▶ loading parquet …")
ETH_DIR  = "competition_4712551_ethereum"
BASE_DIR = "competition_4712551_base"
train_eth = pd.read_parquet(f"{ETH_DIR}/train_addresses.parquet")
test_eth  = pd.read_parquet(f"{ETH_DIR}/test_addresses.parquet")
train_base = pd.read_parquet(f"{BASE_DIR}/train_addresses.parquet")
test_base  = pd.read_parquet(f"{BASE_DIR}/test_addresses.parquet")

# ❶ concat & de-duplicate
train_df = (
    pd.concat([train_eth, train_base], ignore_index=True)
      .drop_duplicates(subset="ADDRESS", keep="first")
      .reset_index(drop=True)
)

## Ben2k potential
ben2k = set(pd.read_csv("https://pond-open-files.s3.us-east-1.amazonaws.com/frontier/others/01OHhiW4/false_negatives_Ben2k.csv")['Address'].str.lower())
train_df.loc[train_df['ADDRESS'].str.lower().isin(ben2k), 'LABEL'] = 1

test_df = (
    pd.concat([test_eth, test_base], ignore_index=True)
      .drop_duplicates(subset="ADDRESS")
      .reset_index(drop=True)
)

X = pd.read_parquet("features_out/features.parquet")
missing_train = set(train_df.ADDRESS) - set(X.index)
missing_test  = set(test_df.ADDRESS)  - set(X.index)
assert not missing_train, f"missing features for {len(missing_train)} train wallets"
assert not missing_test,  f"missing features for {len(missing_test)} test wallets"
X_train  = X.loc[train_df.ADDRESS]
y_train  = train_df.LABEL.values
y_train = train_df.LABEL.astype(int).to_numpy(dtype=int)
np.bincount(y_train)
X_test   = X.loc[test_df.ADDRESS]
del X  # free memory


# ─────────────────────────────────────────────────────────
# optuna objective
# ─────────────────────────────────────────────────────────
def undersample(X, y, ratio=1.0, seed=42):
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    np.random.seed(seed)
    sampled_neg_idx = np.random.choice(neg_idx, size=int(len(pos_idx) * ratio), replace=False)
    idx = np.concatenate([pos_idx, sampled_neg_idx])
    np.random.shuffle(idx)
    return X.iloc[idx], y[idx]

def make_objective(X_tr, y_tr):
    X_tr_sub, X_val_sub, y_tr_sub, y_val_sub = train_test_split(X_tr, y_tr, test_size=0.10, stratify=y_tr, random_state=seed)
    # X_tr_sub, y_tr_sub = undersample(X_tr_sub, y_tr_sub, ratio=3.0, seed=seed)
    lgb_tr = lgb.Dataset(X_tr_sub, y_tr_sub)
    lgb_va = lgb.Dataset(X_val_sub, y_val_sub)
    def objective(trial):
        param = {
            "objective":         "binary",
            "is_unbalance": True,
            "metric":            "auc",
            "verbosity":         -1,
            "boosting_type":     "gbdt",
            "n_estimators":      10000,
            "learning_rate":     trial.suggest_float("lr", 0.01, 0.1, log=True),
            "num_leaves":        trial.suggest_int("leaves", 31, 511, step=32),
            "max_depth":         trial.suggest_int("depth", -1, 20),
            "min_child_samples": trial.suggest_int("min_child", 20, 100),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample", 0.4, 1.0),
            "reg_lambda":        trial.suggest_float("l2", 1e-3, 100.0, log=True),
            "reg_alpha":        trial.suggest_float("l1", 1e-3, 10.0, log=True),
            "feature_pre_filter": False,
            "min_split_gain": 0.01,
            "seed": seed,
            "n_jobs": -1,
        }
        clf = lgb.train(
            param, lgb_tr,
            valid_sets=[lgb_va],
            callbacks=[early_stopping(100, verbose=False)],
        )
        return clf.best_score["valid_0"]["auc"]
    return objective

# ─────────────────────────────────────────────────────────
# CV loop
# ─────────────────────────────────────────────────────────
oof = np.zeros(len(X_train))
cv_report = []
skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=seed)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"Fold {fold}: train positives = {np.sum(y_train[tr_idx])}, val positives = {np.sum(y_train[va_idx])}")
    print(f"\n── Fold {fold} ──")
    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
    y_tr, y_va = y_train[tr_idx],  y_train[va_idx]
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(make_objective(X_tr, y_tr), n_trials=40, show_progress_bar=False)
    best_params = study.best_trial.params
    print("best AUC:", study.best_value, "\nparams:", best_params)
    # train model on full fold-train with best params
    best_params.update({
        "objective": "binary", "metric": "auc",
        "verbosity": -1, "boosting_type": "gbdt",
        "is_unbalance": True,
        "n_estimators": 10_000, "seed": seed, "n_jobs": -1
    })
    lgb_tr = lgb.Dataset(X_tr, y_tr)
    lgb_va = lgb.Dataset(X_va, y_va)
    clf = lgb.train(
        best_params, lgb_tr,
        valid_sets=[lgb_va],
        callbacks=[early_stopping(100, verbose=False)],
    )
    # save booster txt
    # clf.save_model(f"features_out/lgb_fold{fold}.txt")
    # out-of-fold preds
    oof[va_idx] = clf.predict(X_va, num_iteration=clf.best_iteration)
    auc_fold = roc_auc_score(y_va, oof[va_idx])
    print(f"fold-{fold} AUC: {auc_fold:.6f}")
    cv_report.append({
        "fold": fold,
        "auc": auc_fold,
        "best_iter": clf.best_iteration,
        "best_params": best_params,
    })

overall_auc = roc_auc_score(y_train, oof)
print(f"\n==== CV DONE – overall OOF AUC: {overall_auc:.6f} ====")
# np.save("features_out/oof_preds.npy", oof)

# ─────────────────────────────────────────────────────────
# retrain on all data with median-of-best params
# ─────────────────────────────────────────────────────────
best_params_list = [r["best_params"] for r in cv_report]
avg_best_iter = int(np.mean([r["best_iter"] for r in cv_report]))
df_params = pd.DataFrame(best_params_list)
numeric_medians = df_params.select_dtypes(include=[np.number]).median().to_dict()
median_params = {**best_params_list[0], **numeric_medians}
median_params["num_leaves"]        = int(median_params["leaves"])
median_params["max_depth"]         = int(median_params["depth"])
median_params["min_child_samples"] = int(median_params["min_child"])
median_params["n_estimators"] = 10000
median_params["n_estimators"] = avg_best_iter
median_params.update({
    "objective": "binary", "metric": "auc","is_unbalance": True,"min_split_gain": 0.01,
    "verbosity": -1, "boosting_type": "gbdt",
    "seed": seed, "n_jobs": -1
})
print("re-training on ALL data with params:", median_params)

lgb_all = lgb.Dataset(X_train, y_train)
clf_full = lgb.train(
    median_params, lgb_all,
    valid_sets=[lgb_all],
    callbacks=[log_evaluation(10)],
)
clf_full.save_model("features_out/lgb_full.txt")

# feature importance csv
imp = pd.DataFrame({
    "feature": X_train.columns,
    "gain": clf_full.feature_importance(importance_type="gain"),
    "split": clf_full.feature_importance(importance_type="split"),
}).sort_values("gain", ascending=False)
imp.to_csv("features_out/feature_importance.csv", index=False)

# ─────────────────────────────────────────────────────────
# inference → submission
# ─────────────────────────────────────────────────────────
test_pred = clf_full.predict(X_test, num_iteration=clf_full.best_iteration)
pd.DataFrame({"ADDRESS": test_df.ADDRESS, "PRED": test_pred}).to_csv("submission.csv", index=False, float_format="%.8f")

