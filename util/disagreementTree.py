import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle
import joblib
from sklearn.metrics import mean_absolute_error

from sklearn.tree import _tree
OUTPUT_DIR = Path("disagreement_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def get_leaf_rules(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = {}

    def recurse(node, path):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            # left child
            recurse(tree_.children_left[node],
                    path + [f"{name} <= {threshold:.2f}"])

            # right child
            recurse(tree_.children_right[node],
                    path + [f"{name} > {threshold:.2f}"])
        else:
            # leaf
            paths[node] = path

    recurse(0, [])
    return paths


def encode_categorical_states(df, state_cols):
    for col in state_cols:
        if df[col].dtype == object:
            df[col] = pd.Categorical(df[col]).codes.astype(float)
    return df


def get_split_features(tree, feature_names, model, knob, stage):
    """Return a list of dicts describing every internal split in *tree*."""
    tree_ = tree.tree_
    splits = []
    for nid in range(tree_.node_count):
        if tree_.feature[nid] != _tree.TREE_UNDEFINED:
            splits.append({
                "model": model,
                "knob": knob,
                "stage": stage,
                "feature": feature_names[tree_.feature[nid]],
                "threshold": round(float(tree_.threshold[nid]), 4),
                "n_node_samples": int(tree_.n_node_samples[nid]),
            })
    return splits


import warnings
warnings.filterwarnings('ignore')

# Load log files
DATA_DIR = Path(".")
mlp_logs_dir = DATA_DIR / "Logs/MLP"
tabPFN_logs_dir = DATA_DIR / "Logs/TabPFN"
xgboost_logs_dir = DATA_DIR / "Logs/XGBoost"

mlp_log_files = sorted(mlp_logs_dir.glob("mlp_ece_inputs_*.csv"))
tabPFN_log_files = sorted(tabPFN_logs_dir.glob("tabpfn_ece_inputs_*.csv"))
xgboost_log_files = sorted(xgboost_logs_dir.glob("xgboost_lopo_preds_for_ece_*.csv"))

KNOBS = ["PO2", "PCO2", "SpO2", "FiO2"]


# Load MLP scaler in-memory (no files written)
_mlp_scaler = None
_mlp_state_features = None
if mlp_log_files:
    _ts = "_".join(mlp_log_files[0].stem.split("_")[-2:])
    _mlp_model_dir = DATA_DIR / "models" / f"mlp_models_{_ts}"
    if _mlp_model_dir.exists():
        _mlp_scaler = joblib.load(_mlp_model_dir / "scaler_full.pkl")
        with open(_mlp_model_dir / "metadata.json") as _f:
            _meta = json.load(_f)
        _mlp_state_features = _meta[next(iter(_meta))]["state_features"]
        print(f"Loaded MLP scaler from {_mlp_model_dir}")
    else:
        print(f"Warning: MLP model dir not found at {_mlp_model_dir}, skipping descale")


all_splits = []  

# ============================================================================
# MLP SECTION
# ============================================================================
if mlp_log_files:
    # print(mlp_log_files.length)
    print(f"-" * 80)
    print(f"MLP Disagreement Trees")
    print(f"-" * 80)

    mlp_metrics_a_list = []
    mlp_leaf_a_list = []
    mlp_rules_a_dict = {}
    mlp_worst_leaf_a_dict = {}
    for file in mlp_log_files:
        mlp_df = pd.read_csv(file)
        if _mlp_scaler is not None and _mlp_state_features is not None:
            csv_state_cols = [c for c in mlp_df.columns if c.startswith("state_")]
            if csv_state_cols == _mlp_state_features:
                n = len(_mlp_state_features)
                mlp_df[_mlp_state_features] = (
                    mlp_df[_mlp_state_features].to_numpy()
                    * _mlp_scaler.scale_[:n]
                    + _mlp_scaler.mean_[:n]
                )
            else:
                print(f"  Warning: state column mismatch for {file.name}, skipping descale")
        y_true_binary = np.where(mlp_df['y_true'] == 0, 0, 1)

        mlp_df['error_stage_a'] = y_true_binary - mlp_df['p_change']
        mlp_df['error_stage_a_abs'] = np.abs(mlp_df['error_stage_a'])
        mlp_df['NLL_stage_a'] = - (y_true_binary * np.log(mlp_df['p_change'] + 1e-15) + (1 - y_true_binary) * np.log(1 - mlp_df['p_change'] + 1e-15))

        state_features = [col for col in mlp_df.columns if col.startswith('state_')]
        mlp_df = encode_categorical_states(mlp_df, state_features)

        head = file.stem.split("_")[3]
        print(f"Training disagreement trees for {head}")
        print(f"{'-'*80}\n")

        unique_patients = mlp_df['fold_test_patient'].unique()
        np.random.seed(42)
        test_patients = np.random.choice(unique_patients, size=min(15, len(unique_patients)), replace=False)
        train_patients = np.setdiff1d(unique_patients, test_patients)

        mlp_train = mlp_df[mlp_df['fold_test_patient'].isin(train_patients)].reset_index(drop=True)
        mlp_test = mlp_df[mlp_df['fold_test_patient'].isin(test_patients)].reset_index(drop=True)

        print(f"Patient split: {len(train_patients)} patients for training ({len(mlp_train)} samples), {len(test_patients)} patients for testing ({len(mlp_test)} samples)\n")

        X_train = mlp_train[state_features].values
        X_test = mlp_test[state_features].values

        # ===== STAGE A TREE =====
        y_train_a = mlp_train['error_stage_a_abs'].values
        y_test_a = mlp_test['error_stage_a_abs'].values

        tree_stage_a = DecisionTreeRegressor(random_state=42, max_depth=3, min_samples_leaf=100)
        tree_stage_a.fit(X_train, y_train_a)
        train_score_a = tree_stage_a.score(X_train, y_train_a)
        test_score_a = tree_stage_a.score(X_test, y_test_a)

        pred_a = tree_stage_a.predict(X_test)
        all_splits.extend(get_split_features(tree_stage_a, state_features, "MLP", head, "A"))

        # save
        metrics_a = {
            "model": "MLP",
            "knob": head,
            "stage": "A",
            "train_r2": train_score_a,
            "test_r2": test_score_a,
            "test_mae": mean_absolute_error(y_test_a, pred_a),
            "n_train": len(y_train_a),
            "n_test": len(y_test_a)
        }

        mlp_metrics_a_list.append(metrics_a)

        print(f"  Stage A Tree:")
        print(f"    Test MAE: {mean_absolute_error(y_test_a, pred_a):.4f}")
        print(f"    Train R²: {train_score_a:.4f}")
        print(f"    Test R²:  {test_score_a:.4f}")

        # Export tree rules for Stage A
        tree_rules_a = export_text(tree_stage_a, feature_names=state_features)
        print(f"    Rules:\n{tree_rules_a}\n")
        leaf_rules = get_leaf_rules(tree_stage_a, state_features)

        # Find disagreement hotspots for Stage A
        leaf_id_a = tree_stage_a.apply(X_test)
        test_data_a = mlp_test.copy()

        test_data_a["leaf"] = leaf_id_a
        test_data_a["sev"] = y_test_a
        test_data_a["signed"] = mlp_test["error_stage_a"].values
        test_data_a["abs_signed"] = np.abs(test_data_a["signed"])

        leaf_summary_a = test_data_a.groupby("leaf").agg(
            count=("sev", "count"),
            mean_sev=("sev", "mean"),
            median_sev=("sev", "median"),
            mean_signed=("signed", "mean"),
            mean_abs_signed=("abs_signed", "mean"),
        )
        leaf_summary_a["impact"] = leaf_summary_a["count"] * leaf_summary_a["mean_sev"]
        leaf_summary_a = leaf_summary_a.sort_values("impact", ascending=False)

        top2_ids_a = leaf_summary_a.index[:2].tolist()

        for rank, lid in enumerate(top2_ids_a, 1):
            print(f"\nWorst disagreement leaf #{rank} ({lid}):")
            for cond in leaf_rules[lid]:
                print("  ", cond)

        # ===============================
        # SAVE LEAF SUMMARY
        # ===============================
        leaf_summary_a_reset = leaf_summary_a.reset_index()
        leaf_summary_a_reset["model"] = "MLP"
        leaf_summary_a_reset["knob"] = head
        leaf_summary_a_reset["stage"] = "A"

        mlp_leaf_a_list.append(leaf_summary_a_reset)
        # ===============================
        # ACCUMULATE RULES
        # ===============================
        mlp_rules_a_dict[head] = tree_rules_a
        mlp_worst_leaf_a_dict[head] = [(lid, leaf_rules[lid], leaf_summary_a.loc[lid]) for lid in top2_ids_a]

        print(f"    Disagreement Hotspots (Stage A):")
        print(leaf_summary_a.head(15))

    # ===============================
    # SAVE COMBINED MLP FILES
    # ===============================
    if mlp_metrics_a_list:
        pd.DataFrame(mlp_metrics_a_list).to_csv(OUTPUT_DIR / "metrics_MLP_StageA.csv", index=False)
    if mlp_leaf_a_list:
        pd.concat(mlp_leaf_a_list, ignore_index=True).to_csv(OUTPUT_DIR / "leaf_summary_MLP_StageA.csv", index=False)
    if mlp_rules_a_dict:
        with open(OUTPUT_DIR / "rules_MLP_StageA.txt", "w") as f:
            for knob, rules in mlp_rules_a_dict.items():
                f.write(f"=== {knob} ===\n{rules}\n")
                if knob in mlp_worst_leaf_a_dict:
                    for rank, (lid, conds, stats) in enumerate(mlp_worst_leaf_a_dict[knob], 1):
                        impact = stats['count'] * stats['mean_sev']
                        f.write(f"  Worst disagreement leaf #{rank} ({lid}):\n")
                        for c in conds:
                            f.write(f"    {c}\n")
                        f.write(f"    count={stats['count']:.0f}  mean_sev={stats['mean_sev']:.4f}  mean_signed={stats['mean_signed']:.4f}  impact(count*mean_sev)={impact:.2f}\n")
                    f.write("\n")
    print(f"\nMLP combined files saved to {OUTPUT_DIR}/")


# ============================================================================
# TabPFN SECTION
# ============================================================================
if tabPFN_log_files:
    print(f"-" * 80)
    print(f"TabPFN Disagreement Trees")
    print(f"-" * 80)

    tabpfn_metrics_a_list = []
    tabpfn_leaf_a_list = []
    tabpfn_rules_a_dict = {}
    tabpfn_worst_leaf_a_dict = {}

    for file in tabPFN_log_files:
        tabpfn_df = pd.read_csv(file)
        y_true_binary = np.where(tabpfn_df['y_true'] == 0, 0, 1)

        tabpfn_df['error_stage_a'] = y_true_binary - tabpfn_df['p_change']
        tabpfn_df['error_stage_a_abs'] = np.abs(tabpfn_df['error_stage_a'])
        tabpfn_df['NLL_stage_a'] = - (y_true_binary * np.log(tabpfn_df['p_change'] + 1e-15) + (1 - y_true_binary) * np.log(1 - tabpfn_df['p_change'] + 1e-15))

        state_features = [col for col in tabpfn_df.columns if col.startswith('state_')]
        tabpfn_df = encode_categorical_states(tabpfn_df, state_features)

        head = file.stem.split("_")[3]
        print(f"Training disagreement trees for {head}")
        print(f"{'-'*80}\n")

        unique_patients = tabpfn_df['fold_test_patient'].unique()
        np.random.seed(42)
        test_patients = np.random.choice(unique_patients, size=min(15, len(unique_patients)), replace=False)
        train_patients = np.setdiff1d(unique_patients, test_patients)

        tabpfn_train = tabpfn_df[tabpfn_df['fold_test_patient'].isin(train_patients)].reset_index(drop=True)
        tabpfn_test = tabpfn_df[tabpfn_df['fold_test_patient'].isin(test_patients)].reset_index(drop=True)

        print(f"Patient split: {len(train_patients)} patients for training ({len(tabpfn_train)} samples), {len(test_patients)} patients for testing ({len(tabpfn_test)} samples)\n")

        X_train = tabpfn_train[state_features].values
        X_test = tabpfn_test[state_features].values

        # ===== STAGE A TREE =====
        y_train_a = tabpfn_train['error_stage_a_abs'].values
        y_test_a = tabpfn_test['error_stage_a_abs'].values

        tree_stage_a = DecisionTreeRegressor(random_state=42, max_depth=3, min_samples_leaf=100)
        tree_stage_a.fit(X_train, y_train_a)
        train_score_a = tree_stage_a.score(X_train, y_train_a)
        test_score_a = tree_stage_a.score(X_test, y_test_a)

        pred_a = tree_stage_a.predict(X_test)
        all_splits.extend(get_split_features(tree_stage_a, state_features, "TabPFN", head, "A"))

        # ===============================
        # SAVE STAGE A METRICS
        # ===============================
        metrics_a = {
            "model": "TabPFN",
            "knob": head,
            "stage": "A",
            "train_r2": train_score_a,
            "test_r2": test_score_a,
            "test_mae": mean_absolute_error(y_test_a, pred_a),
            "n_train": len(y_train_a),
            "n_test": len(y_test_a)
        }

        tabpfn_metrics_a_list.append(metrics_a)

        print(f"  Stage A Tree:")
        print(f"    Test MAE: {mean_absolute_error(y_test_a, pred_a):.4f}")
        print(f"    Train R²: {train_score_a:.4f}")
        print(f"    Test R²:  {test_score_a:.4f}")

        # Export tree rules for Stage A
        tree_rules_a = export_text(tree_stage_a, feature_names=state_features)
        print(f"    Rules:\n{tree_rules_a}\n")
        leaf_rules = get_leaf_rules(tree_stage_a, state_features)

        # Find disagreement hotspots for Stage A
        leaf_id_a = tree_stage_a.apply(X_test)
        test_data_a = tabpfn_test.copy()

        test_data_a["leaf"] = leaf_id_a
        test_data_a["sev"] = y_test_a
        test_data_a["signed"] = tabpfn_test["error_stage_a"].values
        test_data_a["abs_signed"] = np.abs(test_data_a["signed"])

        leaf_summary_a = test_data_a.groupby("leaf").agg(
            count=("sev", "count"),
            mean_sev=("sev", "mean"),
            median_sev=("sev", "median"),
            mean_signed=("signed", "mean"),
            mean_abs_signed=("abs_signed", "mean"),
        )
        leaf_summary_a["impact"] = leaf_summary_a["count"] * leaf_summary_a["mean_sev"]
        leaf_summary_a = leaf_summary_a.sort_values("impact", ascending=False)

        top2_ids_a = leaf_summary_a.index[:2].tolist()

        for rank, lid in enumerate(top2_ids_a, 1):
            print(f"\nWorst disagreement leaf #{rank} ({lid}):")
            for cond in leaf_rules[lid]:
                print("  ", cond)

        # ===============================
        # SAVE LEAF SUMMARY
        # ===============================
        leaf_summary_a_reset = leaf_summary_a.reset_index()
        leaf_summary_a_reset["model"] = "TabPFN"
        leaf_summary_a_reset["knob"] = head
        leaf_summary_a_reset["stage"] = "A"

        tabpfn_leaf_a_list.append(leaf_summary_a_reset)

        # ===============================
        # ACCUMULATE RULES
        # ===============================
        tabpfn_rules_a_dict[head] = tree_rules_a
        tabpfn_worst_leaf_a_dict[head] = [(lid, leaf_rules[lid], leaf_summary_a.loc[lid]) for lid in top2_ids_a]

        print(f"    Disagreement Hotspots (Stage A):")
        print(leaf_summary_a.head(15))

    # ===============================
    # SAVE COMBINED TabPFN FILES
    # ===============================
    if tabpfn_metrics_a_list:
        pd.DataFrame(tabpfn_metrics_a_list).to_csv(OUTPUT_DIR / "metrics_TabPFN_StageA.csv", index=False)
    if tabpfn_leaf_a_list:
        pd.concat(tabpfn_leaf_a_list, ignore_index=True).to_csv(OUTPUT_DIR / "leaf_summary_TabPFN_StageA.csv", index=False)
    if tabpfn_rules_a_dict:
        with open(OUTPUT_DIR / "rules_TabPFN_StageA.txt", "w") as f:
            for knob, rules in tabpfn_rules_a_dict.items():
                f.write(f"=== {knob} ===\n{rules}\n")
                if knob in tabpfn_worst_leaf_a_dict:
                    for rank, (lid, conds, stats) in enumerate(tabpfn_worst_leaf_a_dict[knob], 1):
                        impact = stats['count'] * stats['mean_sev']
                        f.write(f"  Worst disagreement leaf #{rank} ({lid}):\n")
                        for c in conds:
                            f.write(f"    {c}\n")
                        f.write(f"    count={stats['count']:.0f}  mean_sev={stats['mean_sev']:.4f}  mean_signed={stats['mean_signed']:.4f}  impact(count*mean_sev)={impact:.2f}\n")
                    f.write("\n")
    print(f"\nTabPFN combined files saved to {OUTPUT_DIR}/")


# ============================================================================
# XGBoost SECTION
# ============================================================================
print(f"-" * 80)
print(f"XGBoost Disagreement Trees")
print(f"-" * 80)

xgb_metrics_a_list = []
xgb_leaf_a_list = []
xgb_rules_a_dict = {}
xgb_worst_leaf_a_dict = {}

if xgboost_log_files:
    for file in xgboost_log_files:
        xgb_file = pd.read_csv(file)
        y_true_binary = np.where(xgb_file['y_true'] == 0, 0, 1)

        xgb_file['error_stage_a'] = y_true_binary - xgb_file['p_change']
        xgb_file['error_stage_a_abs'] = np.abs(xgb_file['error_stage_a'])
        xgb_file['NLL_stage_a'] = - (y_true_binary * np.log(xgb_file['p_change'] + 1e-15) + (1 - y_true_binary) * np.log(1 - xgb_file['p_change'] + 1e-15))

        state_features = [col for col in xgb_file.columns if col.startswith('state_')]
        xgb_file = encode_categorical_states(xgb_file, state_features)

        head = file.stem.split("_")[5]
        # Train decision trees for each model and knob
        print(f"Training disagreement trees for {head}")
        print(f"{'-'*80}\n")

        unique_patients = xgb_file['test_patient'].unique()
        np.random.seed(42)
        test_patients = np.random.choice(unique_patients, size=min(15, len(unique_patients)), replace=False)
        train_patients = np.setdiff1d(unique_patients, test_patients)

        xgb_train = xgb_file[xgb_file['test_patient'].isin(train_patients)].reset_index(drop=True)
        xgb_test = xgb_file[xgb_file['test_patient'].isin(test_patients)].reset_index(drop=True)

        print(f"Patient split: {len(train_patients)} patients for training ({len(xgb_train)} samples), {len(test_patients)} patients for testing ({len(xgb_test)} samples)\n")

        X_train = xgb_train[state_features].values
        X_test = xgb_test[state_features].values

        # ===== STAGE A TREE =====
        y_train_a = xgb_train['error_stage_a_abs'].values
        y_test_a = xgb_test['error_stage_a_abs'].values

        tree_stage_a = DecisionTreeRegressor(random_state=42, max_depth=3, min_samples_leaf=100)
        tree_stage_a.fit(X_train, y_train_a)
        train_score_a = tree_stage_a.score(X_train, y_train_a)
        test_score_a = tree_stage_a.score(X_test, y_test_a)

        pred_a = tree_stage_a.predict(X_test)
        all_splits.extend(get_split_features(tree_stage_a, state_features, "XGBoost", head, "A"))

        # ===============================
        # SAVE STAGE A METRICS
        # ===============================
        metrics_a = {
            "model": "XGBoost",
            "knob": head,
            "stage": "A",
            "train_r2": train_score_a,
            "test_r2": test_score_a,
            "test_mae": mean_absolute_error(y_test_a, pred_a),
            "n_train": len(y_train_a),
            "n_test": len(y_test_a)
        }

        xgb_metrics_a_list.append(metrics_a)

        print(f"  Stage A Tree:")
        print(f"    Test MAE: {mean_absolute_error(y_test_a, pred_a):.4f}")
        print(f"    Train R²: {train_score_a:.4f}")
        print(f"    Test R²:  {test_score_a:.4f}")

        # Export tree rules for Stage A
        tree_rules_a = export_text(tree_stage_a, feature_names=state_features)
        print(f"    Rules:\n{tree_rules_a}\n")
        leaf_rules = get_leaf_rules(tree_stage_a, state_features)

        # Find disagreement hotspots for Stage A
        leaf_id_a = tree_stage_a.apply(X_test)
        test_data_a = xgb_test.copy()

        test_data_a["leaf"] = leaf_id_a
        test_data_a["sev"] = y_test_a
        test_data_a["signed"] = xgb_test["error_stage_a"].values
        test_data_a["abs_signed"] = np.abs(test_data_a["signed"])

        leaf_summary_a = test_data_a.groupby("leaf").agg(
            count=("sev", "count"),
            mean_sev=("sev", "mean"),
            median_sev=("sev", "median"),
            mean_signed=("signed", "mean"),
            mean_abs_signed=("abs_signed", "mean"),
        )
        leaf_summary_a["impact"] = leaf_summary_a["count"] * leaf_summary_a["mean_sev"]
        leaf_summary_a = leaf_summary_a.sort_values("impact", ascending=False)

        top2_ids_a = leaf_summary_a.index[:2].tolist()

        for rank, lid in enumerate(top2_ids_a, 1):
            print(f"\nWorst disagreement leaf #{rank} ({lid}):")
            for cond in leaf_rules[lid]:
                print("  ", cond)

        # ===============================
        # SAVE LEAF SUMMARY
        # ===============================
        leaf_summary_a_reset = leaf_summary_a.reset_index()
        leaf_summary_a_reset["model"] = "XGBoost"
        leaf_summary_a_reset["knob"] = head
        leaf_summary_a_reset["stage"] = "A"

        xgb_leaf_a_list.append(leaf_summary_a_reset)

        # ===============================
        # ACCUMULATE RULES
        # ===============================
        xgb_rules_a_dict[head] = tree_rules_a
        xgb_worst_leaf_a_dict[head] = [(lid, leaf_rules[lid], leaf_summary_a.loc[lid]) for lid in top2_ids_a]

        print(f"    Disagreement Hotspots (Stage A):")
        print(leaf_summary_a.head(15))

# ===============================
# SAVE COMBINED XGBoost FILES
# ===============================
if xgb_metrics_a_list:
    pd.DataFrame(xgb_metrics_a_list).to_csv(OUTPUT_DIR / "metrics_XGB_StageA.csv", index=False)
if xgb_leaf_a_list:
    pd.concat(xgb_leaf_a_list, ignore_index=True).to_csv(OUTPUT_DIR / "leaf_summary_XGB_StageA.csv", index=False)
if xgb_rules_a_dict:
    with open(OUTPUT_DIR / "rules_XGB_StageA.txt", "w") as f:
        for knob, rules in xgb_rules_a_dict.items():
            f.write(f"=== {knob} ===\n{rules}\n")
            if knob in xgb_worst_leaf_a_dict:
                for rank, (lid, conds, stats) in enumerate(xgb_worst_leaf_a_dict[knob], 1):
                    impact = stats['count'] * stats['mean_sev']
                    f.write(f"  Worst disagreement leaf #{rank} ({lid}):\n")
                    for c in conds:
                        f.write(f"    {c}\n")
                    f.write(f"    count={stats['count']:.0f}  mean_sev={stats['mean_sev']:.4f}  mean_signed={stats['mean_signed']:.4f}  impact(count*mean_sev)={impact:.2f}\n")
                f.write("\n")
print(f"\nXGBoost combined files saved to {OUTPUT_DIR}/")

print("\nDone!")





# ============================================================================
# PREDICATE FREQUENCY SUMMARY
# ============================================================================
if all_splits:
    splits_df = pd.DataFrame(all_splits)

    feat_freq = splits_df["feature"].value_counts().reset_index()
    feat_freq.columns = ["feature", "total_splits"]

    model_freq = splits_df.groupby(["feature", "model"]).size().unstack(fill_value=0)
    model_freq["total"] = model_freq.sum(axis=1)
    model_freq = model_freq.sort_values("total", ascending=False)
    feat_freq.to_csv(OUTPUT_DIR / "predicate_frequency_overall.csv", index=False)
    model_freq.to_csv(OUTPUT_DIR / "predicate_frequency_by_model.csv")
    splits_df.to_csv(OUTPUT_DIR / "predicate_splits_detail.csv", index=False)

