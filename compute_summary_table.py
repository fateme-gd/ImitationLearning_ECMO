

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score, balanced_accuracy_score
from itertools import combinations


LOGS_DIR = Path("Logs")

# ECE input files (per-knob, latest timestamps)
ECE_FILES = {
    "MLP": sorted(LOGS_DIR.glob("MLP/mlp_ece_inputs_*_20260217_015336.csv")),
    "TabPFN": sorted(LOGS_DIR.glob("TabPFN/tabpfn_ece_inputs_*_20260216_201158.csv")),
    "XGBoost": sorted(LOGS_DIR.glob("XGBoost/xgboost_lopo_preds_for_ece_*_20260216_19*.csv")),
}

OUTPUT_DIR = Path("comparison_results_fixed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_ece(y_true, p3, n_bins=10):
    y_true = np.asarray(y_true, dtype=int)
    p3 = np.asarray(p3, dtype=float)
    conf = p3.max(axis=1)
    pred = p3.argmax(axis=1)
    correct = (pred == y_true).astype(float)

    valid = np.isfinite(conf) & np.isfinite(correct)
    conf, correct = conf[valid], correct[valid]
    if len(conf) == 0:
        return np.nan

    edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(conf, edges[1:-1], right=True)
    N = len(conf)
    ece = 0.0
    for b in range(n_bins):
        m = bin_idx == b
        nb = m.sum()
        if nb == 0:
            continue
        ece += (nb / N) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def compute_fold_metrics(y_true, y_pred, p3):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    bacc = balanced_accuracy_score(y_true, y_pred)
    ece = compute_ece(y_true, p3)
    return f1, bacc, ece

all_rows = []

for model_name, file_list in ECE_FILES.items():
    if not file_list:
        print(f"  [WARN] No ECE files for {model_name}")
        continue

    # merge across knobs 
    dfs = []
    for fp in file_list:
        d = pd.read_csv(fp)
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)

    if "fold_test_patient" in df.columns:
        fold_col = "fold_test_patient"
    elif "test_patient" in df.columns:
        fold_col = "test_patient"
    else:
        print(f"  [WARN] {model_name}: cannot find fold/patient column")
        continue

    patients = sorted(df[fold_col].unique())
    print(f"  {model_name}: {len(patients)} patients (folds), {len(df)} rows")

    for pat in patients:
        sub = df[df[fold_col] == pat]
        y_true = sub["y_true"].to_numpy().astype(int)
        y_pred = sub["y_pred"].to_numpy().astype(int)
        p3 = sub[["p_same", "p_decrease", "p_increase"]].to_numpy()  # normalized probs

        f1, bacc, ece = compute_fold_metrics(y_true, y_pred, p3)
        all_rows.append({
            "model": model_name,
            "patient": pat,
            "f1": f1,
            "acc": bacc,
            "ece": ece,
        })

fold_metrics = pd.DataFrame(all_rows)
print(f"\n  Total fold-metric rows: {len(fold_metrics)}")


models = sorted(fold_metrics["model"].unique())
summary_rows = []

for m in models:
    sub = fold_metrics[fold_metrics["model"] == m]
    row = {
        "Model": m,
        "F1 (mean)": sub["f1"].mean(),
        "F1 (std)": sub["f1"].std(),
        "Acc (mean)": sub["acc"].mean(),
        "Acc (std)": sub["acc"].std(),
        "ECE (mean)": sub["ece"].mean(),
        "ECE (std)": sub["ece"].std(),
    }
    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)


# Pairwise p-values
print("\nComputing pairwise Wilcoxon p-values …")

def paired_wilcoxon_pvalue(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    # mask = np.isfinite(a) & np.isfinite(b)
    # a, b = a[mask], b[mask]
    if len(a) < 5:
        return np.nan
    try:
        stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        return float(p)
    except Exception:
        return np.nan



# For each metric, compute pairwise p-values
pval_rows = []
for (m1, m2) in combinations(models, 2):
    sub1 = fold_metrics[fold_metrics["model"] == m1].sort_values("patient")
    sub2 = fold_metrics[fold_metrics["model"] == m2].sort_values("patient")

    merged = sub1.merge(sub2, on="patient", suffixes=("_a", "_b"))
    print(merged.head(10))

    print(f"\n  --- {m1} vs {m2}  ({len(merged)} paired patients) ---")

    for metric, col in [("F1", "f1"), ("Acc", "acc"), ("ECE", "ece")]:
        a = merged[f"{col}_a"].to_numpy()
        b = merged[f"{col}_b"].to_numpy()
        # mask = np.isfinite(a) & np.isfinite(b)
        # a, b = a[mask], b[mask]
        diff = a - b

        n_pos = (diff > 0).sum()
        n_neg = (diff < 0).sum()
        n_zero = (diff == 0).sum()

        print(f"    {metric:4s}:  n={len(diff)},  "
              f"{m1}>{m2}={n_pos},  {m1}<{m2}={n_neg},  tied={n_zero},  "
              f"mean_diff={diff.mean():.4f},  std_diff={diff.std():.4f}")

        sorted_diff = np.sort(diff)
        print(f"           min_diff={sorted_diff[0]:.4f}, "
              f"median_diff={np.median(diff):.4f}, "
              f"max_diff={sorted_diff[-1]:.4f}")

        p = paired_wilcoxon_pvalue(a, b)

        pval_rows.append({
            "metric": metric,
            "model_a": m1,
            "model_b": m2,
            "p_value": p,
            "n_pairs": len(diff),
            f"n_{m1}_wins": int(n_pos),
            f"n_{m2}_wins": int(n_neg),
            "n_tied": int(n_zero),
            "mean_diff": float(diff.mean()),
        })

pval_df = pd.DataFrame(pval_rows)

print("\nPairwise Wilcoxon p-values:")
print(pval_df.head(10))


for metric, col in [("F1", "F1 (p-value)"), ("Acc", "Acc (p-value)"), ("ECE", "ECE (p-value)")]:
    for m in models:
        relevant = pval_df[
            (pval_df["metric"] == metric) &
            ((pval_df["model_a"] == m) | (pval_df["model_b"] == m))
        ]

        pmin = relevant["p_value"].min()
        summary.loc[summary["Model"] == m, col] = pmin


col_order = [
    "Model",
    "F1 (mean)", "F1 (std)", "F1 (p-value)",
    "Acc (mean)", "Acc (std)", "Acc (p-value)",
    "ECE (mean)", "ECE (std)", "ECE (p-value)",
]
summary = summary[col_order].sort_values("F1 (mean)", ascending=False)

for pc in ["F1 (p-value)", "Acc (p-value)", "ECE (p-value)"]:
    summary[pc] = summary[pc].map(lambda x: f"{x:.4e}")

out_path = OUTPUT_DIR / "summary_f1_acc_ece.csv"
summary.to_csv(out_path, index=False, float_format="%.6f")

print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\nSaved to: {out_path}")

pval_path = OUTPUT_DIR / "pairwise_wilcoxon_pvalues.csv"
pval_df.to_csv(pval_path, index=False, float_format="%.6e")
print(f"\nPairwise p-values saved to: {pval_path}")
print(pval_df.to_string(index=False, float_format=lambda x: f"{x:.4e}"))
