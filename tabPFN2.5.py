import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import joblib

from tabpfn import TabPFNClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, confusion_matrix,
    recall_score, precision_score
)

KNOBS = [
    ("PO2",   "action_PO2 ARTERIAL (mmHg)_changed",   "action_PO2 ARTERIAL (mmHg)_direction"),
    ("PCO2",  "action_PCO2 ARTERIAL (mmHg)_changed",  "action_PCO2 ARTERIAL (mmHg)_direction"),
    ("SpO2",  "action_SpO2 (%)_changed",              "action_SpO2 (%)_direction"),
    ("FiO2",  "action_FiO2 - ECMO_changed",           "action_FiO2 - ECMO_direction"),
    # ("etCO2", "action_etCO2 (mmHg)_changed",          "action_etCO2 (mmHg)_direction"),
]

DIR_MAP = {"decrease": 0, "increase": 1}  


def choose_threshold_on_train(proba_val, y_val, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    best_thr, best_f1 = 0.5, -1.0
    for thr in thresholds:
        pred = (proba_val >= thr).astype(int)
        f1_pos = f1_score(y_val, pred, average="binary", zero_division=0)
        if f1_pos > best_f1:
            best_f1, best_thr = f1_pos, float(thr)
    return best_thr, float(best_f1)


def make_y3_from_truth(act, direction):
    y3 = np.zeros(len(act), dtype=int)
    mask = (act == 1)
    y3[mask] = np.array([1 if d == "decrease" else 2 for d in direction[mask]])
    return y3


def _safe_device_str(device):
    if device is None:
        return "cuda"
    return str(device).lower()


def fit_predict_two_stage_tabpfn(
    X_train, X_test, act_train, dir_train,
    tune_stageA_threshold=True,
    device="cuda",
    verbose=False,
):
    device = _safe_device_str(device)

    # -----------------
    # Stage A: change?
    # -----------------
    thr = 0.5
    best_f1_pos = 0.0

    if tune_stageA_threshold and len(np.unique(act_train)) > 1 and len(act_train) > 50:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr_idx, va_idx = next(sss.split(X_train, act_train))
        X_tr, y_tr = X_train[tr_idx], act_train[tr_idx]
        X_va, y_va = X_train[va_idx], act_train[va_idx]

        m_change = TabPFNClassifier(device=device)
        m_change.fit(X_tr, y_tr)

        proba_va = m_change.predict_proba(X_va)[:, 1]
        thr, best_f1_pos = choose_threshold_on_train(proba_va, y_va)

        if verbose:
            print(f"      Stage A: thr={thr:.2f} valF1={best_f1_pos:.3f}")

        p_change = m_change.predict_proba(X_test)[:, 1]
    else:
        m_change = TabPFNClassifier(device=device)
        m_change.fit(X_train, act_train)
        p_change = m_change.predict_proba(X_test)[:, 1]

    p_change = np.asarray(p_change, dtype=float)
    p_change = np.clip(p_change, 0.0, 1.0)
    p_change_hard = (p_change >= thr).astype(int)

    # --------------------------
    # Stage B: direction (act==1)
    # --------------------------
    tr_mask = (act_train == 1)
    if int(np.sum(tr_mask)) == 0:
        p_inc_given_change = np.full(len(X_test), 0.5, dtype=float)
        p3 = np.column_stack([
            1.0 - p_change,                         # same
            p_change * (1.0 - p_inc_given_change),   # decrease
            p_change * p_inc_given_change            # increase
        ])
        y3 = np.zeros(len(X_test), dtype=int)
        return {
            "y3_pred": y3,
            "thr": float(thr),
            "stageA_val_f1": float(best_f1_pos),
            "p_change": p_change,
            "p_inc_given_change": p_inc_given_change,
            "p3": p3
        }

    try:
        dir_tr = np.array([DIR_MAP[d] for d in dir_train[tr_mask]])
    except KeyError as e:
        raise ValueError(f"Unexpected direction label {e}. Expected one of {list(DIR_MAP.keys())}.")

    X_tr_dir = X_train[tr_mask]

    uniq_dir = np.unique(dir_tr)
    if len(uniq_dir) < 2:
        dir_default = int(uniq_dir[0])  # 0=decrease, 1=increase
        p_inc_given_change = np.full(len(X_test), 1.0 if dir_default == 1 else 0.0, dtype=float)
    else:
        m_dir = TabPFNClassifier(device=device)
        m_dir.fit(X_tr_dir, dir_tr)
        p_inc_given_change = m_dir.predict_proba(X_test)[:, 1] #predict on the whole set
        p_inc_given_change = np.asarray(p_inc_given_change, dtype=float)
        p_inc_given_change = np.clip(p_inc_given_change, 0.0, 1.0)

    # Final composed 3-class probabilities for calibration/ECE
    p3 = np.column_stack([
        1.0 - p_change,                           # same
        p_change * (1.0 - p_inc_given_change),     # decrease
        p_change * p_inc_given_change              # increase
    ])

    # only assign dec/inc when Stage A hard says "change"
    y3 = np.zeros(len(X_test), dtype=int)
    idx = np.where(p_change_hard == 1)[0]
    if len(idx) > 0:
        inc_hard = (p_inc_given_change >= 0.5).astype(int)  # 0=decrease,1=increase
        y3[idx] = np.where(inc_hard[idx] == 0, 1, 2)

    return {
        "y3_pred": y3,
        "thr": float(thr),
        "stageA_val_f1": float(best_f1_pos),
        "p_change": p_change,
        "p_inc_given_change": p_inc_given_change,
        "p3": p3
    }


def main():
    data_dir = Path(__file__).parent
    df = pd.read_csv(data_dir / "Data/60min_merged_imitation_learning_dataset_new.csv")

    if df is None:
        print(f"ERROR: the dataset files were not found in: {data_dir}!")
        raise SystemExit(1)

    # Remove patient number 72 from the dataset
    df = df[df["patient_id"] != "P-072"].reset_index(drop=True)
    print("Removed patient 72 from dataset")

    print(f"Total samples: {len(df)}")

    state_features = [c for c in df.columns if c.startswith("state_")]
    if len(state_features) == 0:
        print("ERROR: No state features found!")
        raise SystemExit(1)

    X = df[state_features].to_numpy()
    patient_ids = df["patient_id"].to_numpy()
    unique_patients = np.unique(patient_ids)

    print(f"State features: {len(state_features)}")
    print(f"Unique patients: {len(unique_patients)}")

    all_idx = np.arange(len(patient_ids))
    pid_to_idx = {pid: np.where(patient_ids == pid)[0] for pid in unique_patients}

    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = data_dir / "Logs" / "TabPFN"
    output_dir.mkdir(parents=True, exist_ok=True)

    models_root = data_dir / "models" / f"tabpfn_models_{run_ts}"
    models_root.mkdir(parents=True, exist_ok=True)

    results = []

    for knob, act_col, dir_col in KNOBS:
        print(f"\n  Training knob: {knob}")

        act = df[act_col].to_numpy().astype(int)
        direction = df[dir_col].to_numpy().astype(str)

        cm_sum = np.zeros((3, 3), dtype=int)
        f1s, baccs = [], []
        precisions, recalls = [], []
        recalls_per_class = []
        thresholds = []
        stage_a_f1s = []

        
        ece_rows = []

        for fold_idx, test_patient in enumerate(unique_patients):
            te_idx = pid_to_idx[test_patient]

            tr_mask = np.ones(len(all_idx), dtype=bool)
            tr_mask[te_idx] = False
            tr_idx = all_idx[tr_mask]

            X_train, X_test = X[tr_idx], X[te_idx]
            act_train, act_test = act[tr_idx], act[te_idx]
            dir_train, dir_test = direction[tr_idx], direction[te_idx]

            out = fit_predict_two_stage_tabpfn(
                X_train, X_test,
                act_train, dir_train,
                tune_stageA_threshold=True,
                device="cuda",
                verbose=False
            )

            pred_y3 = out["y3_pred"]
            chosen_thr = out["thr"]
            stage_a_f1 = out["stageA_val_f1"]
            p3 = out["p3"]  
            p_change = out["p_change"]
            p_inc_given_change = out["p_inc_given_change"]

            true_y3 = make_y3_from_truth(act_test, dir_test)

            f1s.append(f1_score(true_y3, pred_y3, average="macro", labels=[0, 1, 2], zero_division=0))
            baccs.append(balanced_accuracy_score(true_y3, pred_y3))
            precisions.append(precision_score(true_y3, pred_y3, average="macro", labels=[0, 1, 2], zero_division=0))
            recalls.append(recall_score(true_y3, pred_y3, average="macro", labels=[0, 1, 2], zero_division=0))
            recalls_per_class.append(recall_score(true_y3, pred_y3, average=None, labels=[0, 1, 2], zero_division=0))

            thresholds.append(chosen_thr)
            stage_a_f1s.append(stage_a_f1)

            cm_sum += confusion_matrix(true_y3, pred_y3, labels=[0, 1, 2])

            for i, row_idx in enumerate(te_idx):
                feat_dict = dict(zip(state_features, X_test[i].tolist()))
                ece_rows.append({
                    "knob": knob,
                    "fold_test_patient": str(test_patient),
                    "row_index": int(row_idx),

                    "y_true": int(true_y3[i]),   #same=0, dec=1, inc=2
                    "y_pred": int(pred_y3[i]),   #predicted class 0/1/2

                    "p_same": float(p3[i, 0]),
                    "p_decrease": float(p3[i, 1]),
                    "p_increase": float(p3[i, 2]),

                    "p_change": float(p_change[i]),
                    "p_inc_given_change": float(p_inc_given_change[i]),
                    "thr_stageA": float(chosen_thr),
                    **feat_dict,
                })

            if (fold_idx + 1) % 10 == 0:
                print(f"    Fold {fold_idx+1}/{len(unique_patients)} done")

        knob_result = {
            "head": knob,

            "macro_f1_mean": float(np.mean(f1s)),
            "balanced_acc_mean": float(np.mean(baccs)),
            "macro_precision_mean": float(np.mean(precisions)),
            "macro_recall_mean": float(np.mean(recalls)),

            "macro_f1_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
            "balanced_acc_std": float(np.std(baccs, ddof=1)) if len(baccs) > 1 else 0.0,
            "macro_precision_std": float(np.std(precisions, ddof=1)) if len(precisions) > 1 else 0.0,
            "macro_recall_std": float(np.std(recalls, ddof=1)) if len(recalls) > 1 else 0.0,

            "macro_f1_sem": float(np.std(f1s, ddof=1) / np.sqrt(len(f1s))) if len(f1s) > 1 else 0.0,
            "balanced_acc_sem": float(np.std(baccs, ddof=1) / np.sqrt(len(baccs))) if len(baccs) > 1 else 0.0,

            "recalls_per_class_mean": np.mean(recalls_per_class, axis=0),
            "recalls_per_class_std": np.std(recalls_per_class, axis=0, ddof=1) if len(recalls_per_class) > 1 else np.zeros(3),

            "cm_sum": cm_sum,

            "threshold_mean": float(np.mean(thresholds)),
            "threshold_median": float(np.median(thresholds)),
            "threshold_std": float(np.std(thresholds, ddof=1)) if len(thresholds) > 1 else 0.0,
            "threshold_min": float(np.min(thresholds)),
            "threshold_max": float(np.max(thresholds)),

            "stage_a_f1_mean": float(np.mean(stage_a_f1s)),
            "stage_a_f1_std": float(np.std(stage_a_f1s, ddof=1)) if len(stage_a_f1s) > 1 else 0.0,

            "fold_macro_f1": f1s,
            "fold_balanced_acc": baccs,
            "fold_precision": precisions,
            "fold_recall": recalls,
            "fold_thresholds": thresholds,
        }

        results.append(knob_result)

        print(f"\n[{knob}] macroF1={knob_result['macro_f1_mean']:.4f}  bAcc={knob_result['balanced_acc_mean']:.4f}")
        print(f"  Per-class Recall: same={knob_result['recalls_per_class_mean'][0]:.4f} "
              f"dec={knob_result['recalls_per_class_mean'][1]:.4f} inc={knob_result['recalls_per_class_mean'][2]:.4f}")

        results_df = pd.DataFrame(results)
        output_file = output_dir / f"tabpfn_multihead_lopo_{run_ts}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nResults (so far) saved to: {output_file}")

        ece_df = pd.DataFrame(ece_rows)
        ece_file = output_dir / f"tabpfn_ece_inputs_{knob}_{run_ts}.csv"
        ece_df.to_csv(ece_file, index=False)
        print(f"ECE inputs saved to: {ece_file}")

        print("\n" + "-" * 80)
        print(f"SAVING PRODUCTION MODELS FOR: {knob} (TabPFN)")
        print("-" * 80)

        thr_deploy = float(knob_result["threshold_median"])

        m_change_final = TabPFNClassifier(device="cuda")
        m_change_final.fit(X, act)

        tr_mask_full = (act == 1)
        m_dir_final = None
        if int(np.sum(tr_mask_full)) > 0:
            try:
                dir_full = np.array([DIR_MAP[d] for d in direction[tr_mask_full]])
            except KeyError as e:
                raise ValueError(f"Unexpected direction label {e}. Expected one of {list(DIR_MAP.keys())}.")
            X_dir_full = X[tr_mask_full]

            if len(np.unique(dir_full)) > 1:
                m_dir_final = TabPFNClassifier(device="cuda")
                m_dir_final.fit(X_dir_full, dir_full)

        knob_dir = models_root / knob
        knob_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(m_change_final, knob_dir / "stage_a_change_detector.pkl")
        if m_dir_final is not None:
            joblib.dump(m_dir_final, knob_dir / "stage_b_direction_classifier.pkl")

        metadata_file = models_root / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                model_metadata = json.load(f)
        else:
            model_metadata = {}

        model_metadata[knob] = {
            "knob": knob,
            "action_column": act_col,
            "direction_column": dir_col,
            "state_features": state_features,
            "dir_map": DIR_MAP,

            "thr_deploy": float(thr_deploy),

            "macro_f1_mean": float(knob_result["macro_f1_mean"]),
            "macro_f1_std": float(knob_result["macro_f1_std"]),
            "balanced_acc_mean": float(knob_result["balanced_acc_mean"]),
            "balanced_acc_std": float(knob_result["balanced_acc_std"]),
            "macro_precision_mean": float(knob_result["macro_precision_mean"]),
            "macro_precision_std": float(knob_result["macro_precision_std"]),
            "macro_recall_mean": float(knob_result["macro_recall_mean"]),
            "macro_recall_std": float(knob_result["macro_recall_std"]),

            "threshold_mean": float(knob_result["threshold_mean"]),
            "threshold_median": float(knob_result["threshold_median"]),
            "threshold_std": float(knob_result["threshold_std"]),
            "threshold_min": float(knob_result["threshold_min"]),
            "threshold_max": float(knob_result["threshold_max"]),

            "stage_a_f1_mean": float(knob_result["stage_a_f1_mean"]),
            "stage_a_f1_std": float(knob_result["stage_a_f1_std"]),
        }

        with open(metadata_file, "w") as f:
            json.dump(model_metadata, f, indent=2)

        print(f"Saved models to: {knob_dir}")
        print(f"Updated metadata: {metadata_file}")
        print(f"thr_deploy={thr_deploy:.2f}")

    print("\n" + "=" * 80)
    print("DONE. All knobs processed incrementally.")
    print("=" * 80)
    print(f"Models root: {models_root}")
    print(f"Logs: {output_dir}")


if __name__ == "__main__":
    main()
