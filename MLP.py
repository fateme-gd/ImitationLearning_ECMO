import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import joblib

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, confusion_matrix,
    recall_score, precision_score
)

# ---- configure your 5 knobs and their column names ----
KNOBS = [
    ("PO2",   "action_PO2 ARTERIAL (mmHg)_changed",   "action_PO2 ARTERIAL (mmHg)_direction"),
    ("PCO2",  "action_PCO2 ARTERIAL (mmHg)_changed",  "action_PCO2 ARTERIAL (mmHg)_direction"),
    ("SpO2",  "action_SpO2 (%)_changed",              "action_SpO2 (%)_direction"),
    ("FiO2",  "action_FiO2 - ECMO_changed",           "action_FiO2 - ECMO_direction"),
    # ("etCO2", "action_etCO2 (mmHg)_changed",          "action_etCO2 (mmHg)_direction"),
]

DIR_MAP = {"decrease": 0, "increase": 1}  # for direction-only model


# ---------- threshold tuning ----------
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
    """
    3-class truth:
      0 = same
      1 = decrease
      2 = increase
    """
    y3 = np.zeros(len(act), dtype=int)
    mask = (act == 1)
    y3[mask] = np.array([1 if d == "decrease" else 2 for d in direction[mask]])
    return y3


def _undersample_binary(X, y, random_state=42, ratio=1.0):
    y = np.asarray(y).astype(int)
    X = np.asarray(X)

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    # If only one class exists, return as-is
    if len(idx0) == 0 or len(idx1) == 0:
        return X, y

    # Identify minority/majority
    if len(idx0) <= len(idx1):
        idx_min, idx_maj = idx0, idx1
    else:
        idx_min, idx_maj = idx1, idx0

    rng = np.random.RandomState(random_state)

    # Keep all minority; downsample majority to ratio * minority (capped by available)
    target_maj = int(min(len(idx_maj), max(1, round(ratio * len(idx_min)))))
    idx_maj_down = rng.choice(idx_maj, size=target_maj, replace=False)

    idx = np.concatenate([idx_min, idx_maj_down])
    rng.shuffle(idx)
    return X[idx], y[idx]


def fit_predict_two_stage_mlp(
    X_train, X_test, act_train, dir_train,
    verbose=False,
    tune_stageA_threshold=True,
    undersample_ratio=2.0,  
):

    m_change = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        learning_rate="adaptive",
        max_iter=500,
        alpha=1e-4,
        early_stopping=False,   
        random_state=42,
    )

    thr = 0.5
    best_f1_pos = 0.0

    if tune_stageA_threshold and len(np.unique(act_train)) > 1 and len(act_train) > 50:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr_idx, va_idx = next(sss.split(X_train, act_train))
        X_tr, y_tr = X_train[tr_idx], act_train[tr_idx]
        X_va, y_va = X_train[va_idx], act_train[va_idx]

        X_tr_bal, y_tr_bal = _undersample_binary(X_tr, y_tr, random_state=42, ratio=undersample_ratio)
        m_change.fit(X_tr_bal, y_tr_bal)

        proba_va = m_change.predict_proba(X_va)[:, 1]
        thr, best_f1_pos = choose_threshold_on_train(proba_va, y_va)

        if verbose:
            pos = int(np.sum(act_train == 1))
            neg = int(np.sum(act_train == 0))
            print(f"      Stage A: pos={pos} neg={neg} thr={thr:.2f} valF1={best_f1_pos:.3f}")
    else:
        if len(np.unique(act_train)) > 1:
            X_bal, y_bal = _undersample_binary(X_train, act_train, random_state=42, ratio=undersample_ratio)
            m_change.fit(X_bal, y_bal)
        else:
            m_change.fit(X_train, act_train)

    proba_test_change = m_change.predict_proba(X_test)[:, 1]   # P(change=1)
    p_change_hard = (proba_test_change >= thr).astype(int)     # hard gate

    # --------------------------
    # Stage B: direction (act==1)
    # --------------------------
    tr_mask = (act_train == 1)
    if int(np.sum(tr_mask)) == 0:
        # never saw positive: predictions all "same"
        y3 = np.zeros(len(X_test), dtype=int)

        proba_test_inc_given_change = np.full(len(X_test), 0.5, dtype=float)
        p_same = 1.0 - proba_test_change
        p_inc = proba_test_change * proba_test_inc_given_change
        p_dec = proba_test_change * (1.0 - proba_test_inc_given_change)
        p3 = np.stack([p_same, p_dec, p_inc], axis=1)
        return y3, thr, best_f1_pos, p3, proba_test_change, proba_test_inc_given_change

    try:
        dir_tr = np.array([DIR_MAP[d] for d in dir_train[tr_mask]])
    except KeyError as e:
        raise ValueError(f"Unexpected direction label {e}. Expected one of {list(DIR_MAP.keys())}.")

    X_tr_dir = X_train[tr_mask]

    uniq_dir = np.unique(dir_tr)
    if len(uniq_dir) < 2:
        only = int(uniq_dir[0])  # 0=decrease,1=increase
        y3 = np.zeros(len(X_test), dtype=int)
        idx = np.where(p_change_hard == 1)[0]
        y3[idx] = 1 if only == 0 else 2

        proba_test_inc_given_change = np.full(len(X_test), 1.0 if only == 1 else 0.0, dtype=float)
        p_same = 1.0 - proba_test_change
        p_inc = proba_test_change * proba_test_inc_given_change
        p_dec = proba_test_change * (1.0 - proba_test_inc_given_change)
        p3 = np.stack([p_same, p_dec, p_inc], axis=1)
        return y3, thr, best_f1_pos, p3, proba_test_change, proba_test_inc_given_change

    m_dir = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation="relu",
        solver="adam",
        learning_rate="adaptive",
        max_iter=500,
        alpha=1e-4,
        early_stopping=False,
        random_state=42,
    )

    if len(dir_tr) > 50 and len(np.unique(dir_tr)) > 1:
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr2, _va2 = next(sss2.split(X_tr_dir, dir_tr))
        Xd_tr, yd_tr = X_tr_dir[tr2], dir_tr[tr2]

        Xd_tr_bal, yd_tr_bal = _undersample_binary(Xd_tr, yd_tr, random_state=42, ratio=undersample_ratio)
        m_dir.fit(Xd_tr_bal, yd_tr_bal)
    else:
        X_dir_bal, dir_bal = _undersample_binary(X_tr_dir, dir_tr, random_state=42, ratio=undersample_ratio)
        m_dir.fit(X_dir_bal, dir_bal)

    proba_test_inc_given_change = m_dir.predict_proba(X_test)[:, 1]  # P(increase | change=1)
    dir_pred_hard = (proba_test_inc_given_change >= 0.5).astype(int)  # 0=decrease,1=increase

    y3 = np.zeros(len(X_test), dtype=int)
    idx = np.where(p_change_hard == 1)[0]
    y3[idx] = np.where(dir_pred_hard[idx] == 0, 1, 2)

    p_same = 1.0 - proba_test_change
    p_inc = proba_test_change * proba_test_inc_given_change
    p_dec = proba_test_change * (1.0 - proba_test_inc_given_change)
    p3 = np.stack([p_same, p_dec, p_inc], axis=1)

    val_f1_stageA = best_f1_pos if (tune_stageA_threshold and len(np.unique(act_train)) > 1 and len(act_train) > 50) else 0.0
    return y3, thr, val_f1_stageA, p3, proba_test_change, proba_test_inc_given_change


def main():
    data_dir = Path(__file__).parent
    df = pd.read_csv(data_dir / "Data/60min_merged_imitation_learning_dataset_new.csv")

    if df is None:
        print(f"ERROR: the dataset files were not found in: {data_dir}!")
        raise SystemExit(1)

    # Remove patient number 72 from the dataset  For another evaluation
    df = df[df["patient_id"].astype(str) != "P-072"].reset_index(drop=True)
    print("Removed patient 72 from dataset")

    print(f"Total samples: {len(df)}")

    df['state_Type'] = df['state_Type'].map({'VV': 0, 'VA': 1})
    state_features = [c for c in df.columns if c.startswith("state_")]
    if len(state_features) == 0:
        print("ERROR: No state features found!")
        raise SystemExit(1)

    X_raw = df[state_features].to_numpy()
    patient_ids = df["patient_id"].to_numpy()
    unique_patients = np.unique(patient_ids)

    print(f"State features: {len(state_features)}")
    print(f"Unique patients: {len(unique_patients)}")

    results = []

    all_idx = np.arange(len(patient_ids))
    pid_to_idx = {pid: np.where(patient_ids == pid)[0] for pid in unique_patients}

    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    logs_dir = data_dir / "Logs" / "MLP"
    logs_dir.mkdir(parents=True, exist_ok=True)

    models_dir = data_dir / "models" / f"mlp_models_{run_ts}"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("SAVING DEPLOYMENT PREPROCESSING (MLP: Imputer + Scaler)")
    print("=" * 80)

    imputer_full = SimpleImputer(strategy="median", add_indicator=True)
    scaler_full = StandardScaler()

    X_full_imp = imputer_full.fit_transform(X_raw)
    X_full = scaler_full.fit_transform(X_full_imp)

    joblib.dump(imputer_full, models_dir / "imputer_full.pkl")
    joblib.dump(scaler_full, models_dir / "scaler_full.pkl")

    model_metadata = {}

    # -------------------------
    # LOPO evaluation per knob
    # -------------------------
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

        # per-sample probability logs for ECE
        ece_rows = []

        for fold_idx, test_patient in enumerate(unique_patients):
            te_idx = pid_to_idx[test_patient]
            tr_mask = np.ones(len(all_idx), dtype=bool)
            tr_mask[te_idx] = False
            tr_idx = all_idx[tr_mask]

            X_train_raw, X_test_raw = X_raw[tr_idx], X_raw[te_idx]
            act_train, act_test = act[tr_idx], act[te_idx]
            dir_train, dir_test = direction[tr_idx], direction[te_idx]

            imputer = SimpleImputer(strategy="median", add_indicator=True)
            scaler = StandardScaler()

            X_train_imp = imputer.fit_transform(X_train_raw)
            X_test_imp = imputer.transform(X_test_raw)

            X_train = scaler.fit_transform(X_train_imp)
            X_test = scaler.transform(X_test_imp)

            pred_y3, chosen_thr, stage_a_f1, p3, p_change_prob, p_inc_given_change = fit_predict_two_stage_mlp(
                X_train, X_test,
                act_train, dir_train,
                verbose=False,
                tune_stageA_threshold=True,
                undersample_ratio=2.0, 
            )

            true_y3 = make_y3_from_truth(act_test, dir_test)

            f1s.append(f1_score(true_y3, pred_y3, average="macro", labels=[0, 1, 2], zero_division=0))
            baccs.append(balanced_accuracy_score(true_y3, pred_y3))
            precisions.append(precision_score(true_y3, pred_y3, average="macro", labels=[0, 1, 2], zero_division=0))
            recalls.append(recall_score(true_y3, pred_y3, average="macro", labels=[0, 1, 2], zero_division=0))
            recalls_per_class.append(recall_score(true_y3, pred_y3, average=None, labels=[0, 1, 2], zero_division=0))

            thresholds.append(chosen_thr)
            stage_a_f1s.append(stage_a_f1)

            cm_sum += confusion_matrix(true_y3, pred_y3, labels=[0, 1, 2])

            # p3 columns: [p_same, p_decrease, p_increase]
            for i, row_idx in enumerate(te_idx):
                feat_dict = dict(zip(state_features, X_test[i].tolist()))
                ece_rows.append({
                    "knob": knob,
                    "fold_test_patient": str(test_patient),
                    "row_index": int(row_idx),

                    "y_true": int(true_y3[i]),
                    "y_pred": int(pred_y3[i]),

                    "p_same": float(p3[i, 0]),
                    "p_decrease": float(p3[i, 1]),
                    "p_increase": float(p3[i, 2]),

                    "p_change": float(p_change_prob[i]),
                    "p_inc_given_change": float(p_inc_given_change[i]),
                    "thr_stageA": float(chosen_thr),
                    **feat_dict,
                })

            if (fold_idx + 1) % 10 == 0:
                print(f"    Fold {fold_idx+1}/{len(unique_patients)} done")

        results.append({
            "head": knob,

            "macro_f1_mean": float(np.mean(f1s)),
            "balanced_acc_mean": float(np.mean(baccs)),
            "macro_precision_mean": float(np.mean(precisions)),
            "macro_recall_mean": float(np.mean(recalls)),

            "macro_f1_std": float(np.std(f1s, ddof=1)),
            "balanced_acc_std": float(np.std(baccs, ddof=1)),
            "macro_precision_std": float(np.std(precisions, ddof=1)),
            "macro_recall_std": float(np.std(recalls, ddof=1)),

            "macro_f1_sem": float(np.std(f1s, ddof=1) / np.sqrt(len(f1s))),
            "balanced_acc_sem": float(np.std(baccs, ddof=1) / np.sqrt(len(baccs))),

            "recalls_per_class_mean": np.mean(recalls_per_class, axis=0),
            "recalls_per_class_std": np.std(recalls_per_class, axis=0, ddof=1),

            "cm_sum": cm_sum,

            "threshold_mean": float(np.mean(thresholds)),
            "threshold_median": float(np.median(thresholds)),
            "threshold_std": float(np.std(thresholds, ddof=1)),
            "threshold_min": float(np.min(thresholds)),
            "threshold_max": float(np.max(thresholds)),

            "stage_a_f1_mean": float(np.mean(stage_a_f1s)),
            "stage_a_f1_std": float(np.std(stage_a_f1s, ddof=1)),

            "fold_macro_f1": f1s,
            "fold_balanced_acc": baccs,
            "fold_precision": precisions,
            "fold_recall": recalls,
            "fold_thresholds": thresholds,
        })

        print(f"\n[{knob}] macroF1={results[-1]['macro_f1_mean']:.4f}  bAcc={results[-1]['balanced_acc_mean']:.4f}")
        print(f"  Per-class Recall: same={results[-1]['recalls_per_class_mean'][0]:.4f} "
              f"dec={results[-1]['recalls_per_class_mean'][1]:.4f} inc={results[-1]['recalls_per_class_mean'][2]:.4f}")


        results_df = pd.DataFrame(results)
        output_file = logs_dir / f"mlp_multihead_lopo_{run_ts}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

        ece_df = pd.DataFrame(ece_rows)
        ece_file = logs_dir / f"mlp_ece_inputs_{knob}_{run_ts}.csv"
        ece_df.to_csv(ece_file, index=False)
        print(f"ECE inputs saved to: {ece_file}")

        print("\n" + "=" * 80)
        print(f"SAVING PRODUCTION MODELS FOR: {knob} (MLP + Imputer + Scaler)")
        print("=" * 80)

        thr_deploy = float(results[-1]["threshold_median"])

        m_change_final = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            learning_rate="adaptive",
            max_iter=500,
            alpha=1e-4,
            early_stopping=False,
            random_state=42,
        )
        if len(np.unique(act)) > 1:
            X_A_bal, y_A_bal = _undersample_binary(X_full, act, random_state=42, ratio=2.0)
            m_change_final.fit(X_A_bal, y_A_bal)
        else:
            m_change_final.fit(X_full, act)

        # Stage B final (act==1 only)
        tr_mask_full = (act == 1)
        m_dir_final = None
        if int(np.sum(tr_mask_full)) > 0:
            try:
                dir_full = np.array([DIR_MAP[d] for d in direction[tr_mask_full]])
            except KeyError as e:
                raise ValueError(f"Unexpected direction label {e}. Expected one of {list(DIR_MAP.keys())}.")
            X_dir_full = X_full[tr_mask_full]

            if len(np.unique(dir_full)) > 1:
                m_dir_final = MLPClassifier(
                    hidden_layer_sizes=(64, 32, 16),
                    activation="relu",
                    solver="adam",
                    learning_rate="adaptive",
                    max_iter=500,
                    alpha=1e-4,
                    early_stopping=False,
                    random_state=42,
                )
                X_B_bal, y_B_bal = _undersample_binary(X_dir_full, dir_full, random_state=42, ratio=2.0)
                m_dir_final.fit(X_B_bal, y_B_bal)

        knob_dir = models_dir / knob
        knob_dir.mkdir(exist_ok=True)

        joblib.dump(m_change_final, knob_dir / "stage_a_change_detector.pkl")
        if m_dir_final is not None:
            joblib.dump(m_dir_final, knob_dir / "stage_b_direction_classifier.pkl")

        model_metadata[knob] = {
            "knob": knob,
            "action_column": act_col,
            "direction_column": dir_col,
            "state_features": state_features,
            "dir_map": DIR_MAP,

            "thr_deploy": float(thr_deploy),

            "macro_f1_mean": float(results[-1]["macro_f1_mean"]),
            "macro_f1_std": float(results[-1]["macro_f1_std"]),
            "balanced_acc_mean": float(results[-1]["balanced_acc_mean"]),
            "balanced_acc_std": float(results[-1]["balanced_acc_std"]),
            "macro_precision_mean": float(results[-1]["macro_precision_mean"]),
            "macro_precision_std": float(results[-1]["macro_precision_std"]),
            "macro_recall_mean": float(results[-1]["macro_recall_mean"]),
            "macro_recall_std": float(results[-1]["macro_recall_std"]),

            "threshold_mean": float(results[-1]["threshold_mean"]),
            "threshold_median": float(results[-1]["threshold_median"]),
            "threshold_std": float(results[-1]["threshold_std"]),
            "threshold_min": float(results[-1]["threshold_min"]),
            "threshold_max": float(results[-1]["threshold_max"]),

            "stage_a_f1_mean": float(results[-1]["stage_a_f1_mean"]),
            "stage_a_f1_std": float(results[-1]["stage_a_f1_std"]),

            "imputer": {"strategy": "median", "add_indicator": True},
            "scaler": {"type": "StandardScaler"},
        }

        metadata_file = models_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(model_metadata, f, indent=2)

        print(f"Saved models to: {knob_dir}")
        print(f"Updated metadata: {metadata_file}")

    print("\nDONE.")
    print(f"All models saved under: {models_dir}")
    print(f"Imputer saved to: {models_dir / 'imputer_full.pkl'}")
    print(f"Scaler saved to:  {models_dir / 'scaler_full.pkl'}")


if __name__ == "__main__":
    main()
