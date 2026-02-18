import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, confusion_matrix,
    recall_score, precision_score
)
from sklearn.model_selection import StratifiedShuffleSplit
from datetime import datetime
import json
import joblib

KNOBS = [
    ("PO2",   "action_PO2 ARTERIAL (mmHg)_changed",   "action_PO2 ARTERIAL (mmHg)_direction"),
    ("PCO2",  "action_PCO2 ARTERIAL (mmHg)_changed",  "action_PCO2 ARTERIAL (mmHg)_direction"),
    ("SpO2",  "action_SpO2 (%)_changed",              "action_SpO2 (%)_direction"),
    ("FiO2",  "action_FiO2 - ECMO_changed",           "action_FiO2 - ECMO_direction"),
    ("etCO2", "action_etCO2 (mmHg)_changed",          "action_etCO2 (mmHg)_direction"),
]

DIR_MAP = {"decrease": 0, "increase": 1}  # for direction-only model

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

def fit_predict_two_stage_xgb(
    X_train, X_test, act_train, dir_train,
    verbose=False,
    tune_stageA_threshold=True,
    use_gpu=True,
):
    pos = int(np.sum(act_train == 1))
    neg = int(np.sum(act_train == 0))
    spw = (neg / max(pos, 1))

    tree_method = "hist"
    device = "cuda" if use_gpu else "cpu"

    m_change = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=200,
        max_depth=4,
        min_child_weight=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method=tree_method,
        device=device,
        n_jobs=1 if use_gpu else -1,
        scale_pos_weight=spw,
        random_state=42,
        verbosity=0,
        early_stopping_rounds=20,
    )

    thr = 0.5
    best_f1_pos = 0.0
    if tune_stageA_threshold and len(np.unique(act_train)) > 1 and len(act_train) > 50:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr_idx, va_idx = next(sss.split(X_train, act_train))
        X_tr, y_tr = X_train[tr_idx], act_train[tr_idx]
        X_va, y_va = X_train[va_idx], act_train[va_idx]

        m_change.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )

        proba_va = m_change.predict_proba(X_va)[:, 1]
        thr, best_f1_pos = choose_threshold_on_train(proba_va, y_va)

        if verbose:
            print(f"      Stage A: pos={pos} neg={neg} spw={spw:.2f} thr={thr:.2f} valF1={best_f1_pos:.3f}")
    else:
        m_change.fit(X_train, act_train)

    proba_test_change = m_change.predict_proba(X_test)[:, 1]
    p_change = (proba_test_change >= thr).astype(int)

    tr_mask = (act_train == 1)
    if int(np.sum(tr_mask)) == 0:
        y3 = np.zeros(len(X_test), dtype=int)
        p_inc_given_change_nosample = np.full(len(X_test), 0.5, dtype=float)
        proba_y3 = np.zeros((len(X_test), 3), dtype=float)
        proba_y3[:, 0] = 1.0  # p_same
        conf = proba_y3.max(axis=1)

        val_f1_stageA = best_f1_pos if tune_stageA_threshold and len(np.unique(act_train)) > 1 and len(act_train) > 50 else 0.0
        return y3, thr, val_f1_stageA, proba_y3, conf, proba_test_change, p_inc_given_change_nosample

    dir_tr = np.array([DIR_MAP[d] for d in dir_train[tr_mask]])
    X_tr_dir = X_train[tr_mask]

    pos_dir = int(np.sum(dir_tr == 1))  # increase
    neg_dir = int(np.sum(dir_tr == 0))  # decrease
    spw_dir = (neg_dir / max(pos_dir, 1))

    m_dir = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=150,
        max_depth=3,
        min_child_weight=5,
        learning_rate=0.10,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method=tree_method,
        device=device,
        n_jobs=1 if use_gpu else -1,
        scale_pos_weight=spw_dir,
        random_state=42,
        verbosity=0,
        early_stopping_rounds=15,
    )

    if len(dir_tr) > 50 and len(np.unique(dir_tr)) > 1:
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr2, va2 = next(sss2.split(X_tr_dir, dir_tr))
        Xd_tr, yd_tr = X_tr_dir[tr2], dir_tr[tr2]
        Xd_va, yd_va = X_tr_dir[va2], dir_tr[va2]
        m_dir.fit(
            Xd_tr, yd_tr,
            eval_set=[(Xd_va, yd_va)],
            verbose=False,
        )
    else:
        m_dir.fit(X_tr_dir, dir_tr)

    proba_test_inc_given_change = m_dir.predict_proba(X_test)[:, 1]  # P(increase | change, s)
    dir_pred = (proba_test_inc_given_change >= 0.5).astype(int)      # 0=decrease,1=increase

    # combine into 3-class: 0 same, 1 decrease, 2 increase
    y3 = np.zeros(len(X_test), dtype=int)
    idx = np.where(p_change == 1)[0]
    y3[idx] = np.where(dir_pred[idx] == 0, 1, 2)

    
    p_same = 1.0 - proba_test_change
    p_inc  = proba_test_change * proba_test_inc_given_change
    p_dec  = proba_test_change * (1.0 - proba_test_inc_given_change)

    proba_y3 = np.stack([p_same, p_dec, p_inc], axis=1)
    proba_y3 = np.clip(proba_y3, 0.0, 1.0)
    row_sums = proba_y3.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    proba_y3 = proba_y3 / row_sums

    conf = proba_y3.max(axis=1)

    val_f1_stageA = best_f1_pos if tune_stageA_threshold and len(np.unique(act_train)) > 1 and len(act_train) > 50 else 0.0
    return y3, thr, val_f1_stageA, proba_y3, conf, proba_test_change, proba_test_inc_given_change

def make_y3_from_truth(act, direction):
    y3 = np.zeros(len(act), dtype=int)
    mask = (act == 1)
    y3[mask] = np.array([1 if d == "decrease" else 2 for d in direction[mask]])
    return y3

data_dir = Path(__file__).parent
df = pd.read_csv(data_dir / "Data/60min_merged_imitation_learning_dataset.csv")

if df is None:
    print(f"ERROR: the dataset files were not found in: {data_dir}!")
    raise SystemExit(1)

# Remove patient number 72 from the dataset
df = df[df["patient_id"] != "P-072"].reset_index(drop=True)
print(f"Removed patient 72 from dataset")

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

results = []

all_idx = np.arange(len(patient_ids))
pid_to_idx = {pid: np.where(patient_ids == pid)[0] for pid in unique_patients}


for knob, act_col, dir_col in KNOBS:
    ece_rows = [] 
    print(f"\n  Training knob: {knob}")
    act = df[act_col].to_numpy().astype(int)
    direction = df[dir_col].to_numpy().astype(str)

    cm_sum = np.zeros((3, 3), dtype=int)
    f1s, baccs = [], []
    precisions, recalls = [], []
    recalls_per_class = []
    thresholds = []
    stage_a_f1s = []

    for fold_idx, test_patient in enumerate(unique_patients):
        test_patient_idxes = pid_to_idx[test_patient]
        tr_mask = np.ones(len(all_idx), dtype=bool)
        tr_mask[test_patient_idxes] = False
        tr_idx = all_idx[tr_mask]

        X_train, X_test = X[tr_idx], X[test_patient_idxes]
        act_train, act_test = act[tr_idx], act[test_patient_idxes]
        dir_train, dir_test = direction[tr_idx], direction[test_patient_idxes]

        pred_y3, chosen_thr, stage_a_f1, proba_y3, conf, p_change, p_inc_given_change = fit_predict_two_stage_xgb(
            X_train, X_test,
            act_train, dir_train,
            verbose=False,
            tune_stageA_threshold=True,
            use_gpu=True
        )

        true_y3 = make_y3_from_truth(act_test, dir_test)

        for i in range(len(true_y3)):
            feat_dict = dict(zip(state_features, X_test[i].tolist()))

            ece_rows.append({
                "model": "XGBoost",
                "knob": knob,
                "fold_idx": int(fold_idx),
                "test_patient": str(test_patient),
                "row_index": int(test_patient_idxes[i]),

                "y_true": int(true_y3[i]),
                "y_pred": int(pred_y3[i]),

                "p_same": float(proba_y3[i, 0]),
                "p_decrease": float(proba_y3[i, 1]),
                "p_increase": float(proba_y3[i, 2]),
                "p_change": float(p_change[i]),
                "p_inc_given_change": float(p_inc_given_change[i]) ,
                "conf": float(conf[i]),
                **feat_dict,
            })

        f1s.append(f1_score(true_y3, pred_y3, average="macro", labels=[0, 1, 2], zero_division=0))
        baccs.append(balanced_accuracy_score(true_y3, pred_y3))
        precisions.append(precision_score(true_y3, pred_y3, average="macro", labels=[0, 1, 2], zero_division=0))
        recalls.append(recall_score(true_y3, pred_y3, average="macro", labels=[0, 1, 2], zero_division=0))
        recalls_per_class.append(recall_score(true_y3, pred_y3, average=None, labels=[0, 1, 2], zero_division=0))

        thresholds.append(chosen_thr)
        stage_a_f1s.append(stage_a_f1)

        cm_sum += confusion_matrix(true_y3, pred_y3, labels=[0, 1, 2])

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

    output_dir = data_dir / "Logs" / "XGBoost"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"xgboost_multihead_gpu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    ece_df = pd.DataFrame(ece_rows)
    ece_file = output_dir / f"xgboost_lopo_preds_for_ece_{knob}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    ece_df.to_csv(ece_file, index=False)
    print(f"ECE log saved to: {ece_file}")

    print("\n" + "="*80)
    print("SAVING PRODUCTION MODELS FOR DEPLOYMENT")
    print("="*80)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    models_dir = data_dir / "models" / f"xgboost_models_{timestamp}"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_metadata = {}

    for result_idx, r in enumerate(results):
        knob = r['head']
        act_col = KNOBS[result_idx][1]
        dir_col = KNOBS[result_idx][2]

        print(f"\n  Training production model for: {knob}")

        act = df[act_col].to_numpy().astype(int)
        direction = df[dir_col].to_numpy().astype(str)

        pos = int(np.sum(act == 1))
        neg = int(np.sum(act == 0))
        spw = (neg / max(pos, 1))

        m_change_final = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=200,
            max_depth=4,
            min_child_weight=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            device="cuda",
            scale_pos_weight=spw,
            random_state=42,
            verbosity=0,
        )
        m_change_final.fit(X, act)

        tr_mask = (act == 1)
        if np.sum(tr_mask) > 0:
            dir_full = np.array([DIR_MAP[d] for d in direction[tr_mask]])
            X_dir_full = X[tr_mask]

            pos_dir = int(np.sum(dir_full == 1))
            neg_dir = int(np.sum(dir_full == 0))
            spw_dir = (neg_dir / max(pos_dir, 1))

            m_dir_final = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=150,
                max_depth=3,
                min_child_weight=5,
                learning_rate=0.10,
                subsample=0.9,
                colsample_bytree=0.9,
                tree_method="hist",
                device="cuda",
                scale_pos_weight=spw_dir,
                random_state=42,
                verbosity=0,
            )
            m_dir_final.fit(X_dir_full, dir_full)
        else:
            m_dir_final = None

        knob_dir = models_dir / knob
        knob_dir.mkdir(exist_ok=True)

        m_change_final.save_model(str(knob_dir / "stage_a_change_detector.ubj"))
        if m_dir_final:
            m_dir_final.save_model(str(knob_dir / "stage_b_direction_classifier.ubj"))

        metadata = {
            "knob": knob,
            "macro_f1_mean": float(r['macro_f1_mean']),
            "macro_f1_std": float(r['macro_f1_std']),
            "balanced_acc_mean": float(r['balanced_acc_mean']),
            "balanced_acc_std": float(r['balanced_acc_std']),
            "macro_precision_mean": float(r['macro_precision_mean']),
            "macro_precision_std": float(r['macro_precision_std']),
            "macro_recall_mean": float(r['macro_recall_mean']),
            "macro_recall_std": float(r['macro_recall_std']),
            "threshold_mean": float(r['threshold_mean']),
            "threshold_median": float(r['threshold_median']),
            "threshold_std": float(r['threshold_std']),
            "threshold_min": float(r['threshold_min']),
            "threshold_max": float(r['threshold_max']),
            "stage_a_f1_mean": float(r['stage_a_f1_mean']),
            "stage_a_f1_std": float(r['stage_a_f1_std']),
            "action_column": act_col,
            "direction_column": dir_col,
            "state_features": state_features,
            "dir_map": DIR_MAP,
        }
        model_metadata[knob] = metadata

        print(f"    Saved Stage A and Stage B models for {knob}")

    metadata_file = models_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(model_metadata, f, indent=2)

    print(f"\nAll models saved to: {models_dir}")
    print(f"Metadata saved to: {metadata_file}")

    print(f"\n All models saved to: {models_dir}")
    print(f"   Metadata saved to: {metadata_file}")
