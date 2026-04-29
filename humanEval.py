import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os

"""
Human Evaluation Dataset Generator

Picks 6 patients at random (3 injured, 3 uninjured), trains FQI and
FQI_SPIBB (Nwedge=5) on the remaining patients, then annotates the
held-out trajectories with per-head recommended actions from each policy.

Output columns (appended to the original trajectory rows):
  - action_fqi_{head}       : greedy FQI action (0=same, 1=decrease, 2=increase)
  - action_spibb_{head}     : SPIBB-constrained FQI action (Nwedge=5)
  - injured                 : 1 if patient has Any_Injury, 0 otherwise
"""

# -----------------------------
# CONFIG  (mirrors testtt.py)
# -----------------------------
DATA_FILE   = "60min_imitation_learning_with_rewards.csv"
PATIENT_COL = "patient_id"
TIMESTEP_COL = "timestep"
REWARD_COL  = "reward"

RANDOM_STATE  = 42
GAMMA         = 0.99
K_BINS        = 100       # KMeans clusters for support estimation
NWEDGE        = 5         # SPIBB threshold
FQI_ITERS     = 10
N_ACTIONS     = 3
PIB_ALPHA     = 1e-3
EPS           = 1e-12

_N_HEADS = 5
TREE_KW = dict(
    n_estimators=30,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=max(1, (os.cpu_count() or 1) // _N_HEADS),
    tree_method="hist",
)

HEAD_NAMES = ["PO2", "PCO2", "SpO2", "FiO2", "etCO2"]
ACTION_COLS = [
    ("PO2",   "action_PO2 ARTERIAL (mmHg)_changed",   "action_PO2 ARTERIAL (mmHg)_direction"),
    ("PCO2",  "action_PCO2 ARTERIAL (mmHg)_changed",  "action_PCO2 ARTERIAL (mmHg)_direction"),
    ("SpO2",  "action_SpO2 (%)_changed",              "action_SpO2 (%)_direction"),
    ("FiO2",  "action_FiO2 - ECMO_changed",           "action_FiO2 - ECMO_direction"),
    ("etCO2", "action_etCO2 (mmHg)_changed",          "action_etCO2 (mmHg)_direction"),
]
DIR_TO_Y = {"decrease": 1, "increase": 2}


# -----------------------------
# Helpers (same as testtt.py)
# -----------------------------
def _normalize_dir_token(x) -> str:
    s = str(x).strip().lower()
    s = s.replace('"', "").replace("'", "")
    return s


def parse_actions_cell_to_y3(row) -> np.ndarray:
    y3 = np.zeros(len(HEAD_NAMES), dtype=int)
    for head_k, (head_name, changed_col, direction_col) in enumerate(ACTION_COLS):
        c = int(row[changed_col])
        d_tok = _normalize_dir_token(row[direction_col])
        if c == 0:
            y3[head_k] = 0
        elif c == 1:
            if d_tok in DIR_TO_Y:
                y3[head_k] = DIR_TO_Y[d_tok]
            else:
                raise ValueError(f"Unknown direction for {head_name}: {row[direction_col]}")
        else:
            raise ValueError(f"Changed flag must be 0/1 for {head_name}, got {c}")
    return y3


def build_action_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.vstack([parse_actions_cell_to_y3(row) for _, row in df.iterrows()])


def get_state_columns(df: pd.DataFrame):
    state_cols = sorted([c for c in df.columns if c.startswith("state_")])
    if not state_cols:
        raise ValueError("No state_* columns found.")
    return state_cols


def train_mean_impute(X_train: np.ndarray, X_other: np.ndarray):
    means = np.nanmean(X_train, axis=0)
    means = np.where(np.isnan(means), 0.0, means)
    X_train_out = X_train.copy()
    X_other_out = X_other.copy()
    mask_tr = np.isnan(X_train_out)
    if mask_tr.any():
        X_train_out[mask_tr] = np.take(means, np.where(mask_tr)[1])
    mask_ot = np.isnan(X_other_out)
    if mask_ot.any():
        X_other_out[mask_ot] = np.take(means, np.where(mask_ot)[1])
    return X_train_out, X_other_out, means


def compute_pi_b_bin_counts(b_train, a_train, K, n_actions=N_ACTIONS, alpha=PIB_ALPHA):
    counts = np.zeros((K, n_actions), dtype=float)
    for bt, at in zip(b_train, a_train):
        counts[int(bt), int(at)] += 1.0
    counts += float(alpha)
    return counts / np.maximum(counts.sum(axis=1, keepdims=True), EPS)


def compute_support_counts(b_train, a_train, K, n_actions=N_ACTIONS):
    counts = np.zeros((K, n_actions), dtype=int)
    for bt, at in zip(b_train, a_train):
        counts[int(bt), int(at)] += 1
    return counts


# -----------------------------
# Q-model container
# -----------------------------
@dataclass
class MultiActionQModel:
    models: list
    n_actions: int = N_ACTIONS

    def predict_all(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([m.predict(X) for m in self.models])


# -----------------------------
# FQI
# -----------------------------
class FittedQIteration:
    def __init__(self, n_actions=N_ACTIONS, gamma=GAMMA, n_iters=FQI_ITERS, tree_kw=None):
        self.n_actions = n_actions
        self.gamma = gamma
        self.n_iters = n_iters
        self.tree_kw = TREE_KW if tree_kw is None else tree_kw
        self.model_ = None

    def fit(self, X, a, r, X_next, done):
        target = r.astype(float).copy()
        models = [XGBRegressor(**self.tree_kw) for _ in range(self.n_actions)]
        for act in range(self.n_actions):
            idx = np.where(a == act)[0]
            if len(idx) == 0:
                models[act].fit(X[:1], np.array([0.0]))
            else:
                models[act].fit(X[idx], target[idx])
        q_model = MultiActionQModel(models=models, n_actions=self.n_actions)

        for _ in tqdm(range(self.n_iters), desc="FQI", leave=False):
            Q_next = q_model.predict_all(X_next)
            bellman = r + self.gamma * (~done).astype(float) * np.max(Q_next, axis=1)
            new_models = []
            for act in range(self.n_actions):
                idx = np.where(a == act)[0]
                model = XGBRegressor(**self.tree_kw)
                if len(idx) == 0:
                    model.fit(X[:1], np.array([0.0]))
                else:
                    model.fit(X[idx], bellman[idx])
                new_models.append(model)
            q_model = MultiActionQModel(models=new_models, n_actions=self.n_actions)

        self.model_ = q_model
        return self

    def predict_all(self, X):
        return self.model_.predict_all(X)

    def greedy_action(self, X):
        return np.argmax(self.predict_all(X), axis=1)


# -----------------------------
# SPIBB backup helper
# -----------------------------
def spibb_projected_next_value(Q_next, P_base_next, bins_next, support_counts, Nwedge):
    n, n_actions = Q_next.shape
    V_next = np.zeros(n, dtype=float)
    for i, b in enumerate(bins_next):
        b = int(b)
        boot_mask = support_counts[b] < int(Nwedge)
        nonboot_mask = ~boot_mask
        v = float(np.dot(P_base_next[i, boot_mask], Q_next[i, boot_mask]))
        if np.any(nonboot_mask):
            nonboot_actions = np.where(nonboot_mask)[0]
            remaining_mass = float(np.sum(P_base_next[i, nonboot_actions]))
            best_nonboot = nonboot_actions[np.argmax(Q_next[i, nonboot_actions])]
            v += remaining_mass * float(Q_next[i, best_nonboot])
        else:
            v = float(np.dot(P_base_next[i], Q_next[i]))
        V_next[i] = v
    return V_next


# -----------------------------
# SPIBB FQI
# -----------------------------
class SPIBBFittedQIteration:
    def __init__(self, km, scaler, support_counts, Nwedge, baseline_policy_fn,
                 n_actions=N_ACTIONS, gamma=GAMMA, n_iters=FQI_ITERS, tree_kw=None):
        self.km = km
        self.scaler = scaler
        self.support_counts = support_counts
        self.Nwedge = int(Nwedge)
        self.baseline_policy_fn = baseline_policy_fn
        self.n_actions = n_actions
        self.gamma = gamma
        self.n_iters = n_iters
        self.tree_kw = TREE_KW if tree_kw is None else tree_kw
        self.model_ = None

    def fit(self, X, a, r, X_next, done):
        target = r.astype(float).copy()
        models = [XGBRegressor(**self.tree_kw) for _ in range(self.n_actions)]
        for act in range(self.n_actions):
            idx = np.where(a == act)[0]
            if len(idx) == 0:
                models[act].fit(X[:1], np.array([0.0]))
            else:
                models[act].fit(X[idx], target[idx])
        q_model = MultiActionQModel(models=models, n_actions=self.n_actions)

        X_next_scaled = self.scaler.transform(X_next)
        bins_next = self.km.predict(X_next_scaled)
        P_base_next = self.baseline_policy_fn(X_next)

        for _ in tqdm(range(self.n_iters), desc="SPIBB-FQI", leave=False):
            Q_next = q_model.predict_all(X_next)
            V_next = spibb_projected_next_value(
                Q_next=Q_next,
                P_base_next=P_base_next,
                bins_next=bins_next,
                support_counts=self.support_counts,
                Nwedge=self.Nwedge,
            )
            bellman = r + self.gamma * (~done).astype(float) * V_next
            new_models = []
            for act in range(self.n_actions):
                idx = np.where(a == act)[0]
                model = XGBRegressor(**self.tree_kw)
                if len(idx) == 0:
                    model.fit(X[:1], np.array([0.0]))
                else:
                    model.fit(X[idx], bellman[idx])
                new_models.append(model)
            q_model = MultiActionQModel(models=new_models, n_actions=self.n_actions)

        self.model_ = q_model
        return self

    def predict_all(self, X):
        return self.model_.predict_all(X)

    def supported_greedy_action(self, X, baseline_policy_fn):
        """
        For each state, pick the greedy action among supported actions.
        Falls back to baseline argmax if no action is supported.
        """
        Xs = self.scaler.transform(X)
        bins = self.km.predict(Xs)
        Q = self.predict_all(X)
        P_base = baseline_policy_fn(X)
        actions = np.zeros(len(X), dtype=int)
        for i, b in enumerate(bins):
            supported = np.where(self.support_counts[int(b)] >= self.Nwedge)[0]
            if supported.size == 0:
                # no supported action — follow baseline
                actions[i] = int(np.argmax(P_base[i]))
            else:
                actions[i] = int(supported[np.argmax(Q[i, supported])])
        return actions


# -----------------------------
# Count-based baseline policy
# -----------------------------
def make_count_policy_fn(km, scaler, pi_b_bin):
    def policy_fn(X_raw):
        Xs = scaler.transform(X_raw)
        bins = km.predict(Xs)
        return pi_b_bin[bins]
    return policy_fn


# -----------------------------
# Patient sampling
# -----------------------------
def sample_holdout_patients(patient_ids_all, injury_map, n_each=3, seed=RANDOM_STATE):
    """
    Sample n_each injured and n_each uninjured patients for hold-out.
    Returns (holdout_ids, train_ids).
    """
    rng = np.random.default_rng(seed)
    unique_pids = np.unique(patient_ids_all)

    injured   = [p for p in unique_pids if injury_map.get(p, np.nan) == 1]
    uninjured = [p for p in unique_pids if injury_map.get(p, np.nan) == 0]

    if len(injured) < n_each:
        raise ValueError(f"Only {len(injured)} injured patients, need {n_each}")
    if len(uninjured) < n_each:
        raise ValueError(f"Only {len(uninjured)} uninjured patients, need {n_each}")

    holdout_injured   = list(rng.choice(injured,   n_each, replace=False))
    holdout_uninjured = list(rng.choice(uninjured, n_each, replace=False))
    holdout_ids = set(holdout_injured + holdout_uninjured)
    train_ids   = set(unique_pids) - holdout_ids

    print(f"\nHold-out patients ({n_each} injured + {n_each} uninjured):")
    print(f"  Injured:   {holdout_injured}")
    print(f"  Uninjured: {holdout_uninjured}")
    print(f"  Train patients: {len(train_ids)}")

    return holdout_ids, train_ids, holdout_injured, holdout_uninjured


# -----------------------------
# Main
# -----------------------------
def main():
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    data_path = Path(__file__).parent / "Data" / DATA_FILE
    if not data_path.exists():
        raise SystemExit(f"File not found: {data_path}")

    df = pd.read_csv(data_path)
    df = df.sort_values([PATIENT_COL, TIMESTEP_COL]).reset_index(drop=True)

    state_cols      = get_state_columns(df)
    next_state_cols = ["next_" + c for c in state_cols]

    # Load injury labels
    outcomes_path = Path(__file__).parent / "Data" / "60min_merged_with_outcomes_imitation_learning_dataset.csv"
    if not outcomes_path.exists():
        raise SystemExit(f"Outcomes file not found: {outcomes_path}")
    outcomes_df = pd.read_csv(outcomes_path, usecols=[PATIENT_COL, "Any_Injury"])
    injury_map  = outcomes_df.groupby(PATIENT_COL)["Any_Injury"].first().to_dict()

    patient_ids_all = df[PATIENT_COL].to_numpy()

    # ------------------------------------------------------------------
    # 1. Sample 6 hold-out patients (3 injured, 3 uninjured)
    # ------------------------------------------------------------------
    holdout_ids, train_ids, holdout_injured, holdout_uninjured = sample_holdout_patients(
        patient_ids_all, injury_map, n_each=3
    )

    holdout_mask = np.array([pid in holdout_ids for pid in patient_ids_all])
    train_mask   = ~holdout_mask

    df_train   = df[train_mask].copy().reset_index(drop=True)
    df_holdout = df[holdout_mask].copy().reset_index(drop=True)

    print(f"\nTrain transitions:   {len(df_train)}")
    print(f"Hold-out transitions: {len(df_holdout)}")

    # ------------------------------------------------------------------
    # 2. Build arrays
    # ------------------------------------------------------------------
    X_train_raw    = df_train[state_cols].to_numpy(dtype=float)
    X_next_train_raw = df_train[next_state_cols].to_numpy(dtype=float)
    X_holdout_raw  = df_holdout[state_cols].to_numpy(dtype=float)
    X_next_holdout_raw = df_holdout[next_state_cols].to_numpy(dtype=float)

    A_train   = build_action_matrix(df_train)
    r_train   = df_train[REWARD_COL].to_numpy(dtype=float)

    is_terminal_train = np.zeros(len(df_train), dtype=bool)
    last_idx_train    = df_train.groupby(PATIENT_COL)[TIMESTEP_COL].idxmax().values
    is_terminal_train[last_idx_train] = True

    # ------------------------------------------------------------------
    # 3. Imputation (fit on train, apply to both)
    # ------------------------------------------------------------------
    X_train_imp, X_holdout_imp, impute_means = train_mean_impute(X_train_raw, X_holdout_raw)
    X_next_train_imp, _, _                    = train_mean_impute(X_next_train_raw, X_next_holdout_raw)

    # ------------------------------------------------------------------
    # 4. Scaling + KMeans (fit on train)
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)

    print(f"\nFitting KMeans with K={K_BINS} ...")
    km = KMeans(n_clusters=K_BINS, random_state=RANDOM_STATE, n_init="auto")
    km.fit(X_train_scaled)
    bin_train = km.predict(X_train_scaled)

    # ------------------------------------------------------------------
    # 5. Train FQI and SPIBB for each head, annotate holdout
    # ------------------------------------------------------------------
    # Prepare output columns
    for head_name in HEAD_NAMES:
        df_holdout[f"action_fqi_{head_name}"]   = np.nan
        df_holdout[f"action_spibb_{head_name}"] = np.nan

    for head_k, head_name in enumerate(HEAD_NAMES):
        print(f"\n{'='*60}")
        print(f"Head: {head_name}")
        print(f"{'='*60}")

        y_train = A_train[:, head_k].astype(int)

        # --- count-based baseline policy (needed for SPIBB) ---
        pi_b_bin = compute_pi_b_bin_counts(bin_train, y_train, K=K_BINS)
        support_counts = compute_support_counts(bin_train, y_train, K=K_BINS)
        count_policy_fn = make_count_policy_fn(km, scaler, pi_b_bin)

        # --- plain FQI ---
        print(f"  Training FQI ...")
        fqi = FittedQIteration()
        fqi.fit(X_train_imp, y_train, r_train, X_next_train_imp, is_terminal_train)

        fqi_actions = fqi.greedy_action(X_holdout_imp)
        df_holdout[f"action_fqi_{head_name}"] = fqi_actions

        # --- SPIBB FQI (Nwedge=5) ---
        print(f"  Training SPIBB-FQI (Nwedge={NWEDGE}) ...")
        spibb = SPIBBFittedQIteration(
            km=km,
            scaler=scaler,
            support_counts=support_counts,
            Nwedge=NWEDGE,
            baseline_policy_fn=count_policy_fn,
        )
        spibb.fit(X_train_imp, y_train, r_train, X_next_train_imp, is_terminal_train)

        spibb_actions = spibb.supported_greedy_action(X_holdout_imp, count_policy_fn)
        df_holdout[f"action_spibb_{head_name}"] = spibb_actions

        print(f"  FQI   action distribution: {np.bincount(fqi_actions, minlength=3)}")
        print(f"  SPIBB action distribution: {np.bincount(spibb_actions, minlength=3)}")

    # ------------------------------------------------------------------
    # 6. Add injury label column
    # ------------------------------------------------------------------
    df_holdout["injured"] = df_holdout[PATIENT_COL].map(lambda pid: injury_map.get(pid, np.nan))

    # ------------------------------------------------------------------
    # 7. Add human-readable action labels (optional convenience columns)
    # ------------------------------------------------------------------
    action_label = {0: "same", 1: "decrease", 2: "increase"}
    for head_name in HEAD_NAMES:
        df_holdout[f"action_fqi_{head_name}_label"]   = (
            df_holdout[f"action_fqi_{head_name}"].map(action_label)
        )
        df_holdout[f"action_spibb_{head_name}_label"] = (
            df_holdout[f"action_spibb_{head_name}"].map(action_label)
        )

    # ------------------------------------------------------------------
    # 8. Save output
    # ------------------------------------------------------------------
    out_path = Path(__file__).parent / "Data" / f"human_eval_holdout_{run_ts}.csv"
    df_holdout.to_csv(out_path, index=False)
    print(f"\nSaved hold-out annotated dataset to: {out_path}")

    # Quick summary
    print("\nHold-out summary:")
    print(f"  Total rows:        {len(df_holdout)}")
    print(f"  Unique patients:   {df_holdout[PATIENT_COL].nunique()}")
    print(f"  Injured rows:      {(df_holdout['injured'] == 1).sum()}")
    print(f"  Uninjured rows:    {(df_holdout['injured'] == 0).sum()}")
    print("\nPer-patient row counts:")
    print(df_holdout.groupby([PATIENT_COL, "injured"]).size().rename("n_timesteps").to_string())

    print("\nColumn list in output:")
    policy_cols = (
        [f"action_fqi_{h}" for h in HEAD_NAMES]
        + [f"action_fqi_{h}_label" for h in HEAD_NAMES]
        + [f"action_spibb_{h}" for h in HEAD_NAMES]
        + [f"action_spibb_{h}_label" for h in HEAD_NAMES]
        + ["injured"]
    )
    print("  " + "\n  ".join(policy_cols))


if __name__ == "__main__":
    main()
