import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import threading

"""
Per-head Offline RL on ECMO trajectories with Fitted Q Iteration (FQI)
and support-aware fallback policies.

Compared policies per head:
  1) count-based policy in train bins
  2) imitation-learning policy (optional; loaded from disk)
  3) plain FQI greedy policy
  4) support-constrained FQI with count fallback
  5) support-constrained FQI with IL fallback

Evaluation:
  - trajectory-level WIS on held-out patients
  - FQE on the held-out fold initial states (train-fitted evaluator)

Main design choices:
  - train/val split is GroupKFold by patient_id
  - preprocessing (imputation, scaling, KMeans support bins) is fit on TRAIN only
  - support is defined by train counts N(bin, action) >= Nwedge
  - behavior policy denominator for WIS is the train-bin counting policy
  - deterministic policies are epsilon-smoothed for OPE stability

This script intentionally avoids the previous model-based tabular transition model.
It uses continuous-state FQI / FQE instead.
"""

# -----------------------------
# CONFIG
# -----------------------------
# DATA_FILE = "60min_imitation_learning_terminal_only_rewards.csv"
DATA_FILE = "60min_imitation_learning_with_rewards.csv"
# DATA_FILE = "60min_imitation_learning_step_and_terminal_rewards.csv"
PATIENT_COL = "patient_id"
TIMESTEP_COL = "timestep"
REWARD_COL = "reward"


N_SPLITS = 3
RANDOM_STATE = 42
GAMMA = 0.99

# Support bins only (not for Q-learning)
# K_GRID = [50, 100]
K_GRID = [100]
# NWEDGE_GRID = [5, 10]
NWEDGE_GRID = [5,10,15]
PIB_ALPHA = 1e-3

# OPE stability
EPS = 1e-12
IS_CLIP = (1e-6, 1e3)
POLICY_SMOOTH_EPS = 0.02  # for deterministic policies during OPE
IL_TEMPERATURE = 3.0      # > 1 softens IL probabilities for OPE (reduces IS blowup)
WIS_HORIZON = 50          # truncate trajectories at this many steps for WIS

# FQI / FQE / CQL
FQI_ITERS = 10
FQE_ITERS = 10
# CQL_ALPHA = 1.0   # conservative penalty weight for CQL

# FQI_ITERS = 30
# FQE_ITERS = 30
N_ACTIONS = 3

# XGBoost defaults — cap per-model threads so 5 parallel heads don't over-subscribe
_N_HEADS = len(["PO2", "PCO2", "SpO2", "FiO2", "etCO2"])  # == len(HEAD_NAMES), defined below
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

# Semaphore to serialize GPU (CUDA) access across head threads — TabPFN
_GPU_SEM = threading.Semaphore(1)

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
# Helpers
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


def smooth_deterministic_action(a_star: np.ndarray, n_actions: int = N_ACTIONS, eps: float = POLICY_SMOOTH_EPS):
    n = len(a_star)
    P = np.full((n, n_actions), eps / n_actions, dtype=float)
    P[np.arange(n), a_star.astype(int)] += (1.0 - eps)
    return P


# -----------------------------
# IL fitting / policy
# -----------------------------
def fit_il_head_for_fold(X_train, y_train, state_features, device="cuda"):
    """Fit a two-stage TabPFN IL head on fold training data.

    y_train is the 3-class encoded action: 0=same, 1=decrease, 2=increase.
    Returns an il_head dict compatible with il_policy_proba / make_il_policy_fn.
    """
    from tabpfn import TabPFNClassifier

    act_train = (y_train > 0).astype(int)  # stage A: changed or not

    stage_a = TabPFNClassifier(device=device)
    stage_a.fit(X_train, act_train)

    stage_b = None
    tr_mask = act_train == 1
    if int(tr_mask.sum()) > 0:
        # stage B: 0=decrease (y==1), 1=increase (y==2)
        dir_tr = (y_train[tr_mask] == 2).astype(int)
        if len(np.unique(dir_tr)) > 1:
            stage_b = TabPFNClassifier(device=device)
            stage_b.fit(X_train[tr_mask], dir_tr)

    return {
        "stage_a": stage_a,
        "stage_b": stage_b,
        "thr": None,           # not used by il_policy_proba
        "state_features": state_features,
    }


def il_policy_proba(X_val, il_head):
    stage_a = il_head["stage_a"]
    stage_b = il_head["stage_b"]

    p_change = stage_a.predict_proba(X_val)[:, 1]
    p_no_change = 1.0 - p_change

    if stage_b is not None:
        p_inc_given_change = stage_b.predict_proba(X_val)[:, 1]
    else:
        p_inc_given_change = np.full(len(X_val), 0.5)

    p_same = p_no_change
    p_decrease = p_change * (1.0 - p_inc_given_change)
    p_increase = p_change * p_inc_given_change

    P = np.stack([p_same, p_decrease, p_increase], axis=1)
    P = P ** (1.0 / IL_TEMPERATURE)
    P = P / np.maximum(P.sum(axis=1, keepdims=True), EPS)

    # Diagnostic: print IL probability stats once
    # if not getattr(il_policy_proba, "_printed", False):
    #     il_policy_proba._printed = True
    #     print(f"\n[IL proba diagnostic]")
    #     print(f"  p_change:          mean={p_change.mean():.3f}  min={p_change.min():.3f}  max={p_change.max():.3f}")
    #     print(f"  p_inc_given_change: mean={p_inc_given_change.mean():.3f}  min={p_inc_given_change.min():.3f}  max={p_inc_given_change.max():.3f}")
    #     print(f"  P[:,0] same:       mean={P[:,0].mean():.3f}  min={P[:,0].min():.3f}  max={P[:,0].max():.3f}")
    #     print(f"  P[:,1] decrease:   mean={P[:,1].mean():.3f}  min={P[:,1].min():.3f}  max={P[:,1].max():.3f}")
    #     print(f"  P[:,2] increase:   mean={P[:,2].mean():.3f}  min={P[:,2].min():.3f}  max={P[:,2].max():.3f}")
    #     print(f"  argmax counts:     {np.bincount(np.argmax(P, axis=1), minlength=3)}")

    return P


# -----------------------------
# Support / behavior counts in bins
# -----------------------------
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
# FQI / FQE models
# -----------------------------
@dataclass
class MultiActionQModel:
    models: list
    n_actions: int = N_ACTIONS

    def predict_all(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([m.predict(X) for m in self.models])

    def predict_action_values(self, X: np.ndarray, a: np.ndarray) -> np.ndarray:
        Qa = self.predict_all(X)
        return Qa[np.arange(len(X)), a.astype(int)]


class FittedQIteration:
    def __init__(self, n_actions=N_ACTIONS, gamma=GAMMA, n_iters=FQI_ITERS, tree_kw=None):
        self.n_actions = n_actions
        self.gamma = gamma
        self.n_iters = n_iters
        self.tree_kw = TREE_KW if tree_kw is None else tree_kw
        self.model_ = None

    def fit(self, X, a, r, X_next, done):
        n = len(X)
        target = r.astype(float).copy()
        models = [XGBRegressor(**self.tree_kw) for _ in range(self.n_actions)]

        # Warm start with reward-only fit
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


# class CQLFittedQIteration:
#     """
#     Conservative Q-Learning (CQL) variant of FQI.

#     For each action a, the Q-regression target is:
#       - observed (s, a) pairs  → standard Bellman target
#       - unobserved (s, a) pairs → current Q(s, a) − alpha_cql
#     This pushes Q-values down for out-of-dataset actions, providing a
#     conservative lower bound without requiring an explicit behavior policy.
#     """

#     def __init__(self, n_actions=N_ACTIONS, gamma=GAMMA, n_iters=FQI_ITERS,
#                  tree_kw=None, alpha_cql=CQL_ALPHA):
#         self.n_actions = n_actions
#         self.gamma = gamma
#         self.n_iters = n_iters
#         self.tree_kw = TREE_KW if tree_kw is None else tree_kw
#         self.alpha_cql = alpha_cql
#         self.model_ = None

#     def fit(self, X, a, r, X_next, done):
#         target = r.astype(float).copy()
#         models = [XGBRegressor(**self.tree_kw) for _ in range(self.n_actions)]

#         # Warm start: fit each action model on reward only
#         for act in range(self.n_actions):
#             idx = np.where(a == act)[0]
#             if len(idx) == 0:
#                 models[act].fit(X[:1], np.array([0.0]))
#             else:
#                 models[act].fit(X[idx], target[idx])

#         q_model = MultiActionQModel(models=models, n_actions=self.n_actions)

#         for _ in tqdm(range(self.n_iters), desc="CQL", leave=False):
#             Q_next = q_model.predict_all(X_next)
#             bellman = r + self.gamma * (~done).astype(float) * np.max(Q_next, axis=1)
#             Q_curr = q_model.predict_all(X)

#             new_models = []
#             for act in range(self.n_actions):
#                 obs_idx = np.where(a == act)[0]
#                 unobs_idx = np.where(a != act)[0]

#                 # Observed (s, act): pull Q up toward Bellman target
#                 # Unobserved (s, act): push Q down by alpha_cql
#                 X_fit = np.concatenate([X[obs_idx], X[unobs_idx]], axis=0)
#                 y_obs = bellman[obs_idx]
#                 y_unobs = Q_curr[unobs_idx, act] - self.alpha_cql
#                 y_fit = np.concatenate([y_obs, y_unobs])

#                 model = XGBRegressor(**self.tree_kw)
#                 if len(obs_idx) == 0:
#                     model.fit(X[:1], np.array([0.0]))
#                 else:
#                     model.fit(X_fit, y_fit)
#                 new_models.append(model)

#             q_model = MultiActionQModel(models=new_models, n_actions=self.n_actions)

#         self.model_ = q_model
#         return self

#     def predict_all(self, X):
#         return self.model_.predict_all(X)


class FittedQEvaluation:
    # Learn the Q values of a specific policy
    def __init__(self, n_actions=N_ACTIONS, gamma=GAMMA, n_iters=FQE_ITERS, tree_kw=None):
        self.n_actions = n_actions
        self.gamma = gamma
        self.n_iters = n_iters
        self.tree_kw = TREE_KW if tree_kw is None else tree_kw
        self.model_ = None

    def fit(self, X, a, r, X_next, done, next_policy_proba_fn):
        target = r.astype(float).copy()
        models = [XGBRegressor(**self.tree_kw) for _ in range(self.n_actions)]

        for act in range(self.n_actions):
            idx = np.where(a == act)[0]
            if len(idx) == 0:
                models[act].fit(X[:1], np.array([0.0]))
            else:
                models[act].fit(X[idx], target[idx])

        q_model = MultiActionQModel(models=models, n_actions=self.n_actions)

        for _ in tqdm(range(self.n_iters), desc="FQE", leave=False):
            Q_next = q_model.predict_all(X_next)
            P_next = next_policy_proba_fn(X_next)
            V_next = np.sum(P_next * Q_next, axis=1)
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

    def value_on_initial_states(self, X0, policy_proba_fn):
        Q0 = self.model_.predict_all(X0)
        P0 = policy_proba_fn(X0)
        V0 = np.sum(P0 * Q0, axis=1)
        return float(np.mean(V0))

# -----------------------------
# SPIBB helper: Eq. (9) backup value
# -----------------------------
def spibb_projected_next_value(Q_next, P_base_next, bins_next, support_counts, Nwedge):
    """
    Approximate model-free Π_b-SPIBB backup value using your bin-based support.

    In the paper:
      B(x') = bootstrapped actions = low-count / uncertain actions
      On B(x'): follow baseline π_b
      On A \ B(x'): put all remaining mass on argmax non-bootstrapped action

    Here:
      support_counts[b, a] >= Nwedge  => non-bootstrapped / supported
      support_counts[b, a] <  Nwedge  => bootstrapped / uncertain
    """
    n, n_actions = Q_next.shape
    V_next = np.zeros(n, dtype=float)

    for i, b in enumerate(bins_next):
        b = int(b)

        # bootstrapped = uncertain = must follow baseline there
        boot_mask = support_counts[b] < int(Nwedge)
        nonboot_mask = ~boot_mask

        # baseline contribution on bootstrapped actions
        v = float(np.dot(P_base_next[i, boot_mask], Q_next[i, boot_mask]))

        # remaining baseline mass over non-bootstrapped actions
        if np.any(nonboot_mask):
            nonboot_actions = np.where(nonboot_mask)[0]
            remaining_mass = float(np.sum(P_base_next[i, nonboot_actions]))
            best_nonboot = nonboot_actions[np.argmax(Q_next[i, nonboot_actions])]
            v += remaining_mass * float(Q_next[i, best_nonboot])
        else:
            # if everything is bootstrapped, just follow baseline entirely
            v = float(np.dot(P_base_next[i], Q_next[i]))

        V_next[i] = v

    return V_next


# -----------------------------
# True-ish model-free Π_b-SPIBB for your setup
# -----------------------------
class SPIBBFittedQIteration:
    """
    Model-free Π_b-SPIBB-style fitted Q iteration.

    Important:
    - This matches the SPIBB backup structure from Eq. (9),
      but still uses your continuous features + KMeans bins as an approximation
      of the tabular state-action counts in the original paper.
    - For the original SPIBB guarantee, the baseline should be the behavior policy.
      So count_policy_fn is the safer / more faithful choice than IL fallback.
    """
    def __init__(
        self,
        km,
        scaler,
        support_counts,
        Nwedge,
        baseline_policy_fn,
        n_actions=N_ACTIONS,
        gamma=GAMMA,
        n_iters=FQI_ITERS,
        tree_kw=None,
    ):
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

        # warm start: reward-only fit
        for act in range(self.n_actions):
            idx = np.where(a == act)[0]
            if len(idx) == 0:
                models[act].fit(X[:1], np.array([0.0]))
            else:
                models[act].fit(X[idx], target[idx])

        q_model = MultiActionQModel(models=models, n_actions=self.n_actions)

        # these do not change across iterations
        X_next_scaled = self.scaler.transform(X_next)
        bins_next = self.km.predict(X_next_scaled)
        P_base_next = self.baseline_policy_fn(X_next)

        for _ in tqdm(range(self.n_iters), desc="SPIBB-FQI", leave=False):
            Q_next = q_model.predict_all(X_next)

            # SPIBB Eq. (9) backup
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
# -----------------------------
# Policy builders
# -----------------------------
def make_count_policy_fn(km, scaler, state_cols, pi_b_bin):
    def policy_fn(X_raw):
        Xs = scaler.transform(X_raw)
        bins = km.predict(Xs)
        return pi_b_bin[bins]
    return policy_fn


def make_il_policy_fn(il_head, state_cols):
    feat_cols = il_head["state_features"]
    feat_idx = [state_cols.index(c) for c in feat_cols if c in state_cols]

    def policy_fn(X_raw):
        X_sub = X_raw[:, feat_idx]
        return il_policy_proba(X_sub, il_head)
    return policy_fn


def make_plain_fqi_policy_fn(q_model: FittedQIteration, eps=POLICY_SMOOTH_EPS):
    def policy_fn(X_raw):
        Q = q_model.predict_all(X_raw)
        a_star = np.argmax(Q, axis=1)
        return smooth_deterministic_action(a_star, eps=eps)
    return policy_fn



def make_supported_fqi_policy_fn(q_model, km, scaler, support_counts, Nwedge, baseline_policy_fn):
    K, n_actions = support_counts.shape
    _cache = {}  # id(X_raw) -> P  — Q/bins/P_base are fixed for a given input array

    def policy_fn(X_raw):
        key = id(X_raw)
        if key in _cache:
            return _cache[key]
        Xs = scaler.transform(X_raw)
        bins = km.predict(Xs)
        Q = q_model.predict_all(X_raw)
        P_base = baseline_policy_fn(X_raw)
        P = np.zeros_like(P_base)

        for i, b in enumerate(bins):
            supported = np.where(support_counts[int(b)] >= int(Nwedge))[0]

            # If nothing is supported, keep the baseline policy unchanged
            if supported.size == 0:
                P[i] = P_base[i]
                continue

            # Start by keeping baseline mass on unsupported actions
            P[i] = 0.0
            unsupported = np.setdiff1d(np.arange(n_actions), supported)
            if unsupported.size > 0:
                P[i, unsupported] = P_base[i, unsupported]

            # Total baseline mass available on supported actions
            supported_mass = float(np.sum(P_base[i, supported]))

            # Put all supported mass on the best supported action
            best_a = supported[np.argmax(Q[i, supported])]
            P[i, best_a] += supported_mass

            # Numerical guard
            row_sum = P[i].sum()
            if row_sum <= EPS:
                P[i] = P_base[i]
            else:
                P[i] /= row_sum

        _cache[key] = P
        return P

    return policy_fn

# def make_supported_fqi_policy_fn(q_model, km, scaler, support_counts, Nwedge, fallback_policy_fn, eps=POLICY_SMOOTH_EPS):
#     K, n_actions = support_counts.shape

#     def policy_fn(X_raw):
#         Xs = scaler.transform(X_raw)
#         bins = km.predict(Xs)
#         Q = q_model.predict_all(X_raw)
#         P_fb = fallback_policy_fn(X_raw)
#         P = np.zeros_like(P_fb)

#         for i, b in enumerate(bins):
#             supported = np.where(support_counts[int(b)] >= int(Nwedge))[0]
#             if supported.size == 0:
#                 P[i] = P_fb[i]
#             else:
#                 best_a = supported[np.argmax(Q[i, supported])]
#                 P[i] = smooth_deterministic_action(np.array([best_a]), n_actions=n_actions, eps=eps)[0]
#         return P
#     return policy_fn


def compute_fallback_rate(X_val, km, scaler, support_counts, Nwedge, q_model):    
    Xs = scaler.transform(X_val)
    bins = km.predict(Xs)
    Q = q_model.predict_all(X_val)
    fallback = []
    for i, b in enumerate(bins):
        supported = np.where(support_counts[int(b)] >= int(Nwedge))[0]
        if supported.size == 0:
            fallback.append(1)
        else:
            greedy_supported = supported[np.argmax(Q[i, supported])]
            greedy_plain = np.argmax(Q[i])
            fallback.append(int(greedy_plain not in supported))
    return float(np.mean(fallback))


# -----------------------------
# OPE: WIS
# -----------------------------
def wis_trajectory_level_logstable(patient_ids, r, pi_e_a, pi_b_a, clip_ratio=IS_CLIP, horizon=WIS_HORIZON):
    unique_p = np.unique(patient_ids)
    pi_b_a = np.clip(pi_b_a, EPS, None)
    pi_e_a = np.clip(pi_e_a, EPS, None)
    ratios = np.clip(pi_e_a / pi_b_a, clip_ratio[0], clip_ratio[1])

    logw, G = [], []
    for pid in unique_p:
        # idx = np.where(patient_ids == pid)[0][:horizon]   # truncate at horizon
        idx = np.where(patient_ids == pid)[0] # the whole horizon
        logw.append(float(np.sum(np.log(ratios[idx]))))
        G.append(float(np.sum(r[idx])))

    logw = np.array(logw, dtype=float)
    G = np.array(G, dtype=float)
    m = np.max(logw)
    w = np.exp(logw - m)
    denom = np.sum(w)
    if denom <= EPS:
        return float(np.mean(G))
    return float(np.sum(w * G) / denom)


def _stratified_dev(P_eval, y_expert, inj_mask, no_inj_mask):
    """Return (dev_injury, dev_no_injury) deviation rates vs. expert actions in data."""
    hard_dev = np.argmax(P_eval, axis=1) != y_expert
    dev_inj    = float(np.mean(hard_dev[inj_mask]))    if inj_mask.any()    else float("nan")
    dev_no_inj = float(np.mean(hard_dev[no_inj_mask])) if no_inj_mask.any() else float("nan")
    return dev_inj, dev_no_inj


# -----------------------------
# Per-head worker (runs in a thread)
# -----------------------------
def _run_head(head_k, head_name, fold,
              X_train_imp, X_val_imp, X_next_train_imp,
              X_train_scaled, X_val_scaled,
              A_train, A_val, r_train, r_val,
              pid_val, done_train, inj_mask, no_inj_mask,
              X0_val_raw, V_logged_mean_return,
              n_val, state_cols, scaler, USE_IL):
    """Evaluate all policies for one (fold, head). Returns list of row dicts."""
    print(f"  [Fold {fold}] Starting head: {head_name}")
    y_train = A_train[:, head_k].astype(int)
    y_val   = A_val[:,   head_k].astype(int)

    # --- FQI ---
    fqi = FittedQIteration()
    fqi.fit(X_train_imp, y_train, r_train, X_next_train_imp, done_train)
    plain_fqi_policy_fn = make_plain_fqi_policy_fn(fqi)

    # --- IL (TabPFN): fit once per head — serialized via semaphore to avoid GPU contention ---
    il_policy_fn = None
    if USE_IL:
        with _GPU_SEM:
            il_head_obj = fit_il_head_for_fold(X_train_imp, y_train, state_cols)
        il_policy_fn = make_il_policy_fn(il_head_obj, state_cols)

    # --- IL FQE: computed ONCE outside K loop (opt 3 — doesn't depend on K) ---
    il_fqe_result = None
    if il_policy_fn is not None:
        il_fqe = FittedQEvaluation()
        il_fqe.fit(X_train_imp, y_train, r_train, X_next_train_imp, done_train, il_policy_fn)
        P_val_il      = il_policy_fn(X_val_imp)
        V_fqe_il      = il_fqe.value_on_initial_states(X0_val_raw, il_policy_fn)
        d_il, dni_il  = _stratified_dev(P_val_il, y_val, inj_mask, no_inj_mask)
        il_fqe_result = {
            "V_fqe":        V_fqe_il,
            "P_val":        P_val_il,
            "dev_rate":     float(np.mean(np.argmax(P_val_il, axis=1) != y_val)),
            "dev_injury":   d_il,
            "dev_no_injury": dni_il,
        }

    head_rows = []
    for K in K_GRID:
        km = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init="auto")
        km.fit(X_train_scaled)
        bin_train = km.predict(X_train_scaled)

        pi_b_bin       = compute_pi_b_bin_counts(bin_train, y_train, K=K)
        support_counts = compute_support_counts(bin_train, y_train, K=K)
        count_policy_fn = make_count_policy_fn(km, scaler, state_cols, pi_b_bin)

        P_b_val         = count_policy_fn(X_val_imp)
        pi_b_logged_val = np.clip(P_b_val[np.arange(len(y_val)), y_val], EPS, None)

        # Fixed policies independent of Nwedge (count + fqi_plain)
        policy_bank = {
            "count":     count_policy_fn,
            "fqi_plain": plain_fqi_policy_fn,
        }
        fixed_results = {}
        for pol_name, pol_fn in policy_bank.items():
            P_val        = pol_fn(X_val_imp)
            pi_e_logged  = np.clip(P_val[np.arange(len(y_val)), y_val], EPS, None)
            V_wis        = wis_trajectory_level_logstable(pid_val, r_val, pi_e_logged, pi_b_logged_val)
            fqe = FittedQEvaluation()
            fqe.fit(X_train_imp, y_train, r_train, X_next_train_imp, done_train, pol_fn)
            V_fqe        = fqe.value_on_initial_states(X0_val_raw, pol_fn)
            d, d_ni      = _stratified_dev(P_val, y_val, inj_mask, no_inj_mask)
            fixed_results[pol_name] = {
                "V_wis": V_wis, "V_fqe": V_fqe,
                "dev_rate": float(np.mean(np.argmax(P_val, axis=1) != y_val)),
                "dev_injury": d, "dev_no_injury": d_ni,
            }

        # IL fixed result: reuse pre-computed FQE; only WIS is K-dependent (via pi_b denominator)
        if il_fqe_result is not None:
            pi_e_il  = np.clip(il_fqe_result["P_val"][np.arange(len(y_val)), y_val], EPS, None)
            V_wis_il = wis_trajectory_level_logstable(pid_val, r_val, pi_e_il, pi_b_logged_val)
            fixed_results["il"] = {
                "V_wis":        V_wis_il,
                "V_fqe":        il_fqe_result["V_fqe"],
                "dev_rate":     il_fqe_result["dev_rate"],
                "dev_injury":   il_fqe_result["dev_injury"],
                "dev_no_injury": il_fqe_result["dev_no_injury"],
            }

        for Nwedge in NWEDGE_GRID:
            # -------------------------
            # count-baseline Π_b-SPIBB
            # -------------------------
            spibb_count = SPIBBFittedQIteration(
                km=km,
                scaler=scaler,
                support_counts=support_counts,
                Nwedge=int(Nwedge),
                baseline_policy_fn=count_policy_fn,   # closest to original SPIBB
            )
            spibb_count.fit(X_train_imp, y_train, r_train, X_next_train_imp, done_train)

            spibb_count_fn = make_supported_fqi_policy_fn(
                q_model=spibb_count,
                km=km,
                scaler=scaler,
                support_counts=support_counts,
                Nwedge=int(Nwedge),
                baseline_policy_fn=count_policy_fn,
            )

            candidate_policies = {"spibb_count": spibb_count_fn}
            # if il_policy_fn is not None:
            #     fqi_il_fb_fn = make_supported_fqi_policy_fn(
            #         q_model=fqi, km=km, scaler=scaler, support_counts=support_counts,
            #         Nwedge=int(Nwedge), baseline_policy_fn=il_policy_fn,
            #     )
            #     candidate_policies["fqi_il_fb"] = fqi_il_fb_fn

            dynamic_results = {}
            for pol_name, pol_fn in candidate_policies.items():
                P_val = pol_fn(X_val_imp)
                pi_e_logged = np.clip(P_val[np.arange(len(y_val)), y_val], EPS, None)

                V_wis = wis_trajectory_level_logstable(pid_val, r_val, pi_e_logged, pi_b_logged_val)

                fqe = FittedQEvaluation()
                fqe.fit(X_train_imp, y_train, r_train, X_next_train_imp, done_train, pol_fn)
                V_fqe = fqe.value_on_initial_states(X0_val_raw, pol_fn)

                d, d_ni = _stratified_dev(P_val, y_val, inj_mask, no_inj_mask)

                # use the matching SPIBB model for fallback diagnostics
                q_for_fb = spibb_count

                dynamic_results[pol_name] = {
                    "V_wis": V_wis,
                    "V_fqe": V_fqe,
                    "dev_rate": float(np.mean(np.argmax(P_val, axis=1) != y_val)),
                    "dev_injury": d,
                    "dev_no_injury": d_ni,
                    "fallback_rate": compute_fallback_rate(
                        X_val_imp, km, scaler, support_counts, Nwedge, q_for_fb
                    ),
                }

            row = {
                "fold": fold,
                "head": head_name,
                "K_bins": int(K),
                "Nwedge": int(Nwedge),
                "n_val_transitions": n_val,
                "n_val_patients": int(len(np.unique(pid_val))),
                "V_logged_mean_return": V_logged_mean_return,

                "V_wis_count": fixed_results["count"]["V_wis"],
                "V_fqe_count": fixed_results["count"]["V_fqe"],
                "dev_count": fixed_results["count"]["dev_rate"],
                "dev_count_injury": fixed_results["count"]["dev_injury"],
                "dev_count_no_injury": fixed_results["count"]["dev_no_injury"],

                "V_wis_fqi_plain": fixed_results["fqi_plain"]["V_wis"],
                "V_fqe_fqi_plain": fixed_results["fqi_plain"]["V_fqe"],
                "dev_fqi_plain": fixed_results["fqi_plain"]["dev_rate"],
                "dev_fqi_plain_injury": fixed_results["fqi_plain"]["dev_injury"],
                "dev_fqi_plain_no_injury": fixed_results["fqi_plain"]["dev_no_injury"],

                # renamed to SPIBB for clarity
                "V_wis_spibb_count": dynamic_results["spibb_count"]["V_wis"],
                "V_fqe_spibb_count": dynamic_results["spibb_count"]["V_fqe"],
                "dev_spibb_count": dynamic_results["spibb_count"]["dev_rate"],
                "dev_spibb_count_injury": dynamic_results["spibb_count"]["dev_injury"],
                "dev_spibb_count_no_injury": dynamic_results["spibb_count"]["dev_no_injury"],
                "fallback_rate_spibb_count": dynamic_results["spibb_count"]["fallback_rate"],
            }

            if "il" in fixed_results:
                row.update({
                    "V_wis_il":             fixed_results["il"]["V_wis"],
                    "V_fqe_il":             fixed_results["il"]["V_fqe"],
                    "dev_il":               fixed_results["il"]["dev_rate"],
                    "dev_il_injury":        fixed_results["il"]["dev_injury"],
                    "dev_il_no_injury":     fixed_results["il"]["dev_no_injury"],
                    # "V_wis_fqi_il_fb":      dynamic_results["fqi_il_fb"]["V_wis"],
                    # "V_fqe_fqi_il_fb":      dynamic_results["fqi_il_fb"]["V_fqe"],
                    # "dev_fqi_il_fb":        dynamic_results["fqi_il_fb"]["dev_rate"],
                    # "dev_fqi_il_fb_injury": dynamic_results["fqi_il_fb"]["dev_injury"],
                    # "dev_fqi_il_fb_no_injury": dynamic_results["fqi_il_fb"]["dev_no_injury"],
                    # "fallback_rate_il":     dynamic_results["fqi_il_fb"]["fallback_rate"],
                })
            else:
                row.update({
                    "V_wis_il": np.nan,         "V_fqe_il": np.nan,
                    "dev_il": np.nan,            "dev_il_injury": np.nan,   "dev_il_no_injury": np.nan,
                    # "V_wis_fqi_il_fb": np.nan,  "V_fqe_fqi_il_fb": np.nan,
                    # "dev_fqi_il_fb": np.nan,    "dev_fqi_il_fb_injury": np.nan,
                    # "dev_fqi_il_fb_no_injury": np.nan, 
                    # "fallback_rate_il": np.nan,
                })

            head_rows.append(row)
            print(f"  [Fold {fold} | {head_name}] K={K} Nwedge={Nwedge} — done")

        print(f"  [Fold {fold}] Done K={K}")

    print(f"  [Fold {fold}] {head_name} complete ({len(head_rows)} rows)")
    return head_rows


# -----------------------------
# Main
# -----------------------------
def main():
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    data_path = Path(__file__).parent / "Data" / DATA_FILE
    if not data_path.exists():
        raise SystemExit(f"File not found: {data_path}")

    df = pd.read_csv(data_path)
    if PATIENT_COL not in df.columns or TIMESTEP_COL not in df.columns:
        raise ValueError(f"Missing required columns: {PATIENT_COL}, {TIMESTEP_COL}")

    # Explicit sort so trajectory OPE uses real time order
    df = df.sort_values([PATIENT_COL, TIMESTEP_COL]).reset_index(drop=True)

    state_cols = get_state_columns(df)
    next_state_cols = ["next_" + c for c in state_cols]

    if REWARD_COL not in df.columns:
        raise ValueError(f"Missing reward column: {REWARD_COL}")

    X_all_raw = df[state_cols].to_numpy(dtype=float)
    X_next_all_raw = df[next_state_cols].to_numpy(dtype=float)
    A_all = build_action_matrix(df)
    r_all = df[REWARD_COL].to_numpy(dtype=float)
    patient_ids_all = df[PATIENT_COL].to_numpy()

    # Load per-patient Any_Injury flag from outcomes file
    outcomes_path = Path(__file__).parent / "Data" / "60min_merged_with_outcomes_imitation_learning_dataset.csv"
    if outcomes_path.exists():
        outcomes_df = pd.read_csv(outcomes_path, usecols=[PATIENT_COL, "Any_Injury"])
        injury_map = outcomes_df.groupby(PATIENT_COL)["Any_Injury"].first().to_dict()
        injury_all = np.array([injury_map.get(pid, np.nan) for pid in patient_ids_all], dtype=float)
        print(f"Loaded Any_Injury for {len(injury_map)} patients")
    else:
        injury_all = np.full(len(patient_ids_all), np.nan)
        print("WARNING: outcomes file not found; injury stratification will be NaN")

    is_terminal_all = np.zeros(len(df), dtype=bool)
    last_idx = df.groupby(PATIENT_COL)[TIMESTEP_COL].idxmax().values
    is_terminal_all[last_idx] = True

    USE_IL = True  # set False to skip TabPFN IL fitting entirely

    print("=" * 100)
    print("Loaded:", data_path.name)
    print(f"Transitions: {len(df)} | Patients: {len(np.unique(patient_ids_all))}")
    print(f"State dims: {len(state_cols)}")
    print("Heads:", HEAD_NAMES)
    print(f"Support grids: K={K_GRID}, Nwedge={NWEDGE_GRID}")
    print("=" * 100)

    gkf = GroupKFold(n_splits=N_SPLITS)
    rows = []

    fold_iter = tqdm(enumerate(gkf.split(X_all_raw, groups=patient_ids_all), start=1),
                     total=N_SPLITS, desc="Folds")
    for fold, (train_idx, val_idx) in fold_iter:
        fold_iter.set_postfix(fold=fold)
        print(f"\n--- Fold {fold}/{N_SPLITS} ---")

        pid_train = patient_ids_all[train_idx]
        pid_val = patient_ids_all[val_idx]

        X_train_raw = X_all_raw[train_idx]
        X_val_raw = X_all_raw[val_idx]
        X_next_train_raw = X_next_all_raw[train_idx]
        X_next_val_raw = X_next_all_raw[val_idx]

        # train-only imputation
        X_train_imp, X_val_imp, impute_means = train_mean_impute(X_train_raw, X_val_raw)
        X_next_train_imp, X_next_val_imp, _ = train_mean_impute(X_next_train_raw, X_next_val_raw)
       
        r_train = r_all[train_idx]
        r_val = r_all[val_idx]

        A_train = A_all[train_idx]
        A_val = A_all[val_idx]

        done_train = is_terminal_all[train_idx]
        done_val = is_terminal_all[val_idx]

        injury_val = injury_all[val_idx]          # 1=injury, 0=no injury, nan=unknown
        inj_mask  = injury_val == 1
        no_inj_mask = injury_val == 0

        # scaler only for support binning
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_val_scaled = scaler.transform(X_val_imp)

        # held-out start states for FQE reporting
        first_val_idx = df.iloc[val_idx].groupby(PATIENT_COL)[TIMESTEP_COL].idxmin().values
        X0_val_raw = df.loc[first_val_idx, state_cols].to_numpy(dtype=float)
        mask0 = np.isnan(X0_val_raw)
        if mask0.any():
            X0_val_raw[mask0] = np.take(impute_means, np.where(mask0)[1])

        logged_returns = []
        for pid in np.unique(pid_val):
            idxp = np.where(pid_val == pid)[0]
            logged_returns.append(float(np.sum(r_val[idxp])))   # sum of all rewards of one patient
        V_logged_mean_return = float(np.mean(logged_returns)) if logged_returns else float(np.mean(r_val))

        # Run all heads in parallel — one thread per head (opt 2)
        with ThreadPoolExecutor(max_workers=len(HEAD_NAMES)) as executor:
            futures = {
                executor.submit(
                    _run_head,
                    head_k, head_name, fold,
                    X_train_imp, X_val_imp, X_next_train_imp,
                    X_train_scaled, X_val_scaled,
                    A_train, A_val, r_train, r_val,
                    pid_val, done_train, inj_mask, no_inj_mask,
                    X0_val_raw, V_logged_mean_return,
                    int(len(val_idx)), state_cols, scaler, USE_IL,
                ): head_name
                for head_k, head_name in enumerate(HEAD_NAMES)
            }
            for future in as_completed(futures):
                head_name_done = futures[future]
                head_rows = future.result()
                rows.extend(head_rows)
                print(f"  ✓ Fold {fold} | {head_name_done}: {len(head_rows)} rows collected")

        # Checkpoint after all heads in this fold complete
        out = Path(__file__).parent / "Data" / f"fqi_support_fallback_{Path(DATA_FILE).stem}_{run_ts}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  [checkpoint] {len(rows)} rows saved after fold {fold}")

    res = pd.DataFrame(rows)
    out = Path(__file__).parent / "Data" / f"fqi_support_fallback_{Path(DATA_FILE).stem}_{run_ts}.csv"
    res.to_csv(out, index=False)
    print("\nSaved fold-level results to:", out)

    agg = (
    res.groupby(["head", "K_bins", "Nwedge"], as_index=False)
       .agg(
           mean_logged_return=("V_logged_mean_return", "mean"),
           mean_wis_count=("V_wis_count", "mean"),
           mean_fqe_count=("V_fqe_count", "mean"),
           mean_wis_il=("V_wis_il", "mean"),
           mean_fqe_il=("V_fqe_il", "mean"),
           mean_wis_fqi_plain=("V_wis_fqi_plain", "mean"),
           mean_fqe_fqi_plain=("V_fqe_fqi_plain", "mean"),
           mean_wis_spibb_count=("V_wis_spibb_count", "mean"),
           mean_fqe_spibb_count=("V_fqe_spibb_count", "mean"),
           mean_dev_fqi_plain=("dev_fqi_plain", "mean"),
           mean_dev_spibb_count=("dev_spibb_count", "mean"),
           mean_fb_rate_spibb_count=("fallback_rate_spibb_count", "mean"),
           folds=("fold", "nunique"),
       )
    )

    summary_out = Path(__file__).parent / "Data" / f"fqi_support_fallback_summary_{Path(DATA_FILE).stem}_{run_ts}.csv"
    agg.to_csv(summary_out, index=False)
    print("Saved summary results to:", summary_out)

    print("\n" + "=" * 100)
    print("Top configs per head by mean_fqe_spibb_count")
    print("=" * 100)
    for head in agg["head"].unique():
        top = (
            agg[agg["head"] == head]
            .sort_values("mean_fqe_spibb_count", ascending=False)
            .head(5)
        )
        print(f"\n[{head}]")
        print(top[[
            "K_bins", "Nwedge",
            "mean_logged_return",
            "mean_fqe_count", "mean_fqe_il",
            "mean_fqe_fqi_plain",
            "mean_fqe_spibb_count",
            "mean_fb_rate_spibb_count",
            "folds"
        ]].to_string(index=False))


if __name__ == "__main__":
    main()
