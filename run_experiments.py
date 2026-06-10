"""
Experiment runner for LOX and LOX-R evaluation.

Compares LOX (perturbation-based) and LOX-R (rank-aware conditional) against
standard baselines: SHAP-Permutation, LIME-ScoreReg, PFI-Rank, and LOFO-Rank.

Usage:
    pip install -e ".[explain]"
    python run_experiments.py --datasets cancer adult credit heart
"""
import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional
from urllib.request import urlopen

import numpy as np
import pandas as pd
import shap
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.iforest import IForest
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from scipy.stats import kendalltau
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_covtype, fetch_openml, load_breast_cancer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pyxod import LOXExplainer, LOXRExplainer

RNG = np.random.default_rng(42)

FRED_MD_URL = (
    "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/"
    "fred-md/monthly/2026-02-md.csv"
)
FRED_RECESSION_WINDOWS = [
    ("2007-12-01", "2009-06-30"),
    ("2020-02-01", "2020-04-30"),
    ("2022-03-01", "2023-12-31"),
]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

@dataclass
class DatasetBundle:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    true_context_features: Optional[List[int]] = None


def load_dataset(name: str) -> DatasetBundle:
    n = name.lower()
    if n == "cancer":
        b = load_breast_cancer(as_frame=True)
        return DatasetBundle("cancer", b.data, (b.target == 0).astype(int).to_numpy())
    if n == "adult":
        d = fetch_openml("adult", version=2, as_frame=True)
        y = d.target.astype(str).str.contains(">50K").astype(int).to_numpy()
        return DatasetBundle("adult", d.data, y)
    if n in ("credit", "german_credit"):
        d = fetch_openml("credit-g", version=1, as_frame=True)
        y = (d.target.astype(str) == "bad").astype(int).to_numpy()
        return DatasetBundle("credit", d.data, y)
    if n in ("heart", "heart_statlog"):
        d = fetch_openml("heart-statlog", version=1, as_frame=True)
        y_raw = d.target.astype(str).to_numpy()
        y = (y_raw == np.unique(y_raw)[-1]).astype(int)
        return DatasetBundle("heart", d.data, y)
    if n in ("bank", "bank_marketing"):
        d = fetch_openml("bank-marketing", version=1, as_frame=True)
        y = (d.target.astype(str) == "yes").astype(int).to_numpy()
        return DatasetBundle("bank_marketing", d.data, y)
    if n in ("covertype", "forest_cover"):
        d = fetch_covtype(as_frame=True)
        y = (d.target.astype(int) == 4).astype(int).to_numpy()
        return DatasetBundle("covertype", d.data, y)
    if n in ("mammography", "mammography_openml"):
        X, y_raw = _fetch_openml_with_fallback(["mammography", "Mammography"], [None, 1])
        return DatasetBundle("mammography", X, _minority_as_anomaly(y_raw))
    if n in ("shuttle",):
        X, y_raw = _fetch_openml_with_fallback(["shuttle"], [None, 1])
        return DatasetBundle("shuttle", X, _minority_as_anomaly(y_raw))
    if n in ("satimage", "satimage2", "satimage-2"):
        X, y_raw = _fetch_openml_with_fallback(["satimage", "satimage-2", "satimage2"], [None, 1])
        return DatasetBundle("satimage", X, _minority_as_anomaly(y_raw))
    if n in ("synthetic_context",):
        return _make_synthetic_contextual()
    if n in ("synthetic_guaranteed",):
        return _make_synthetic_guaranteed()
    if n in ("fred_md", "fredmd"):
        return _load_fred_md()
    raise ValueError(f"Unknown dataset: {name}")


def _fetch_openml_with_fallback(names, versions):
    for nm in names:
        for v in versions:
            try:
                d = fetch_openml(nm, version=v, as_frame=True)
                return d.data, pd.Series(d.target)
            except Exception:
                continue
    raise RuntimeError(f"OpenML fetch failed for {names}")


def _minority_as_anomaly(y_raw: pd.Series) -> np.ndarray:
    y = y_raw.astype(str)
    vc = y.value_counts(dropna=False)
    return (y == vc.index[-1]).astype(int).to_numpy()


def _load_fred_md() -> DatasetBundle:
    lp = Path(__file__).resolve().parent / "data" / "raw" / "fred_md" / "2026-02-md.csv"
    if lp.exists():
        df = pd.read_csv(lp)
    else:
        lp.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(FRED_MD_URL, timeout=120) as r:
            lp.write_bytes(r.read())
        df = pd.read_csv(lp)
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "sasdate"})
    parsed = pd.to_datetime(df["sasdate"], format="%m/%d/%Y", errors="coerce")
    if parsed.notna().sum() == 0:
        parsed = pd.to_datetime(df["sasdate"], errors="coerce")
    df["sasdate"] = parsed
    df = df[df["sasdate"].notna()].sort_values("sasdate")
    X = df.drop(columns=["sasdate"])
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(X.median(numeric_only=True))
    dates = df["sasdate"].reset_index(drop=True)
    y = np.zeros(len(dates), dtype=int)
    for start, end in FRED_RECESSION_WINDOWS:
        y[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))] = 1
    return DatasetBundle("fred_md", X.reset_index(drop=True), y)


def _make_synthetic_contextual(n_inliers=4000, n_outliers=120, noise=0.08, seed=42):
    rng = np.random.default_rng(seed)
    z0 = rng.normal(0, 1.0, n_inliers)
    z1 = rng.normal(0, 1.0, n_inliers)
    z2 = rng.normal(0, 1.0, n_inliers)
    x0 = z0 + rng.normal(0, noise, n_inliers)
    x1 = 1.8 * x0 + 0.2 * z1 + rng.normal(0, noise, n_inliers)
    x2 = np.sin(z1) + 0.3 * z2 + rng.normal(0, noise, n_inliers)
    x3 = x1 * x2 + rng.normal(0, noise, n_inliers)
    x4 = z2 + rng.normal(0, noise, n_inliers)
    x5 = 0.5 * x2 - 0.3 * x4 + rng.normal(0, noise, n_inliers)
    inliers = np.column_stack([x0, x1, x2, x3, x4, x5])

    zo = rng.normal(0, 1.0, n_outliers)
    yo = rng.normal(0, 1.0, n_outliers)
    u0 = zo + rng.normal(0, noise, n_outliers)
    u1 = -1.8 * u0 + 0.2 * yo + rng.normal(0, noise, n_outliers)
    u2 = np.sin(yo) + rng.normal(0, noise, n_outliers)
    u3 = -u1 * u2 + rng.normal(0, noise, n_outliers)
    u4 = rng.normal(0, 1.0, n_outliers)
    u5 = 0.5 * u2 - 0.3 * u4 + rng.normal(0, noise, n_outliers)
    outliers = np.column_stack([u0, u1, u2, u3, u4, u5])

    X = np.vstack([inliers, outliers])
    y = np.concatenate([np.zeros(n_inliers, dtype=int), np.ones(n_outliers, dtype=int)])
    cols = [f"f{i}" for i in range(X.shape[1])]
    return DatasetBundle("synthetic_context", pd.DataFrame(X, columns=cols), y, true_context_features=[1, 3])


def _make_synthetic_guaranteed(n_inliers=4500, n_outliers=140, noise=0.03, seed=42):
    rng = np.random.default_rng(seed)
    c0 = rng.uniform(-2.5, 2.5, n_inliers)
    c1 = rng.uniform(-2.0, 2.0, n_inliers)
    f1 = np.sin(1.4 * c0) + rng.normal(0, noise, n_inliers)
    f3 = 0.7 * np.cos(1.2 * c1) + rng.normal(0, noise, n_inliers)
    f4 = 0.4 * c0 - 0.3 * c1 + rng.normal(0, 0.1, n_inliers)
    f5 = rng.normal(0, 1.0, n_inliers)
    inliers = np.column_stack([c0, f1, c1, f3, f4, f5])

    co0 = rng.uniform(-2.5, 2.5, n_outliers)
    co1 = rng.uniform(-2.0, 2.0, n_outliers)
    g1 = -np.sin(1.4 * co0) + rng.normal(0, noise, n_outliers)
    g3 = -0.7 * np.cos(1.2 * co1) + rng.normal(0, noise, n_outliers)
    g4 = 0.4 * co0 - 0.3 * co1 + rng.normal(0, 0.1, n_outliers)
    g5 = rng.normal(0, 1.0, n_outliers)
    outliers = np.column_stack([co0, g1, co1, g3, g4, g5])

    X = np.vstack([inliers, outliers])
    y = np.concatenate([np.zeros(n_inliers, dtype=int), np.ones(n_outliers, dtype=int)])
    cols = [f"f{i}" for i in range(X.shape[1])]
    return DatasetBundle("synthetic_guaranteed", pd.DataFrame(X, columns=cols), y, true_context_features=[1, 3])


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(X: pd.DataFrame) -> np.ndarray:
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    tr = ColumnTransformer(
        [
            ("num", Pipeline([("sc", StandardScaler())]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )
    Xt = tr.fit_transform(X)
    return Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt, dtype=float)


def cap_size(X: np.ndarray, y: np.ndarray, max_samples: int) -> tuple[np.ndarray, np.ndarray]:
    if len(y) <= max_samples:
        return X, y
    idx_o = np.where(y == 1)[0]
    idx_n = np.where(y == 0)[0]
    frac = len(idx_o) / max(1, len(y))
    n_o = max(30, int(max_samples * frac))
    n_o = min(n_o, len(idx_o))
    n_n = max(1, max_samples - n_o)
    idx = np.concatenate([
        RNG.choice(idx_o, size=min(n_o, len(idx_o)), replace=False),
        RNG.choice(idx_n, size=min(n_n, len(idx_n)), replace=False),
    ])
    RNG.shuffle(idx)
    return X[idx], y[idx]


# ---------------------------------------------------------------------------
# Baseline explainers
# ---------------------------------------------------------------------------

def detector_score_fn(detector) -> Callable[[np.ndarray], np.ndarray]:
    return lambda Z: detector.decision_function(np.asarray(Z, dtype=float)).ravel()


def local_lofo_rank(x, X_ref, score_fn) -> np.ndarray:
    ref_scores = score_fn(X_ref)
    base = float(score_fn(x.reshape(1, -1))[0])
    r0 = rank_from_scores(ref_scores, base)
    med = np.median(X_ref, axis=0)
    out = np.zeros(x.shape[0], dtype=float)
    for j in range(x.shape[0]):
        z = x.copy()
        z[j] = med[j]
        r = rank_from_scores(ref_scores, float(score_fn(z.reshape(1, -1))[0]))
        out[j] = abs(r - r0)
    return out


def local_pfi_rank(x, X_ref, score_fn, n_perturb: int = 32) -> np.ndarray:
    ref_scores = score_fn(X_ref)
    r0 = rank_from_scores(ref_scores, float(score_fn(x.reshape(1, -1))[0]))
    out = np.zeros(x.shape[0], dtype=float)
    for j in range(x.shape[0]):
        deltas = []
        col = X_ref[:, j]
        for _ in range(n_perturb):
            z = x.copy()
            z[j] = col[RNG.integers(0, len(col))]
            r = rank_from_scores(ref_scores, float(score_fn(z.reshape(1, -1))[0]))
            deltas.append(abs(r - r0))
        out[j] = float(np.mean(deltas))
    return out


def lime_regressor_score(x, X_ref, score_fn, n_samples: int = 200) -> np.ndarray:
    d = x.shape[0]
    samples = np.tile(x, (n_samples, 1))
    sigma = 0.1 * (np.std(X_ref, axis=0) + 1e-12)
    samples += RNG.normal(0.0, sigma, size=samples.shape)
    y = score_fn(samples)
    w = np.exp(-np.linalg.norm(samples - x.reshape(1, -1), axis=1) / (np.sqrt(d) + 1e-12))
    reg = Ridge(alpha=1.0)
    reg.fit(samples - x.reshape(1, -1), y, sample_weight=w)
    return np.abs(reg.coef_)


def shap_permutation_score(x, X_bg, score_fn) -> np.ndarray:
    masker = shap.maskers.Independent(X_bg[: min(120, len(X_bg))])
    explainer = shap.PermutationExplainer(score_fn, masker=masker)
    exp = explainer(x.reshape(1, -1), max_evals=2 * x.shape[0] + 64)
    return np.abs(np.asarray(exp.values).ravel())


# ---------------------------------------------------------------------------
# White-box reference (ground truth)
# ---------------------------------------------------------------------------

def whitebox_local_rank_reference(x, X_ref, score_fn, knn_k=40) -> np.ndarray:
    d = np.linalg.norm(X_ref - x.reshape(1, -1), axis=1)
    nbr_idx = np.argsort(d)[: min(knn_k, len(d))]
    local_mean = np.mean(X_ref[nbr_idx], axis=0)
    ref_scores = score_fn(X_ref)
    r0 = rank_from_scores(ref_scores, float(score_fn(x.reshape(1, -1))[0]))
    out = np.zeros(x.shape[0], dtype=float)
    for j in range(x.shape[0]):
        z = x.copy()
        z[j] = local_mean[j]
        r = rank_from_scores(ref_scores, float(score_fn(z.reshape(1, -1))[0]))
        out[j] = abs(r - r0)
    return out


def iforest_diffi_reference(detector: IForest, x: np.ndarray) -> np.ndarray:
    contrib = np.zeros_like(x, dtype=float)
    det = detector.detector_
    for tree in det.estimators_:
        path = tree.decision_path(x.reshape(1, -1))
        nodes = path.indices[path.indptr[0] : path.indptr[1]]
        depth = 0
        features = tree.tree_.feature
        for node in nodes:
            f = int(features[node])
            if f >= 0:
                contrib[f] += 1.0 / (depth + 1.0)
            depth += 1
    return contrib / (np.sum(contrib) + 1e-12)


def whitebox_detector_reference(detector, x, X_ref, score_fn) -> np.ndarray:
    if isinstance(detector, IForest):
        return iforest_diffi_reference(detector, x)
    return whitebox_local_rank_reference(x, X_ref, score_fn)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def rank_from_scores(ref_scores: np.ndarray, s: float) -> int:
    return int(1 + np.sum(ref_scores > s))


def topk(imp: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-np.abs(imp))[:k]


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = set(a.tolist()), set(b.tolist())
    return len(sa & sb) / max(1, len(sa | sb))


def infidelity(x, imp, score_fn, n_samples=40) -> float:
    sigma = 0.1 * (np.abs(x) + 1e-6)
    fx = float(score_fn(x.reshape(1, -1))[0])
    vals = []
    for _ in range(n_samples):
        eps = RNG.normal(0.0, sigma)
        fx2 = float(score_fn((x - eps).reshape(1, -1))[0])
        vals.append((float(np.dot(eps, imp)) - (fx - fx2)) ** 2)
    return float(np.mean(vals))


def sensitivity_proxy(imp, scale) -> float:
    s = 0.03 * (scale + 1e-12)
    return float(np.linalg.norm(imp * s) / (np.linalg.norm(imp) + 1e-12))


def contextual_counterfactual_validity(x, X_ref, feat_idx, score_fn, n_neighbors=40, random_trials=20) -> float:
    d = np.linalg.norm(X_ref - x.reshape(1, -1), axis=1)
    nbr_idx = np.argsort(d)[: min(n_neighbors, len(d))]
    local_mean = np.mean(X_ref[nbr_idx], axis=0)
    s0 = float(score_fn(x.reshape(1, -1))[0])

    z_sel = x.copy()
    z_sel[feat_idx] = local_mean[feat_idx]
    sel_shift = abs(float(score_fn(z_sel.reshape(1, -1))[0]) - s0)

    k = len(feat_idx)
    all_idx = np.arange(X_ref.shape[1])
    rand_shifts = []
    for _ in range(random_trials):
        ridx = RNG.choice(all_idx, size=k, replace=False)
        zr = x.copy()
        zr[ridx] = local_mean[ridx]
        rand_shifts.append(abs(float(score_fn(zr.reshape(1, -1))[0]) - s0))
    return float(sel_shift / (np.mean(rand_shifts) + 1e-12))


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run(args):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    auc_rows = []

    for ds in args.datasets:
        try:
            data = load_dataset(ds)
        except Exception as e:
            print(f"[WARN] Skipping {ds}: {e}")
            continue
        X = preprocess(data.X)
        y = data.y.astype(int)
        X, y = cap_size(X, y, args.max_samples)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.35, random_state=42, stratify=y)

        detectors = {
            "LODA": LODA(contamination=0.1),
            "IForest": IForest(contamination=0.1, random_state=42),
            "LOF": LOF(contamination=0.1),
            "DeepSVDD": DeepSVDD(n_features=Xtr.shape[1], contamination=0.1, random_state=42, epochs=20, verbose=0),
        }

        for dname, det in detectors.items():
            det.fit(Xtr)
            score_fn = detector_score_fn(det)
            s_te = score_fn(Xte)
            try:
                auc = float(roc_auc_score(yte, s_te))
            except Exception:
                auc = np.nan
            auc_rows.append({"dataset": ds, "detector": dname, "auc": auc})

            top_n = max(1, int(np.ceil(0.05 * len(Xte))))
            top_idx = np.argsort(-s_te)[:top_n]
            ref_scores = score_fn(Xtr)
            scale = np.std(Xtr, axis=0)
            med = np.median(Xtr, axis=0)

            lox = LOXExplainer(n_perturb=args.lox_L, random_state=42).fit(Xtr, det)
            loxr = LOXRExplainer(knn_k=args.loxr_knn, alpha=args.loxr_alpha).fit(Xtr, det)

            for ridx in top_idx:
                x = Xte[ridx]
                gt_wb = whitebox_detector_reference(det, x, Xtr, score_fn)

                methods = {
                    "LOX": lox.explain(x)["importance"],
                    "LOX-R": loxr.explain(x)["importance"],
                    "LOFO-Rank": local_lofo_rank(x, Xtr, score_fn),
                    "PFI-Rank": local_pfi_rank(x, Xtr, score_fn, n_perturb=args.pfi_perturb),
                    "LIME-ScoreReg": lime_regressor_score(x, Xtr, score_fn, n_samples=args.lime_samples),
                    "SHAP-Permutation": shap_permutation_score(x, Xtr, score_fn),
                }

                for k in [3, 5]:
                    gt_k = topk(gt_wb, k)
                    gt_r = np.argsort(-np.abs(gt_wb))[:k]
                    for m, imp in methods.items():
                        tk = topk(imp, k)
                        tau = kendalltau(gt_r, np.argsort(-np.abs(imp))[:k]).correlation
                        tau = 0.0 if np.isnan(tau) else float(tau)
                        z = x.copy()
                        z[tk] = med[tk]
                        r0 = rank_from_scores(ref_scores, float(score_fn(x.reshape(1, -1))[0]))
                        r1 = rank_from_scores(ref_scores, float(score_fn(z.reshape(1, -1))[0]))
                        all_rows.append({
                            "dataset": ds,
                            "detector": dname,
                            "outlier_index": int(ridx),
                            "method": m,
                            "k": k,
                            "fidelity_jaccard": jaccard(gt_k, tk),
                            "kendall_tau": tau,
                            "infidelity": infidelity(x, imp, score_fn, n_samples=20),
                            "sensitivity": sensitivity_proxy(imp, scale),
                            "delta_rank": abs(r1 - r0),
                            "context_counterfactual_validity": contextual_counterfactual_validity(x, Xtr, tk, score_fn),
                        })

    df = pd.DataFrame(all_rows)
    auc_df = pd.DataFrame(auc_rows)
    df.to_csv(out_dir / "all_results.csv", index=False)
    auc_df.to_csv(out_dir / "detector_auc.csv", index=False)

    summary = (
        df.groupby(["dataset", "detector", "k", "method"], as_index=False)[
            ["fidelity_jaccard", "kendall_tau", "infidelity", "sensitivity", "delta_rank", "context_counterfactual_validity"]
        ].mean()
        .sort_values(["dataset", "detector", "k", "fidelity_jaccard"], ascending=[True, True, True, False])
    )
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"[INFO] Done. Results at: {out_dir}")
    print(f"[INFO] {len(df)} evaluation rows across {len(df['dataset'].unique())} datasets.")


def parse_args():
    p = argparse.ArgumentParser(description="LOX / LOX-R experiment runner.")
    p.add_argument(
        "--datasets", nargs="+",
        default=["cancer", "adult", "credit", "heart", "bank_marketing", "covertype",
                 "mammography", "shuttle", "satimage", "synthetic_context",
                 "synthetic_guaranteed", "fred_md"],
    )
    p.add_argument("--out-dir", default="results")
    p.add_argument("--max-samples", type=int, default=2500)
    p.add_argument("--lox-L", type=int, default=32, help="LOX perturbation count.")
    p.add_argument("--loxr-knn", type=int, default=50, help="LOX-R neighborhood size.")
    p.add_argument("--loxr-alpha", type=float, default=0.9, help="LOX-R rank/residual trade-off.")
    p.add_argument("--lime-samples", type=int, default=100)
    p.add_argument("--pfi-perturb", type=int, default=20)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
