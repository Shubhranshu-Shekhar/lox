from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import kendalltau

from .utils import topk_indices


def jaccard_at_k(importance_a: np.ndarray, importance_b: np.ndarray, k: int) -> float:
    a = set(topk_indices(np.asarray(importance_a), k).tolist())
    b = set(topk_indices(np.asarray(importance_b), k).tolist())
    if not a and not b:
        return 1.0
    return float(len(a & b) / max(1, len(a | b)))


def kendall_at_k(importance_a: np.ndarray, importance_b: np.ndarray, k: int) -> float:
    ia = np.argsort(-np.abs(np.asarray(importance_a)))[:k]
    ib = np.argsort(-np.abs(np.asarray(importance_b)))[:k]
    tau = kendalltau(ia, ib).correlation
    return float(0.0 if np.isnan(tau) else tau)


def delta_rank(
    x: np.ndarray,
    top_features: Sequence[int],
    score_fn,
    ref_scores: np.ndarray,
    baseline_values: np.ndarray,
) -> float:
    x = np.asarray(x, dtype=float)
    z = x.copy()
    z[list(top_features)] = baseline_values[list(top_features)]
    s1 = float(score_fn(x.reshape(1, -1))[0])
    s2 = float(score_fn(z.reshape(1, -1))[0])
    r1 = int(1 + np.sum(ref_scores > s1))
    r2 = int(1 + np.sum(ref_scores > s2))
    return float(abs(r2 - r1))


def infidelity(
    x: np.ndarray,
    importance: np.ndarray,
    score_fn,
    n_samples: int = 100,
    sigma_scale: float = 0.1,
    random_state: int = 42,
) -> float:
    rng = np.random.default_rng(random_state)
    x = np.asarray(x, dtype=float)
    imp = np.asarray(importance, dtype=float)
    sigma = sigma_scale * (np.abs(x) + 1e-6)
    fx = float(score_fn(x.reshape(1, -1))[0])
    vals = []
    for _ in range(n_samples):
        eps = rng.normal(0.0, sigma)
        fx2 = float(score_fn((x - eps).reshape(1, -1))[0])
        vals.append((float(np.dot(eps, imp)) - (fx - fx2)) ** 2)
    return float(np.mean(vals))


def sensitivity_proxy(importance: np.ndarray, feature_scale: np.ndarray, sigma_scale: float = 0.03) -> float:
    imp = np.asarray(importance, dtype=float)
    scale = np.asarray(feature_scale, dtype=float)
    sigma = sigma_scale * (scale + 1e-12)
    return float(np.linalg.norm(imp * sigma, ord=2) / (np.linalg.norm(imp, ord=2) + 1e-12))
