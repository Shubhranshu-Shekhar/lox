from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import pairwise_distances

from .base import BaseLOXExplainer
from .utils import rank_of_score


class LOXExplainer(BaseLOXExplainer):
    """
    LOX: Local Outlier eXplanation via feature perturbation (Algorithm 1).

    For each feature j, measures the expected rank shift when j is replaced
    by random draws from the reference distribution.
    """

    def __init__(self, n_perturb: int = 128, random_state: Optional[int] = 42) -> None:
        super().__init__()
        self.n_perturb = int(n_perturb)
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    def _fit_internal(self) -> None:
        return None

    def _explain_internal(self, x: np.ndarray) -> np.ndarray:
        assert self._X_ref is not None and self._score_fn is not None and self._ref_scores is not None
        d = x.shape[0]
        base_rank = self._base_rank(x)
        importance = np.zeros(d, dtype=float)
        for j in range(d):
            deltas = []
            col = self._X_ref[:, j]
            for _ in range(self.n_perturb):
                z = x.copy()
                z[j] = col[self._rng.integers(0, len(col))]
                s = float(self._score_fn(z.reshape(1, -1))[0])
                r = rank_of_score(self._ref_scores, s)
                deltas.append(abs(r - base_rank))
            importance[j] = float(np.mean(deltas))
        return importance


class LOXRExplainer(BaseLOXExplainer):
    """
    LOX-R: Rank-aware LOX variant (Algorithm 2).

    Importance combines:
    1) normalized rank shift after local conditional replacement (kNN)
    2) normalized residual-to-manifold correction magnitude
    """

    def __init__(self, knn_k: int = 30, alpha: float = 0.7) -> None:
        super().__init__()
        self.knn_k = int(knn_k)
        self.alpha = float(alpha)

    def _fit_internal(self) -> None:
        return None

    def _explain_internal(self, x: np.ndarray) -> np.ndarray:
        assert self._X_ref is not None and self._score_fn is not None and self._ref_scores is not None
        d = x.shape[0]
        base_rank = self._base_rank(x)
        std = np.std(self._X_ref, axis=0) + 1e-12
        importance = np.zeros(d, dtype=float)
        for j in range(d):
            mask = np.ones(d, dtype=bool)
            mask[j] = False
            dist = pairwise_distances(self._X_ref[:, mask], x[mask].reshape(1, -1), metric="euclidean").ravel()
            nn_idx = np.argsort(dist)[: min(self.knn_k, len(dist))]
            x_cond = float(np.mean(self._X_ref[nn_idx, j]))
            z = x.copy()
            z[j] = x_cond
            s = float(self._score_fn(z.reshape(1, -1))[0])
            r = rank_of_score(self._ref_scores, s)
            rank_term = abs(r - base_rank) / max(1.0, float(base_rank))
            residual_term = abs(x[j] - x_cond) / std[j]
            importance[j] = self.alpha * rank_term + (1.0 - self.alpha) * residual_term
        return importance
