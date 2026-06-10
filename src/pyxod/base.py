from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np

from .utils import rank_of_score, resolve_score_fn, to_2d, topk_indices


class BaseLOXExplainer(ABC):
    """Base class for LOX explainers."""

    def __init__(self) -> None:
        self._is_fitted = False
        self._X_ref: Optional[np.ndarray] = None
        self._score_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
        self._ref_scores: Optional[np.ndarray] = None

    def fit(
        self,
        X_ref: np.ndarray,
        detector: object,
        score_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> "BaseLOXExplainer":
        self._X_ref = np.asarray(X_ref, dtype=float)
        self._score_fn = resolve_score_fn(detector, score_fn=score_fn)
        self._ref_scores = self._score_fn(self._X_ref)
        self._fit_internal()
        self._is_fitted = True
        return self

    def _check_ready(self) -> None:
        if not self._is_fitted or self._X_ref is None or self._score_fn is None or self._ref_scores is None:
            raise RuntimeError("Explainer is not fitted. Call `fit(X_ref, detector)` first.")

    def explain(self, x: np.ndarray, top_k: Optional[int] = None) -> dict:
        self._check_ready()
        x_arr = np.asarray(x, dtype=float).ravel()
        importance = self._explain_internal(x_arr)
        if top_k is None:
            top_k = len(importance)
        top_idx = topk_indices(importance, top_k)
        return {
            "importance": importance,
            "feature_ranking": top_idx.tolist(),
        }

    def explain_batch(self, X: np.ndarray, top_k: Optional[int] = None) -> list[dict]:
        self._check_ready()
        rows = to_2d(np.asarray(X, dtype=float))
        return [self.explain(x, top_k=top_k) for x in rows]

    def _base_rank(self, x: np.ndarray) -> int:
        self._check_ready()
        assert self._score_fn is not None
        assert self._ref_scores is not None
        s = float(self._score_fn(x.reshape(1, -1))[0])
        return rank_of_score(self._ref_scores, s)

    @abstractmethod
    def _fit_internal(self) -> None:
        ...

    @abstractmethod
    def _explain_internal(self, x: np.ndarray) -> np.ndarray:
        ...
