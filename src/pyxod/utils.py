from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np


def to_2d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def topk_indices(importance: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-np.abs(importance))[:k]


def rank_of_score(reference_scores: np.ndarray, score: float) -> int:
    # Rank 1 means most anomalous under "higher score => more anomalous".
    return int(1 + np.sum(reference_scores > score))


def resolve_score_fn(
    detector: object,
    score_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    if score_fn is not None:
        return lambda X: np.asarray(score_fn(X), dtype=float).ravel()

    if hasattr(detector, "decision_function"):
        return lambda X: np.asarray(detector.decision_function(X), dtype=float).ravel()

    if hasattr(detector, "score_samples"):
        # scikit-learn score_samples often returns normality; flip sign.
        return lambda X: -np.asarray(detector.score_samples(X), dtype=float).ravel()

    raise ValueError(
        "Cannot infer scoring function. Provide `score_fn=` or a detector with "
        "`decision_function` / `score_samples`."
    )


def as_float_array(x: Sequence[float]) -> np.ndarray:
    return np.asarray(x, dtype=float)
