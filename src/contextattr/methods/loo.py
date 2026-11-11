"""Leave-one-out (ContextCite's baseline, SPEX §2): each unit's score is how much the value falls when that unit
alone is removed from the full prompt. n + 1 masks; the budget does not apply."""

from __future__ import annotations

import numpy as np

from .base import Method, Surrogate


class LeaveOneOut(Method):
    name = "loo"

    def masks(self, n: int, budget: int | None = None, rng=None) -> np.ndarray:
        return np.vstack([np.ones((1, n), dtype=bool), ~np.eye(n, dtype=bool)])

    def fit(self, masks: np.ndarray, values: np.ndarray) -> Surrogate:
        masks, values = np.asarray(masks, dtype=bool), np.asarray(values, dtype=float)
        n = masks.shape[1]
        full = np.flatnonzero(masks.all(axis=1))
        if not len(full):
            raise ValueError("leave-one-out needs the full prompt among its masks")
        f_full = values[full[0]]
        scores = np.empty(n)
        for i in range(n):
            row = np.flatnonzero((masks.sum(axis=1) == n - 1) & ~masks[:, i])
            if not len(row):
                raise ValueError(f"leave-one-out needs the mask without unit {i}")
            scores[i] = f_full - values[row[0]]
        # the additive model anchored at the full prompt: f̂(1) = f(1), f̂(1 − e_i) = f(1 − e_i)
        return Surrogate(intercept=f_full - scores.sum(), scores=scores)
