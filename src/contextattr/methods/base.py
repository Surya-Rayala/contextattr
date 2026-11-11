"""What every method is (docs/methods.md): a design of masks to evaluate, and a fit of the surrogate whose terms
are the scores. The surrogate predicts the value of any mask, which is how its faithfulness is measured (eq. 7)."""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np


@dataclass(frozen=True)
class Surrogate:
    """f̂(m) = intercept + Σ_i scores[i]·m_i + Σ_S interactions[S]·Π_{i∈S} m_i, for masks of 0/1. A unit's score is
    its effect alone; an interaction S, a tuple of two or more unit positions, is what the units do together."""

    intercept: float
    scores: np.ndarray
    interactions: Mapping[tuple[int, ...], float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        scores = np.asarray(self.scores, dtype=float).copy()
        scores.flags.writeable = False
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "intercept", float(self.intercept))
        object.__setattr__(self, "interactions", MappingProxyType(
            {tuple(int(i) for i in s): float(w) for s, w in dict(self.interactions).items()}))

    @property
    def n(self) -> int:
        return len(self.scores)

    def predict(self, masks) -> np.ndarray:
        m = np.asarray(masks, dtype=float)
        out = self.intercept + m @ self.scores
        for s, w in self.interactions.items():
            out = out + w * m[:, list(s)].prod(axis=1)
        return out


class Method:
    """A method: `masks(n, budget, rng)` designs the masks to evaluate, `fit(masks, values)` gives the surrogate, and
    `run(game, n, budget, rng)` does both against a game (masks → values). A method that chooses its masks as it
    goes (the shapiq ones) overrides `run` instead."""

    name = "method"

    def masks(self, n: int, budget: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def fit(self, masks: np.ndarray, values: np.ndarray) -> Surrogate:
        raise NotImplementedError

    def run(self, game, n: int, budget: int, rng: np.random.Generator) -> tuple[Surrogate, int]:
        """The surrogate, and how many masks it took."""
        masks = self.masks(n, budget, rng)
        return self.fit(masks, game(masks)), len(masks)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


def all_masks(n: int) -> np.ndarray:
    """Every mask of n units (2^n rows), the empty one first and the full one last."""
    return np.array(list(itertools.product([False, True], repeat=n)), dtype=bool).reshape(2 ** n, n)


def check_budget(name: str, budget: int, least: int) -> None:
    if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)) or budget < least:
        raise ValueError(f"{name} needs a budget of at least {least} masks, not {budget!r}")


def least_squares(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, *, refuse: str | None = None
                  ) -> np.ndarray:
    """The (weighted) least-squares coefficients of y on the columns of x. With `refuse`, a ValueError of that message
    when the rows leave the coefficients undetermined (numerical rank below the number of columns), where lstsq would
    silently give one fit of many."""
    if w is not None:
        root = np.sqrt(w)
        x, y = x * root[:, None], y * root
    coef, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if refuse is not None and rank < x.shape[1]:
        raise ValueError(refuse)
    return coef
