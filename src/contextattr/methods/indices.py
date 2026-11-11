"""One surrogate read as the classic indices (SPEX App. C, Table 1). A surrogate's terms are Möbius coefficients a(S):
f̂(m) = Σ_S a(S)·Π_{i∈S} m_i, the intercept being a(∅). From them, exactly:

    Shapley value      φ_i = Σ_{S∋i} a(S) / |S|
    Banzhaf value      β_i = Σ_{S∋i} a(S) / 2^(|S|−1)
    Shapley-Taylor     of order k: a(S) for |S| < k, and Σ_{T⊇S} a(T) / C(|T|, k) for |S| = k  (App. C, eq. 58)
"""

from __future__ import annotations

from math import comb

import numpy as np

from .base import Surrogate


def terms(surrogate: Surrogate) -> dict[tuple[int, ...], float]:
    """Every Möbius term of the surrogate: () → intercept, (i,) → score, S → interaction."""
    out = {(): surrogate.intercept}
    out.update({(i,): float(w) for i, w in enumerate(surrogate.scores)})
    out.update(surrogate.interactions)
    return out


def shapley(surrogate: Surrogate) -> np.ndarray:
    phi = np.zeros(surrogate.n)
    for s, a in terms(surrogate).items():
        for i in s:
            phi[i] += a / len(s)
    return phi


def banzhaf(surrogate: Surrogate) -> np.ndarray:
    beta = np.zeros(surrogate.n)
    for s, a in terms(surrogate).items():
        for i in s:
            beta[i] += a / 2 ** (len(s) - 1)
    return beta


def shapley_taylor(surrogate: Surrogate, order: int = 2) -> dict[tuple[int, ...], float]:
    """The Shapley-Taylor interaction index of the given order, every set of at most `order` units that has a value."""
    if order < 1:
        raise ValueError("order must be at least 1")
    out: dict[tuple[int, ...], float] = {}
    for t, a in terms(surrogate).items():
        if len(t) < order:
            out[t] = out.get(t, 0.0) + a
        else:
            share = a / comb(len(t), order)
            for s in _subsets(t, order):
                out[s] = out.get(s, 0.0) + share
    return out


def _subsets(t: tuple[int, ...], k: int):
    from itertools import combinations
    return combinations(t, k)
