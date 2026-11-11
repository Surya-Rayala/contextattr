"""KernelSHAP (Lundberg & Lee 2017; SPEX B.3): Shapley values as the weighted least-squares fit of the Shapley kernel,
with the empty and the full prompt fitted exactly (efficiency: the scores add up to f(1) − f(0)).

The masks are drawn as the shap library's KernelExplainer draws them. The kernel puts most weight on the smallest and
the largest coalitions, so whole size groups are enumerated first, sizes 1 and n − 1, then 2 and n − 2, …, while the
budget holds them; each enumerated mask gets its kernel weight π(s) = (n − 1) / (C(n, s)·s·(n − s)). The rest of the
budget is drawn from the sizes left, size s with probability ∝ C(n, s)·π(s), each mask beside its complement (paired
sampling), and these masks share the kernel weight of the sizes left equally. A budget of 2^n enumerates every mask
and gives the exact Shapley values; one of 2n + 2 already completes sizes 1 and n − 1, so the fit is always determined.

The kernel's denominator C(n, s)·s·(n − s) outgrows numpy's 64-bit integers from n = 57, and C(n, s) a float from
n ≈ 1,030, so the weights come from Python's exact integers: a size group's mass C(n, s)·π(s) = (n − 1) / (s·(n − s))
needs no C(n, s) at all, and the weights of masks from elsewhere are taken in logarithms, relative to the largest
(the fit does not change with their scale).
"""

from __future__ import annotations

import itertools
from math import comb, exp, log

import numpy as np

from .base import Method, Surrogate, check_budget, least_squares


class KernelSHAP(Method):
    name = "kernelshap"

    def __init__(self, paired: bool = True):
        self.paired = bool(paired)
        self._design: tuple[np.ndarray, np.ndarray] | None = None

    @staticmethod
    def least_budget(n: int) -> int:
        return 2 if n == 1 else min(2 ** n, 2 * n + 2)

    def masks(self, n: int, budget: int, rng: np.random.Generator) -> np.ndarray:
        check_budget(self.name, budget, self.least_budget(n))
        rows, weights = [np.zeros(n, dtype=bool), np.ones(n, dtype=bool)], [0.0, 0.0]
        left = budget - 2
        done: set[int] = set()
        for s in range(1, n // 2 + 1):                     # the kernel's mass of a size group falls as s nears n/2
            group = [s] if 2 * s == n else [s, n - s]
            count = sum(comb(n, k) for k in group)
            if count > left:
                break
            for k in group:
                for subset in itertools.combinations(range(n), k):
                    row = np.zeros(n, dtype=bool)
                    row[list(subset)] = True
                    rows.append(row)
                    weights.append(_kernel(n, k))
            left -= count
            done.update(group)
        sizes = [k for k in range(1, n) if k not in done]
        draws = left - left % 2 if self.paired else left      # every drawn mask beside its complement: an even count
        if sizes and draws > 0:
            mass = np.array([(n - 1) / (k * (n - k)) for k in sizes])      # C(n, k)·π(k), Python's integers
            drawn = []
            while len(drawn) < draws:
                row = np.zeros(n, dtype=bool)
                row[rng.permutation(n)[: rng.choice(sizes, p=mass / mass.sum())]] = True
                drawn.append(row)
                if self.paired:
                    drawn.append(~row)
            rows += drawn
            weights += [mass.sum() / len(drawn)] * len(drawn)
        masks = np.array(rows)
        self._design = (masks.copy(), np.array(weights))
        return masks

    def fit(self, masks: np.ndarray, values: np.ndarray) -> Surrogate:
        masks, values = np.asarray(masks, dtype=bool), np.asarray(values, dtype=float)
        n = masks.shape[1]
        size = masks.sum(axis=1)
        empty, full = np.flatnonzero(size == 0), np.flatnonzero(size == n)
        if not len(empty) or not len(full):
            raise ValueError("KernelSHAP needs the empty and the full prompt among its masks")
        f0, f1 = values[empty[0]], values[full[0]]
        if n == 1:
            return Surrogate(intercept=f0, scores=np.array([f1 - f0]))
        if self._design is not None and np.array_equal(self._design[0], masks):
            weights = self._design[1]
        else:            # masks from elsewhere: each with its kernel weight (right for every mask, or for uniform draws)
            weights = kernel_weights(n, size)
        inner = (size > 0) & (size < n) & (weights > 0)
        x, y, w = masks[inner].astype(float), values[inner] - f0, weights[inner]
        # efficiency, Σφ = f(1) − f(0), eliminates the last score: y − x_n Δ = Σ_{i<n} φ_i (x_i − x_n)
        delta = f1 - f0
        design = x[:, :-1] - x[:, -1:]
        phi = least_squares(design, y - x[:, -1] * delta, w, refuse=(
            f"KernelSHAP: these masks leave the {n} scores undetermined; give a budget of at least "
            f"{self.least_budget(n)}"))
        return Surrogate(intercept=f0, scores=np.append(phi, delta - phi.sum()))

    def __repr__(self) -> str:
        return f"KernelSHAP(paired={self.paired})"


def _kernel(n: int, s: int) -> float:
    """The Shapley kernel's weight π(s) of one coalition of size s, 0 < s < n, for the sizes enumerated whole (C(n, s)
    at most the budget)."""
    n, s = int(n), int(s)
    return (n - 1) / (comb(n, s) * s * (n - s))


def kernel_weights(n: int, sizes) -> np.ndarray:
    """π(s) of masks of these sizes relative to the largest among them, from log π(s) (exact integers inside the
    logarithm); 0 for the empty and the full mask."""
    n = int(n)
    logs = {k: log(n - 1) - log(comb(n, k)) - log(k) - log(n - k) for k in {int(s) for s in sizes} if 0 < k < n}
    top = max(logs.values(), default=0.0)
    return np.array([exp(logs[int(s)] - top) if 0 < s < n else 0.0 for s in sizes])
