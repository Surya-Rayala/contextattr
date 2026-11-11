"""Linear fits on random masks (docs/methods.md): Banzhaf, LIME and ContextCite. Each fits f(m) ≈ b + Σ_i w_i·m_i;
they differ in which masks they draw and how they fit. scikit-learn is imported only when ContextCite or LIME fits
(its import takes long on a busy machine, and `import contextattr` should not wait for it)."""

from __future__ import annotations

import numpy as np

from .base import Method, Surrogate, all_masks, check_budget, least_squares

EXACT_UP_TO = 16    # Banzhaf enumerates every mask when the budget covers all 2^n of them and n is at most this


class Banzhaf(Method):
    """Banzhaf values: least squares on masks drawn uniformly (each unit kept with probability ½), which is
    Faith-Banzhaf at order 1 (SPEX B.3, eq. 32). With a budget of at least 2^n masks every mask is used once and the
    values are exact. Masks that leave the fit undetermined (too few, or units always kept or removed together) are
    refused rather than fitted."""

    name = "banzhaf"

    def masks(self, n: int, budget: int, rng: np.random.Generator) -> np.ndarray:
        check_budget(self.name, budget, n + 1)
        if n <= EXACT_UP_TO and budget >= 2 ** n:
            return all_masks(n)
        return rng.random((budget, n)) < 0.5

    def fit(self, masks: np.ndarray, values: np.ndarray) -> Surrogate:
        x = np.hstack([np.ones((len(masks), 1)), np.asarray(masks, dtype=float)])
        coef = least_squares(x, np.asarray(values, dtype=float), refuse=(
            f"Banzhaf: these {len(x)} masks leave the {x.shape[1] - 1} scores undetermined (too few masks, or units "
            f"always kept or removed together); give a larger budget"))
        return Surrogate(intercept=coef[0], scores=coef[1:])


class ContextCite(Method):
    """ContextCite (Cohen-Wang 2024 §3): a Lasso on masks that keep each unit with probability ½, so most scores are
    exactly 0. alpha = 0.01 as in ContextCite. With standardize (the default) the Lasso fits the values divided by
    their spread, so alpha means the same whatever the target's units; standardize=False fits the raw values, as
    ContextCite does with its log-odds target."""

    name = "contextcite"

    def __init__(self, alpha: float = 0.01, standardize: bool = True):
        if not alpha > 0:
            raise ValueError("ContextCite: alpha must be positive")
        self.alpha, self.standardize = float(alpha), bool(standardize)

    def masks(self, n: int, budget: int, rng: np.random.Generator) -> np.ndarray:
        check_budget(self.name, budget, 2)
        return rng.random((budget, n)) < 0.5

    def fit(self, masks: np.ndarray, values: np.ndarray) -> Surrogate:
        x, y = np.asarray(masks, dtype=float), np.asarray(values, dtype=float)
        mean, spread = y.mean(), y.std()
        if spread == 0:
            return Surrogate(intercept=mean, scores=np.zeros(x.shape[1]))
        from sklearn.linear_model import Lasso  # noqa: PLC0415

        scale = spread if self.standardize else 1.0
        model = Lasso(alpha=self.alpha, fit_intercept=True, max_iter=100_000, tol=1e-10)
        model.fit(x, (y - mean) / scale)
        return Surrogate(intercept=mean + scale * model.intercept_, scores=scale * model.coef_)

    def __repr__(self) -> str:
        return f"ContextCite(alpha={self.alpha}, standardize={self.standardize})"


class LIME(Method):
    """LIME (Ribeiro 2016) as its text explainer does it: the full prompt, then masks that each remove a number of
    units drawn uniformly from 1 to n; each mask weighted by sqrt(exp(−d²/width²)) of its cosine distance d (×100) to
    the full prompt, width 25; a weighted ridge fit (alpha 1) on every unit (no feature selection)."""

    name = "lime"

    def __init__(self, kernel_width: float = 25.0, alpha: float = 1.0):
        if not kernel_width > 0 or not alpha >= 0:
            raise ValueError("LIME: kernel_width must be positive and alpha at least 0")
        self.kernel_width, self.alpha = float(kernel_width), float(alpha)

    def masks(self, n: int, budget: int, rng: np.random.Generator) -> np.ndarray:
        check_budget(self.name, budget, 2)
        out = np.ones((budget, n), dtype=bool)
        for row in out[1:]:
            row[rng.choice(n, size=rng.integers(1, n + 1), replace=False)] = False
        return out

    def weights(self, masks: np.ndarray) -> np.ndarray:
        kept = np.asarray(masks, dtype=float).sum(axis=1)
        n = masks.shape[1]
        cosine = np.sqrt(kept / n)                       # the cosine of a 0/1 mask with the all-ones vector
        distance = (1 - cosine) * 100
        return np.sqrt(np.exp(-(distance ** 2) / self.kernel_width ** 2))

    def fit(self, masks: np.ndarray, values: np.ndarray) -> Surrogate:
        from sklearn.linear_model import Ridge  # noqa: PLC0415

        x, y = np.asarray(masks, dtype=float), np.asarray(values, dtype=float)
        model = Ridge(alpha=self.alpha, fit_intercept=True)
        model.fit(x, y, sample_weight=self.weights(masks))
        return Surrogate(intercept=model.intercept_, scores=model.coef_)

    def __repr__(self) -> str:
        return f"LIME(kernel_width={self.kernel_width}, alpha={self.alpha})"
