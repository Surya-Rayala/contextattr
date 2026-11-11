"""The interaction methods (docs/methods.md), through shapiq, the extra [interactions]: Faith-Banzhaf and Faith-Shap
(every interaction up to an order; small n), SPEX (sparse interactions up to a degree, 5 unless set; large n) and
ProxySPEX (sparse, about a tenth of SPEX's masks). shapiq chooses its own masks and asks the value of each through
the game it is given.

Each gives its answer as Faith-Banzhaf (FBII) coefficients: at an order at least the surrogate's degree they are its
Möbius coefficients, the product basis Surrogate uses, so predictions, R², removal and the indices of App. C apply as
for the per-unit methods (Faith-Shap keeps its own FSII coefficients, a product-basis polynomial too). shapiq is
imported only when one of these runs: its import takes 25–45 s and galois's first codes about 15 s more on a shared
two-CPU machine (measured 2026-11-06), which `import contextattr` should not wait for."""

from __future__ import annotations

import itertools
import random
from contextlib import contextmanager

import numpy as np

from .base import Method, Surrogate, check_budget


@contextmanager
def _global_seed(seed: int):
    """Seed numpy's and Python's global generators for one approximation, and give the caller's states back after.
    SPEX's codes and delays are drawn by sparse_transform from numpy's global generator, which shapiq's random_state
    does not reach: unseeded, the same seed gave other masks each run (found 2026-11-07, when a re-run found none of
    them in the cache). The game is called inside, so a wrapper drawing from these generators draws seeded values."""
    np_state, py_state = np.random.get_state(), random.getstate()
    np.random.seed(seed)
    random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)


class ShapiqMethod(Method):
    """A shapiq approximator run against the game; `make(shapiq, n, seed)` builds it, `finish` reads its answer."""

    index = "FBII"

    def __init__(self, max_order: int = 2):
        if isinstance(max_order, bool) or not isinstance(max_order, int) or max_order < 1:
            raise ValueError(f"{type(self).__name__}: max_order must be a positive integer")
        self.max_order = max_order

    def make(self, shapiq, n: int, seed: int):
        raise NotImplementedError

    def check(self, approximator, n: int, budget: int) -> None:
        """Called before the first value is asked: refuse what shapiq changed on its own."""

    def run(self, game, n: int, budget: int, rng: np.random.Generator) -> tuple[Surrogate, int]:
        check_budget(self.name, budget, 2)
        import shapiq   # noqa: PLC0415 - imported only here, see the module's note

        asked = []
        seed = int(rng.integers(2 ** 31))
        approximator = self.make(shapiq, n, seed)

        def g(masks):
            if not asked:
                self.check(approximator, n, budget)
            masks = np.array(masks, dtype=bool)
            values = np.asarray(game(masks), dtype=float)
            asked.append((masks, values))
            return values

        with _global_seed(seed):
            values = approximator.approximate(budget, g)
        return self.finish(values, n, asked), int(sum(len(m) for m, _ in asked))

    def finish(self, values, n: int, asked: list) -> Surrogate:
        """The surrogate from shapiq's answer; `asked` holds every (masks, values) batch shapiq asked for."""
        return from_interaction_values(values, n)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(max_order={self.max_order})"


class FaithBanzhaf(ShapiqMethod):
    """Faith-Banzhaf (Tsai 2023): the best polynomial of order max_order under uniform weights over masks. shapiq's
    RegressionFBII draws the masks. With `ridge` (the default since 2026-11-09), they are fitted as SPEX does (B.3:
    cross-validated ridge regression; B.1: scikit-learn's RidgeCV), by `ridge_fit`; but when every one of the 2ⁿ masks
    was asked, by shapiq's least squares, which then gives the index exactly (ridge could only shrink it). Least squares
    on sampled masks (`ridge=False`) overfits as the terms near the masks. With noise of 0.3 of the function's spread, on
    the same masks (measured 2026-11-08): R² −0.08 against ridge's 0.79 at n = 31, order 3 (4,992 terms on 5,376 masks),
    where ridge took 9 min on one CPU against 5 s; within 0.005 of ridge at 2,017 terms or fewer on 5,376–6,912 masks."""

    name = "faithbanzhaf"

    def __init__(self, max_order: int = 2, ridge: bool = True):
        super().__init__(max_order=max_order)
        self.ridge = bool(ridge)

    def make(self, shapiq, n, seed):
        return shapiq.RegressionFBII(n, max_order=self.max_order, random_state=seed)

    def finish(self, values, n, asked):
        masks = np.concatenate([m for m, _ in asked])
        if not self.ridge or len(np.unique(masks, axis=0)) == 2 ** n:
            return super().finish(values, n, asked)
        return ridge_fit(masks, np.concatenate([v for _, v in asked]), self.max_order)

    def __repr__(self) -> str:   # always names the fit: before 2026-11-09, FaithBanzhaf(max_order=2) meant least squares
        return f"FaithBanzhaf(max_order={self.max_order}, ridge={self.ridge})"


class FaithShap(ShapiqMethod):
    """Faith-Shap (Tsai 2023): the best polynomial of order max_order under the Shapley kernel's weights."""

    name = "faithshap"
    index = "FSII"

    def make(self, shapiq, n, seed):
        return shapiq.RegressionFSII(n, max_order=self.max_order, random_state=seed)


class SPEX(ShapiqMethod):
    """SPEX (Kang 2025, §5): sparse Fourier recovery with masks from channel codes; interactions up to `degree`
    (SPEX's t), reported in full as FBII of that order, so the surrogate is SPEX's own. `chase_depth` is the soft
    decoder's Chase depth (App. A, Algorithm 3, which does not give its value): by default (since 2026-11-09) 2t, as
    the SPEX authors' spectralexplain, whose predictions this SPEX then gives on the same masks (measured 2026-11-08 in
    8 of 8 noisy cases; with t, shapiq's own, R² was 0.003–0.070 lower there, and each fit 1–11 s faster).
    `chase_depth=degree` is shapiq's SPEX exactly (sparse_transform takes no depth as t)."""

    name = "spex"

    def __init__(self, degree: int = 5, chase_depth: int | None = None):
        super().__init__(max_order=degree)
        if chase_depth is not None and (isinstance(chase_depth, bool) or not isinstance(chase_depth, int)
                                        or chase_depth < 1):
            raise ValueError("SPEX: chase_depth must be a positive integer, or None for 2·degree")
        self.chase_depth = 2 * degree if chase_depth is None else chase_depth

    def make(self, shapiq, n, seed):
        approximator = shapiq.SPEX(n=n, max_order=self.max_order, index="FBII", degree_parameter=self.max_order,
                                   random_state=seed)
        from sparse_transform.qsft.utils.query import get_bch_decoder  # noqa: PLC0415 - a shapiq dependency
        approximator.decoder_args["source_decoder"] = get_bch_decoder(n, self.max_order, "soft",
                                                                      chase_depth=self.chase_depth)
        return approximator

    def check(self, approximator, n, budget):
        """shapiq lowers the code's degree t when the budget is too small for it (at n = 20, measured 2026-11-06: to 2
        at 300 masks, to 4 at 600, none from 1,200), which would explain at a degree other than the one named."""
        t = getattr(approximator, "degree_parameter", self.max_order)
        if t < self.max_order:
            raise ValueError(f"SPEX: a budget of {budget} masks is too small for degree {self.max_order} at n = {n} "
                             f"(shapiq would lower it to {t}); give a larger budget (SPEX Table 2) or "
                             f"SPEX(degree={t})")

    def __repr__(self) -> str:   # always names the depth: before 2026-11-09, SPEX(degree=5) meant depth t
        return f"SPEX(degree={self.max_order}, chase_depth={self.chase_depth})"


class ProxySPEX(ShapiqMethod):
    """ProxySPEX (Butler 2025): gradient-boosted trees fitted to uniform masks, their Fourier transform read off the
    trees. With `hpo` (shapiq's default) LightGBM is chosen by 5-fold cross-validation over depth 3 or 5, 500 or 1,000
    trees and learning rate 0.01 or 0.1, shapiq's grid for Butler 2025's proxy (scikit-learn's GridSearchCV); without
    it, LightGBM's own defaults (100 trees of up to 31 leaves, any depth). LightGBM takes as many threads as the
    process may use CPUs, whatever OMP_NUM_THREADS says (LOKY_MAX_CPU_COUNT limits it)."""

    name = "proxyspex"

    def __init__(self, max_order: int = 5, hpo: bool = True):
        super().__init__(max_order=max_order)
        self.hpo = bool(hpo)

    def make(self, shapiq, n, seed):
        return shapiq.ProxySPEX(n=n, max_order=self.max_order, index="FBII", hpo=self.hpo, random_state=seed)

    def __repr__(self) -> str:
        return f"ProxySPEX(max_order={self.max_order}, hpo={self.hpo})"


def ridge_fit(masks, values, order: int) -> Surrogate:
    """Every term up to `order` fitted by scikit-learn's RidgeCV, the penalty chosen by leave-one-out among 100 from
    1e-6 to 1e6, the intercept unpenalised. The fit is in the ±1 basis z = 1 − 2m (orthonormal under uniform masks, so
    the penalty shrinks every term alike); its terms are then written in the product basis Surrogate uses:
    Π_{i∈S} (1 − 2mᵢ) = Σ_{T⊆S} (−2)^|T| Π_{i∈T} mᵢ."""
    from sklearn.linear_model import RidgeCV  # noqa: PLC0415 - as LIME and ContextCite import theirs

    masks = np.asarray(masks, dtype=bool)
    n = masks.shape[1]
    sets = [list(s) for k in range(1, order + 1) for s in itertools.combinations(range(n), k)]
    z = 1.0 - 2.0 * masks
    x = np.empty((len(masks), len(sets)))
    for j, s in enumerate(sets):
        x[:, j] = z[:, s].prod(axis=1)
    model = RidgeCV(alphas=np.logspace(-6, 6, 100)).fit(x, np.asarray(values, dtype=float))
    terms = {(): float(model.intercept_)}
    for s, c in zip(sets, model.coef_):
        for k in range(len(s) + 1):
            for t in itertools.combinations(s, k):
                terms[t] = terms.get(t, 0.0) + (-2.0) ** k * float(c)
    intercept = terms.pop(())
    scores = np.zeros(n)
    for t in [t for t in terms if len(t) == 1]:
        scores[t[0]] = terms.pop(t)
    return Surrogate(intercept=intercept, scores=scores, interactions={t: w for t, w in terms.items() if w != 0.0})


def from_interaction_values(values, n: int) -> Surrogate:
    """A Surrogate from shapiq's InteractionValues of a product-basis index (FBII, FSII): () is the intercept, single
    units the scores, larger sets the interactions."""
    terms = {tuple(int(i) for i in s): float(v) for s, v in values.dict_values.items()}
    intercept = terms.pop((), float(getattr(values, "baseline_value", 0.0)))
    scores = np.zeros(n)
    for s in [s for s in terms if len(s) == 1]:
        scores[s[0]] = terms.pop(s)
    return Surrogate(intercept=intercept, scores=scores,
                     interactions={s: w for s, w in terms.items() if s and w != 0.0})
