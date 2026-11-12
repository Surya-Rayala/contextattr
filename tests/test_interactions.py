"""The interaction parts that need no shapiq: the indices of App. C read from a surrogate, top-r removal (SPEX eq. 8),
Recovery@r (eq. 9), interaction surrogates, and the adapter that runs a shapiq approximator (here a fake one)."""

import itertools
import sys
import types

import numpy as np
import pytest
from prompt_helpers import rio
from test_methods import GAMES, banzhaf, game, random_game, shapley

import contextattr as ca
from contextattr import evaluate
from contextattr.methods import FaithBanzhaf, Surrogate, all_masks, indices
from contextattr.methods.interactions import from_interaction_values, ridge_fit
from contextattr.wrapper.toy import log_sigmoid

D3_SCORE = 2 * (log_sigmoid(1.0) - log_sigmoid(-1.0)) / 8


def surrogate_of(coefficients, n):
    """The surrogate whose Möbius terms are these coefficients."""
    c = dict(coefficients)
    scores = np.zeros(n)
    for s in [s for s in c if len(s) == 1]:
        scores[s[0]] = c.pop(s)
    return Surrogate(intercept=c.pop((), 0.0), scores=scores, interactions=c)


@pytest.mark.parametrize("coefficients", GAMES)
def test_indices_from_a_surrogate_equal_the_closed_forms(coefficients):
    n = 1 + max([max(s) for s in coefficients if s] + [0])
    s = surrogate_of(coefficients, n)
    assert indices.shapley(s) == pytest.approx(shapley(coefficients, n), abs=1e-12)
    assert indices.banzhaf(s) == pytest.approx(banzhaf(coefficients, n), abs=1e-12)
    m = all_masks(n)
    assert s.predict(m) == pytest.approx(game(coefficients)(m), abs=1e-10)       # the surrogate is the game itself


def test_shapley_taylor():
    c = {(): 1.0, (0,): 1.0, (1,): 2.0, (0, 1): 0.6, (0, 1, 2): 0.9, (1, 2, 3): -0.3}
    st2 = indices.shapley_taylor(surrogate_of(c, 4), order=2)
    assert st2[(0,)] == 1.0 and st2[(1,)] == 2.0 and st2[()] == 1.0                 # below the order: the Möbius terms
    assert st2[(0, 1)] == pytest.approx(0.6 + 0.9 / 3)                              # a(S) + its share of each superset
    assert st2[(1, 2)] == pytest.approx(0.9 / 3 - 0.3 / 3) and st2[(2, 3)] == pytest.approx(-0.3 / 3)
    f = game(c)
    assert sum(w for s, w in st2.items() if s) == pytest.approx(f(np.ones((1, 4)))[0] - f(np.zeros((1, 4)))[0])
    pairs = random_game(np.random.default_rng(3), 6, 2)
    exact = indices.shapley_taylor(surrogate_of(pairs, 6), order=2)
    assert {s: w for s, w in exact.items() if abs(w) > 1e-12} == pytest.approx(
        {s: w for s, w in pairs.items() if abs(w) > 1e-12})                         # order ≥ the game's: Möbius
    with pytest.raises(ValueError):
        indices.shapley_taylor(surrogate_of(pairs, 6), order=0)


def test_removal_finds_the_masks_the_surrogate_ranks_first():
    c = {(): 0.0, (0,): -1.0, (1,): 3.0, (2,): 0.5, (3,): -2.5, (1, 3): 2.0}
    s, f = surrogate_of(c, 4), game(c)
    for r in (1, 2, 3):
        brute = max((m for m in all_masks(4) if m.sum() == 4 - r),
                    key=lambda m: abs(s.predict(np.ones((1, 4)))[0] - s.predict(m[None])[0]))
        assert (evaluate.removal_mask(s, r) == brute).all()
    curve = evaluate.removal(s, f, rs=(1, 2))
    f1 = f(np.ones((1, 4)))[0]
    assert [row["r"] for row in curve] == [1, 2]
    for row in curve:
        m = np.ones(4, dtype=bool)
        m[row["removed"]] = False
        assert row["value"] == pytest.approx(f(m[None])[0]) and row["rem"] == pytest.approx(abs(f1 - row["value"]) / abs(f1))
        assert row["predicted_change"] == pytest.approx(row["change"])              # this surrogate is exact


def test_removal_is_greedy_beyond_the_exact_search(monkeypatch):
    s = Surrogate(intercept=0.0, scores=np.array([1.0, -4.0, 2.0, 0.5, 3.0]))
    monkeypatch.setattr(evaluate, "EXACT_REMOVAL", 1)
    assert sorted(np.flatnonzero(~evaluate.removal_mask(s, 2)).tolist()) == [1, 3]   # greedy: −4 first, then 0.5
    monkeypatch.setattr(evaluate, "EXACT_REMOVAL", 20_000)
    assert sorted(np.flatnonzero(~evaluate.removal_mask(s, 2)).tolist()) == [2, 4]   # exact: 2 + 3 = 5 > 4 − 0.5
    with pytest.raises(ValueError):
        evaluate.removal_mask(s, 6)


def test_recovery():
    s = surrogate_of({(): 0.0, (0,): 5.0, (1,): -4.0, (2, 3): 3.0, (4,): 0.1}, 5)
    assert evaluate.recovery(s, {0, 2}, 1) == 1.0                                   # the top term is unit 0
    assert evaluate.recovery(s, {0, 2}, 2) == pytest.approx((1 + 0) / 2)
    assert evaluate.recovery(s, {0, 2}, 3) == pytest.approx((1 + 0 + 0.5) / 3)
    assert evaluate.recovery(s, {0}, 9) == pytest.approx(1 / 9)             # four non-zero terms; the slots left add 0
    bad = surrogate_of({(): 0.0, (0,): float("nan"), (1,): -4.0}, 3)
    with pytest.raises(ValueError, match="finite"):                       # not a ZeroDivisionError (review 11-07)
        evaluate.recovery(bad, {0}, 1)


def test_explain_reports_removal_and_the_indices():
    e = ca.explain(rio(), ca.ToyModel.demo(), ca.LogProb(), method="kernelshap", budget=8, removal=(1, 2))
    assert e.removal[0]["removed"] == [2] and e.removal[0]["change"] < 0           # d3 first: the bad answer falls
    assert e.removal[0]["change"] == pytest.approx(-D3_SCORE)
    assert e.shapley() == pytest.approx(e.scores) and e.banzhaf() == pytest.approx(e.scores)   # additive: all agree
    assert "removing d3: the value moves by" in e.table() and e.as_dict()["removal"][0]["r"] == 1


class FakeValues:
    def __init__(self, dict_values, baseline_value=0.0):
        self.dict_values, self.baseline_value = dict_values, baseline_value


def test_the_shapiq_adapter(monkeypatch):
    seen = {}

    class FakeFBII:
        def __init__(self, n, max_order=2, random_state=None):
            seen.update(n=n, max_order=max_order, seed=random_state)

        def approximate(self, budget, g):
            rng = np.random.default_rng(0)
            values = g(rng.random((budget // 2, seen["n"])) < 0.5)
            g(np.ones((budget - budget // 2, seen["n"]), dtype=bool))
            seen["values"] = values
            return FakeValues({(): 1.0, (0,): 2.0, (0, 1): 0.5, (2,): 0.0}, baseline_value=1.0)

    monkeypatch.setitem(sys.modules, "shapiq", types.SimpleNamespace(RegressionFBII=FakeFBII))
    e = ca.attribute(lambda m: m[:, 0] * 2.0, 3, method=FaithBanzhaf(max_order=2, ridge=False), budget=10,
                     test_masks=0)                               # shapiq's own answer, passed through
    assert seen["n"] == 3 and seen["max_order"] == 2 and isinstance(seen["seed"], int) and len(seen["values"]) == 5
    assert e.masks == 10 and e.surrogate.intercept == 1.0 and e.scores == {"0": 2.0, "1": 0.0, "2": 0.0}
    assert e.interactions == {("0", "1"): 0.5} and e.method == "FaithBanzhaf(max_order=2, ridge=False)"


def test_the_adapter_seeds_the_global_generators(monkeypatch):
    """sparse_transform draws SPEX's masks from numpy's global generator, out of reach of shapiq's random_state (found
    2026-11-07): the adapter seeds it and Python's for the call, from the method's seed, and gives the caller's back."""
    import random

    drawn = []

    class FakeSPEX:
        def __init__(self, n, max_order=2, random_state=None):
            self.n = n

        def approximate(self, budget, g):
            masks = np.random.randint(2, size=(budget, self.n)).astype(bool)
            g(masks)
            drawn.append((masks, random.random()))
            return FakeValues({(): 0.0})

    monkeypatch.setitem(sys.modules, "shapiq", types.SimpleNamespace(RegressionFBII=FakeSPEX))
    np.random.seed(5)
    random.seed(5)
    np_before, py_before = np.random.get_state(), random.getstate()
    for seed in (3, 3, 4):
        ca.attribute(lambda m: m.sum(axis=1) * 1.0, 6, method=FaithBanzhaf(), budget=20, test_masks=0, seed=seed)
    np_after = np.random.get_state()
    assert np.array_equal(np_after[1], np_before[1]) and np_after[2:] == np_before[2:]
    assert random.getstate() == py_before
    (a, x), (b, y), (c, z) = drawn
    assert np.array_equal(a, b) and x == y and not np.array_equal(a, c) and x != z


def test_ridge_fit_is_exact_from_every_mask():
    """Without noise and from all 2ⁿ masks, the least penalty wins and the fit gives the game's own terms."""
    coefficients = random_game(np.random.default_rng(3), 8, 3)
    m = all_masks(8)
    got = indices.terms(ridge_fit(m, game(coefficients)(m), 3))
    assert max(abs(got.get(k, 0.0) - coefficients.get(k, 0.0)) for k in set(got) | set(coefficients)) < 1e-6


def test_ridge_fit_predicts_what_ridgecv_fitted():
    """The product-basis surrogate predicts what RidgeCV fitted in the ±1 basis, penalty and all."""
    from sklearn.linear_model import RidgeCV

    rng = np.random.default_rng(4)
    masks, test = rng.random((300, 9)) < 0.5, rng.random((200, 9)) < 0.5
    values = 2.0 * masks[:, 0] - masks[:, 1] * masks[:, 2] + rng.normal(0, 0.5, 300)
    sets = [list(s) for k in (1, 2) for s in itertools.combinations(range(9), k)]

    def design(x):
        return np.column_stack([(1 - 2 * x[:, s].astype(float)).prod(axis=1) for s in sets])

    model = RidgeCV(alphas=np.logspace(-6, 6, 100)).fit(design(masks), values)
    assert model.alpha_ > 1.0                                    # the penalty is not negligible here
    assert ridge_fit(masks, values, 2).predict(test) == pytest.approx(model.predict(design(test)), abs=1e-9)


def test_ridge_fit_holds_where_least_squares_overfits():
    """299 terms of order 3 at n = 12 on 330 masks with noise: least squares fits the noise, ridge does not."""
    rng = np.random.default_rng(5)
    truth = game({(): 0.5, (1,): 1.0, (4,): -1.0, (2, 7): 0.8, (3, 5, 9): -0.6, (0, 6): 0.7})
    masks, test = rng.random((330, 12)) < 0.5, rng.random((2000, 12)) < 0.5
    values = truth(masks) + rng.normal(0, 0.3 * truth(test).std(), 330)
    sets = [list(s) for k in (1, 2, 3) for s in itertools.combinations(range(12), k)]

    def design(x):
        return np.column_stack([np.ones(len(x))] + [x[:, s].prod(axis=1) for s in sets]).astype(float)

    least = design(test) @ np.linalg.lstsq(design(masks), values, rcond=None)[0]
    ridge = ridge_fit(masks, values, 3).predict(test)
    r2 = [evaluate.faithfulness(p, truth(test)) for p in (least, ridge)]
    assert r2[1] > 0.6 and r2[0] < 0.4, r2                     # measured 0.71 and 0.23 (2026-11-08)


def test_faith_banzhaf_by_ridge_refits_what_shapiq_asked(monkeypatch):
    """The refit takes every batch shapiq asked, each once, and not shapiq's answer. The values carry a part that no
    order-2 polynomial fits, so a batch dropped or counted twice moves the fit (the review of 2026-11-08: noise-free
    values let either batch alone give the same terms)."""
    asked = []

    class FakeFBII:                                              # asks two batches, then answers terms to be ignored
        def __init__(self, n, max_order=2, random_state=None):
            self.n = n

        def approximate(self, budget, g):
            rng = np.random.default_rng(0)
            for size in (budget // 2, budget - budget // 2):
                asked.append(rng.random((size, self.n)) < 0.5)
                g(asked[-1])
            return FakeValues({(): 9.0, (0,): 9.0})

    monkeypatch.setitem(sys.modules, "shapiq", types.SimpleNamespace(RegressionFBII=FakeFBII))
    planted = game({(): 1.0, (0,): 2.0, (1, 2): -1.5})

    def f(m):                                                    # deterministic per mask, beyond order 2
        return planted(m) + 0.8 * np.sin(np.asarray(m, dtype=float) @ (1.0 + np.arange(8)) * 1.3)

    e = ca.attribute(f, 8, method=FaithBanzhaf(max_order=2, ridge=True), budget=200, test_masks=0)
    assert e.method == "FaithBanzhaf(max_order=2, ridge=True)" and e.masks == 200
    test = np.random.default_rng(9).random((300, 8)) < 0.5
    every = np.concatenate(asked)
    assert e.surrogate.predict(test) == pytest.approx(ridge_fit(every, f(every), 2).predict(test), abs=1e-9)
    for wrong in (asked[0], asked[1], np.concatenate([asked[0], *asked])):   # what a dropped or doubled batch gives
        assert np.abs(ridge_fit(wrong, f(wrong), 2).predict(test) - e.surrogate.predict(test)).max() > 1e-3
    assert FaithBanzhaf(max_order=2).ridge is True and repr(FaithBanzhaf(max_order=2)) == "FaithBanzhaf(max_order=2, ridge=True)"
    assert repr(FaithBanzhaf(max_order=2, ridge=False)) == "FaithBanzhaf(max_order=2, ridge=False)"


def test_faith_banzhaf_keeps_least_squares_when_every_mask_was_asked(monkeypatch):
    """Ridge (the default) refits sampled masks. When shapiq asked all 2ⁿ masks, its least squares is the index
    exactly and is kept; ridge=False keeps it always."""
    class FakeFBII:                                              # asks the first `budget` masks; answers known terms
        def __init__(self, n, max_order=2, random_state=None):
            self.n = n

        def approximate(self, budget, g):
            g(np.array(list(itertools.product([False, True], repeat=self.n)))[:budget])
            return FakeValues({(): 9.0, (0,): 9.0})

    monkeypatch.setitem(sys.modules, "shapiq", types.SimpleNamespace(RegressionFBII=FakeFBII))
    f = game({(): 1.0, (0,): 2.0, (1, 2): -1.5})

    def fit(budget, **options):
        return ca.attribute(f, 4, method=FaithBanzhaf(max_order=2, **options), budget=budget, test_masks=0).surrogate

    assert (fit(16).intercept, fit(16).scores[0]) == (9.0, 9.0)                # all 16 masks: shapiq's answer
    assert fit(12).intercept != 9.0 and fit(12).scores[0] != 9.0              # 12 of 16: ridge's refit
    assert (fit(12, ridge=False).intercept, fit(12, ridge=False).scores[0]) == (9.0, 9.0)


def test_from_interaction_values_and_method_names():
    s = from_interaction_values(FakeValues({(0,): 1.5, (1, 2): -0.5}, baseline_value=0.25), 3)
    assert s.intercept == 0.25 and s.scores.tolist() == [1.5, 0.0, 0.0] and dict(s.interactions) == {(1, 2): -0.5}
    for name in ("faithbanzhaf", "faithshap", "spex", "proxyspex"):
        assert ca.methods.get_method(name).name == name
    with pytest.raises(ValueError):
        FaithBanzhaf(max_order=0)


def test_proxyspex_refuses_masks_its_trees_cannot_learn_from():
    """Found 2026-11-12: on 3 to 5 units (8 to 32 masks) ProxySPEX gave every score as zero with R² near 0 and no
    error, because LightGBM keeps at least 20 masks on each side of a split. It now refuses before anything is asked
    (and before shapiq is imported, so this test is quick)."""
    asked = []

    def f(masks):
        asked.append(len(masks))
        return masks[:, 0].astype(float)

    for n in (3, 4, 5):
        with pytest.raises(ValueError, match=f"only {2 ** n} different masks"):
            ca.attribute(f, n, method="proxyspex", budget=512)
    with pytest.raises(ValueError, match="a budget of at least 40 masks"):
        ca.attribute(f, 12, method="proxyspex", budget=39)
    assert asked == [] and ca.methods.ProxySPEX.least_masks == 40


def test_interaction_surrogates_predict_products():
    s = Surrogate(intercept=1.0, scores=np.array([0.0, 0.0, 0.0]), interactions={(0, 2): 2.0, (0, 1, 2): -1.0})
    m = np.array(list(itertools.product([0, 1], repeat=3)))
    assert s.predict(m) == pytest.approx(1 + 2 * m[:, 0] * m[:, 2] - m[:, 0] * m[:, 1] * m[:, 2])
