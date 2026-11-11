"""The per-unit methods (docs/methods.md) against exact answers. A game is given by its Möbius
coefficients a(S), f(m) = Σ_S a(S)·Π_{i∈S} m_i, which give closed forms written apart from the methods:
Shapley φ_i = Σ_{S∋i} a(S)/|S|, Banzhaf β_i = Σ_{S∋i} a(S)/2^(|S|−1), leave-one-out Σ_{S∋i} a(S). Brute force over
every mask is a second reference."""

import itertools
from math import comb, factorial

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from contextattr import LIME, Banzhaf, ContextCite, KernelSHAP, LeaveOneOut, attribute
from contextattr.methods import all_masks, get_method


def game(coefficients):
    """f(masks) for Möbius coefficients {subset tuple: a(S)}."""
    def f(masks):
        m = np.asarray(masks, dtype=float)
        return sum(a * (m[:, list(s)].prod(axis=1) if s else np.ones(len(m))) for s, a in coefficients.items())
    return f


def shapley(coefficients, n):
    return np.array([sum(a / len(s) for s, a in coefficients.items() if i in s) for i in range(n)])


def banzhaf(coefficients, n):
    return np.array([sum(a / 2 ** (len(s) - 1) for s, a in coefficients.items() if i in s) for i in range(n)])


def leave_one_out(coefficients, n):
    return np.array([sum(a for s, a in coefficients.items() if i in s) for i in range(n)])


def brute_force(f, n):
    """Shapley and Banzhaf values from every mask, by their definitions."""
    masks = all_masks(n)
    values = dict(zip(map(tuple, masks.astype(int)), f(masks)))
    phi, beta = np.zeros(n), np.zeros(n)
    for row in masks.astype(int):
        for i in np.flatnonzero(row == 0):
            with_i = row.copy()
            with_i[i] = 1
            gain = values[tuple(with_i)] - values[tuple(row)]
            s = row.sum()
            phi[i] += factorial(s) * factorial(n - s - 1) / factorial(n) * gain
            beta[i] += gain / 2 ** (n - 1)
    return phi, beta


def random_game(rng, n, order, density=0.6):
    coefficients = {(): float(rng.normal())}
    for k in range(1, order + 1):
        for s in itertools.combinations(range(n), k):
            if rng.random() < density:
                coefficients[s] = float(rng.normal())
    return coefficients


GAMES = [random_game(np.random.default_rng(seed), n, order) for seed, n, order in
         [(0, 1, 1), (1, 2, 2), (2, 3, 3), (3, 5, 2), (4, 6, 4), (5, 8, 3), (6, 10, 2), (7, 12, 3)]]


@pytest.mark.parametrize("coefficients", GAMES)
def test_the_closed_forms_match_brute_force(coefficients):
    n = 1 + max([max(s) for s in coefficients if s] + [0])
    phi, beta = brute_force(game(coefficients), n)
    assert phi == pytest.approx(shapley(coefficients, n), abs=1e-10)
    assert beta == pytest.approx(banzhaf(coefficients, n), abs=1e-10)


@pytest.mark.parametrize("coefficients", GAMES)
def test_exact_methods_give_exact_values(coefficients):
    n = 1 + max([max(s) for s in coefficients if s] + [0])
    f = game(coefficients)
    full = 2 ** n
    assert attribute(f, n, method="kernelshap", budget=full).surrogate.scores == pytest.approx(
        shapley(coefficients, n), abs=1e-9)
    assert attribute(f, n, method="banzhaf", budget=full).surrogate.scores == pytest.approx(
        banzhaf(coefficients, n), abs=1e-9)
    loo = attribute(f, n, method="loo")
    assert loo.surrogate.scores == pytest.approx(leave_one_out(coefficients, n), abs=1e-12)
    assert loo.masks == n + 1


@given(st.integers(1, 7), st.integers(0, 2 ** 31), st.integers(1, 4))
def test_exact_values_on_any_game(n, seed, order):
    coefficients = random_game(np.random.default_rng(seed), n, min(order, n))
    f = game(coefficients)
    k = KernelSHAP().fit(all_masks(n), f(all_masks(n)))
    assert k.scores == pytest.approx(shapley(coefficients, n), abs=1e-8)
    full, empty = f(np.ones((1, n)))[0], f(np.zeros((1, n)))[0]
    assert k.scores.sum() == pytest.approx(full - empty, abs=1e-9)         # efficiency
    b = Banzhaf().fit(all_masks(n), f(all_masks(n)))
    assert b.scores == pytest.approx(banzhaf(coefficients, n), abs=1e-8)


def test_axioms():
    """Symmetry: two units with the same part in the game get the same score; dummy: a unit that does nothing gets 0."""
    coefficients = {(): 1.0, (0,): 2.0, (1,): 2.0, (0, 1): -1.5, (0, 2): 0.7, (1, 2): 0.7}   # unit 3 does nothing
    f, n = game(coefficients), 4
    for method in ("kernelshap", "banzhaf"):
        s = attribute(f, n, method=method, budget=16).surrogate.scores
        assert s[0] == pytest.approx(s[1], abs=1e-12) and s[3] == pytest.approx(0, abs=1e-12)


def test_an_additive_game_is_recovered_by_every_method():
    rng = np.random.default_rng(11)
    n = 10
    w = rng.normal(size=n)
    coefficients = {(): 0.5, **{(i,): float(w[i]) for i in range(n)}}
    f = game(coefficients)
    for method, budget, tol in [("loo", None, 1e-12), ("banzhaf", 64, 1e-9), ("kernelshap", 64, 1e-9),
                                (ContextCite(standardize=False, alpha=1e-6), 64, 1e-4), (LIME(alpha=1e-9), 64, 1e-6)]:
        e = attribute(f, n, method=method, budget=budget or 1)
        assert e.surrogate.scores == pytest.approx(w, abs=tol), method
        assert e.r2 == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("method, reference, tol", [("kernelshap", shapley, 0.10), ("banzhaf", banzhaf, 0.35)])
def test_sampled_estimates_converge(method, reference, tol):
    """A 12-unit game with interactions up to order 3, which 4,096 masks would solve exactly: ten times the budget at
    least halves the largest error (measured 2026-11-06, mean of 3 seeds: KernelSHAP 0.43 → 0.064, Banzhaf 0.89 →
    0.23, as 1/√budget predicts)."""
    exact = reference(GAMES[7], 12)
    error = {b: np.mean([np.abs(attribute(game(GAMES[7]), 12, method=method, budget=b, seed=s, test_masks=0)
                                .surrogate.scores - exact).max() for s in range(3)]) for b in (300, 3000)}
    assert error[3000] < error[300] / 2 and error[3000] < tol, error


@given(st.integers(2, 13), st.integers(0, 2 ** 31), st.integers(0, 3))
def test_paired_kernelshap_is_exact_for_pairwise_games(n, seed, extra):
    """With every drawn mask beside its complement, a game without interactions above pairs gets its exact Shapley
    values from any budget the method accepts."""
    coefficients = random_game(np.random.default_rng(seed), n, 2)
    budget = KernelSHAP.least_budget(n) * (1 + extra)
    e = attribute(game(coefficients), n, method="kernelshap", budget=budget, seed=seed, test_masks=0)
    assert e.surrogate.scores == pytest.approx(shapley(coefficients, n), abs=1e-8)


def test_kernelshap_enumerates_the_heaviest_sizes_then_draws_the_rest():
    n, budget = 16, 4002
    masks = KernelSHAP().masks(n, budget, np.random.default_rng(0))
    sizes = masks.sum(axis=1)
    assert masks[0].sum() == 0 and masks[1].all() and len(masks) == budget
    for k in (1, 2, 3, 13, 14, 15):                                     # complete: each coalition once
        assert (sizes == k).sum() == comb(n, k) == len({tuple(r) for r in masks[sizes == k]})
    drawn = masks[2 + 2 * (comb(16, 1) + comb(16, 2) + comb(16, 3)):]
    assert len(drawn) == 2608 and (drawn[0::2] ^ drawn[1::2]).all()     # the rest drawn in complementary pairs
    first = drawn[0::2].sum(axis=1)
    assert set(first) <= set(range(4, 13))
    p = np.array([1 / (k * (n - k)) for k in range(4, 13)])
    assert np.bincount(first, minlength=13)[4:] / len(first) == pytest.approx(p / p.sum(), abs=0.03)
    assert len(KernelSHAP().masks(6, 15, np.random.default_rng(0))) == 14   # one mask left over cannot be paired


def test_kernelshap_refuses_an_undetermined_fit():
    masks = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0], [0, 1, 1, 1]], dtype=bool)
    with pytest.raises(ValueError, match="undetermined"):
        KernelSHAP().fit(masks, np.zeros(4))


def test_contextcite_finds_a_sparse_support():
    rng = np.random.default_rng(3)
    n = 40
    w = np.zeros(n)
    w[[3, 17, 29]] = [1.5, -2.0, 1.0]
    f = lambda m: 0.2 + m.astype(float) @ w + 0.01 * rng.normal(size=len(m))     # noqa: E731
    e = attribute(f, n, method="contextcite", budget=128)
    assert set(np.flatnonzero(np.abs(e.surrogate.scores) > 0.1)) == {3, 17, 29}
    assert (e.surrogate.scores == 0).sum() > n / 2                       # most scores are exactly 0
    raw = attribute(f, n, method=ContextCite(standardize=False), budget=128).surrogate.scores
    assert set(np.flatnonzero(np.abs(raw) > 0.1)) == {3, 17, 29}


def test_lime_draws_and_weights_masks_as_its_text_explainer():
    lime = LIME()
    masks = lime.masks(10, 500, np.random.default_rng(0))
    assert masks[0].all() and (masks[1:].sum(axis=1) <= 9).all()        # the full prompt, then 1..n removed
    removed = 10 - masks[1:].sum(axis=1)
    assert set(removed) == set(range(1, 11))
    w = lime.weights(np.array([[1] * 10, [1] * 5 + [0] * 5, [0] * 10]))
    d = (1 - np.sqrt([1.0, 0.5, 0.0])) * 100
    assert w == pytest.approx(np.sqrt(np.exp(-d ** 2 / 25 ** 2)))


def test_methods_by_name_and_their_checks():
    assert isinstance(get_method("loo"), LeaveOneOut) and isinstance(get_method(Banzhaf()), Banzhaf)
    with pytest.raises(ValueError):
        get_method("shap")
    for make in (lambda: Banzhaf().masks(5, 3, np.random.default_rng()),
                 lambda: KernelSHAP().masks(5, 4, np.random.default_rng()),
                 lambda: ContextCite(alpha=0), lambda: LIME(kernel_width=0)):
        with pytest.raises(ValueError):
            make()
    with pytest.raises(ValueError, match="empty and the full"):
        KernelSHAP().fit(np.ones((3, 2), dtype=bool), np.zeros(3))
    with pytest.raises(ValueError, match="full prompt"):
        LeaveOneOut().fit(np.zeros((3, 2), dtype=bool), np.zeros(3))


def test_a_constant_value_gives_zero_scores():
    e = attribute(lambda m: np.full(len(m), 2.0), 5, method="contextcite", budget=32)
    assert (e.surrogate.scores == 0).all() and e.surrogate.intercept == 2.0
    assert np.isnan(e.r2)
