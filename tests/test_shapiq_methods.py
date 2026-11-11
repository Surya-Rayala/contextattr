"""The interaction methods through shapiq, marked `shapiq`: run only with CA_TEST_SHAPIQ=1, because they take 5–7
minutes on a login node (importing shapiq 25–45 s, ProxySPEX's trees most of the rest). Planted sparse interactions must be recovered
exactly where there is no noise, and the indices read from a surrogate must equal shapiq's own exact ones."""

import time

import numpy as np
import pytest
from test_methods import game, random_game

import contextattr as ca
from contextattr.methods import FaithBanzhaf, FaithShap, ProxySPEX, SPEX, all_masks, indices
from contextattr.methods.indices import terms

pytestmark = pytest.mark.shapiq

PLANTED = {(): 0.5, (3,): 2.0, (7,): -1.5, (3, 11): 1.2, (2, 5, 9): -0.8, (14,): 0.6, (0, 18): -0.9}


def nonzero(surrogate, tol=1e-6):
    return {s: w for s, w in terms(surrogate).items() if abs(w) > tol}


def test_import_time():
    start = time.time()
    import shapiq  # noqa: F401
    print(f"import shapiq: {time.time() - start:.1f} s")


@pytest.mark.parametrize("method, tol", [(FaithBanzhaf(max_order=3), 1e-9), (FaithBanzhaf(max_order=3, ridge=False), 1e-9),
                                         (FaithShap(max_order=3), 1e-5)])
def test_faith_indices_from_every_mask_are_the_mobius_terms(method, tol):
    """An order-3 game's own Möbius terms. Faith-Banzhaf's default, ridge, keeps shapiq's least squares when every
    mask was asked, so both are exact. Faith-Shap comes within 1e-5, not exactly: shapiq's regression gives the empty
    and the full prompt a large finite weight in place of the kernel's infinite one (measured 2026-11-06: 4.1e-6 at any
    budget, while shapiq's exact FSII equals the Möbius terms to 6e-11)."""
    coefficients = random_game(np.random.default_rng(1), 8, 3)
    e = ca.attribute(game(coefficients), 8, method=method, budget=2 ** 8, test_masks=64)
    got = terms(e.surrogate)
    assert max(abs(got.get(k, 0.0) - coefficients.get(k, 0.0)) for k in set(got) | set(coefficients)) < tol
    assert e.r2 == pytest.approx(1.0, abs=1e-6)


def test_spex_recovers_planted_interactions_exactly():
    e = ca.attribute(game(PLANTED), 20, method=SPEX(degree=5), budget=6000, test_masks=256)
    print(f"SPEX: {e.masks} masks, R² {e.r2}")
    assert nonzero(e.surrogate, 1e-6) == pytest.approx(PLANTED, abs=1e-6)
    assert e.r2 == pytest.approx(1.0, abs=1e-9)


def test_spex_refuses_a_degree_shapiq_would_lower():
    asked = []

    def f(masks):
        asked.append(len(masks))
        return game(PLANTED)(masks)

    with pytest.raises(ValueError, match="lower it to 4"):          # 600 masks at n = 20: shapiq's t falls to 4
        ca.attribute(f, 20, method=SPEX(degree=5), budget=600, test_masks=0)
    assert asked == []                                              # refused before any value was asked


def test_spex_draws_the_same_masks_for_the_same_seed():
    """Found 2026-11-07: sparse_transform draws SPEX's masks from numpy's global generator, out of reach of shapiq's
    random_state, so the same seed gave other masks each run; the adapter now seeds it for the call."""
    asked, runs = [], []

    def f(masks):
        asked.append(np.array(masks))
        return game(PLANTED)(masks)

    for _ in range(2):
        np.random.rand(7)                                           # the caller's own draws must not matter
        e = ca.attribute(f, 20, method=SPEX(degree=3), budget=1200, test_masks=0, seed=2)
        runs.append((np.concatenate(asked), terms(e.surrogate)))
        asked.clear()
    assert np.array_equal(runs[0][0], runs[1][0]) and runs[0][1] == runs[1][1]


def test_spex_chase_depth_is_2t_unless_set_and_t_is_shapiq_s_own():
    """The soft decoder's Chase depth: 2t by default, as the SPEX authors' spectralexplain. chase_depth=degree builds
    the decoder shapiq builds itself (which leaves the depth to sparse_transform's code, whose own is t). No decoding:
    quick once shapiq is imported."""
    import shapiq

    def decoder(approximator):
        return approximator.decoder_args["source_decoder"]

    assert decoder(SPEX(degree=3).make(shapiq, 20, 0)).keywords["chase_depth"] == 6
    assert decoder(SPEX(degree=3, chase_depth=4).make(shapiq, 20, 0)).keywords["chase_depth"] == 4
    ours = decoder(SPEX(degree=3, chase_depth=3).make(shapiq, 20, 0))
    own = decoder(shapiq.SPEX(n=20, max_order=3, index="FBII", degree_parameter=3, random_state=0))
    code, own_code = ours.func.__self__, own.func.__self__
    assert own.keywords["chase_depth"] is None and own_code.chase_depth == 3                 # None: the code's t
    assert ours.func.__name__ == own.func.__name__ and ours.keywords == {**own.keywords, "chase_depth": 3}
    assert (code.n, code.k, code.chase_depth) == (own_code.n, own_code.k, own_code.chase_depth)
    assert repr(SPEX()) == "SPEX(degree=5, chase_depth=10)" and repr(SPEX(chase_depth=5)) == "SPEX(degree=5, chase_depth=5)"
    with pytest.raises(ValueError):
        SPEX(chase_depth=0)


def test_spex_with_t_recovers_planted_interactions_exactly():
    """Depth t, shapiq's own, is exact without noise too (the default, 2t: test_spex_recovers_planted_interactions_exactly)."""
    e = ca.attribute(game(PLANTED), 20, method=SPEX(degree=5, chase_depth=5), budget=6000, test_masks=256)
    assert nonzero(e.surrogate, 1e-6) == pytest.approx(PLANTED, abs=1e-6) and e.method == "SPEX(degree=5, chase_depth=5)"


def test_proxyspex_comes_close():
    e = ca.attribute(game(PLANTED), 20, method=ProxySPEX(), budget=1000, test_masks=256)
    print(f"ProxySPEX: {e.masks} masks, R² {e.r2}")
    assert e.r2 > 0.95


def test_indices_equal_shapiq_s_exact_ones():
    import shapiq

    coefficients = random_game(np.random.default_rng(2), 7, 3)
    f = game(coefficients)
    exact = shapiq.ExactComputer(game=f, n_players=7)
    sv, st = exact(index="SV", order=1), exact(index="STII", order=2)
    s = ca.attribute(f, 7, method="kernelshap", budget=2 ** 7, test_masks=0).surrogate
    full = FaithBanzhaf(max_order=7).run(f, 7, 2 ** 7, np.random.default_rng(0))[0]
    assert indices.shapley(full) == pytest.approx([sv[(i,)] for i in range(7)], abs=1e-8)
    assert s.scores == pytest.approx([sv[(i,)] for i in range(7)], abs=1e-8)
    ours = indices.shapley_taylor(full, order=2)
    assert {k: ours.get(k, 0.0) for k in st.dict_values if len(k) == 2} == pytest.approx(
        {k: v for k, v in st.dict_values.items() if len(k) == 2}, abs=1e-8)
    assert all_masks(7).shape == (128, 7)
