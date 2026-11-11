"""Bugs found by a review of the code, each pinned by a test: KernelSHAP beyond 56 units,
a sampled target's sampling and a check's code in a run's settings, resuming a changed prompt or a cut line, Banzhaf's
undetermined fits, the command line's checks, Recovery@r's zero and tied terms, and attribute's masks."""

import json
import re
from fractions import Fraction
from math import comb

import numpy as np
import pytest
from prompt_helpers import RIO_LINE, rio

import contextattr as ca
from contextattr import cli, evaluate
from contextattr.methods import Banzhaf, KernelSHAP, Surrogate
from contextattr.responses import describe


@pytest.mark.parametrize("n", [57, 60, 67, 100, 300])
def test_kernelshap_beyond_56_units(n):
    w = np.random.default_rng(n).normal(size=n)
    e = ca.attribute(lambda m: m.astype(float) @ w, n, method="kernelshap", budget=4 * n + 2, test_masks=0)
    assert e.surrogate.scores == pytest.approx(w, abs=1e-9)                # an additive game: exact


def test_kernelshap_masks_for_2000_units():
    m = KernelSHAP().masks(2000, 6000, np.random.default_rng(0))
    assert m.shape == (6000, 2000) and m[0].sum() == 0 and m[1].all()
    assert (m.sum(axis=1) == 1).sum() == 2000 and (m.sum(axis=1) == 1999).sum() == 2000


@pytest.mark.parametrize("n", [60, 100, 1200])
def test_kernelshap_weighs_masks_from_elsewhere_by_the_kernel(n):
    """Masks KernelSHAP did not draw get π(s) each: the fit equals the constrained weighted least squares solved by
    hand, with π(s) from exact fractions, for a game with interactions (where the weights change the answer)."""
    rng = np.random.default_rng(n)
    inner = rng.random((3 * n if n > 100 else 3000, n)) < rng.uniform(0.45, 0.55, (3 * n if n > 100 else 3000, 1))
    masks = np.vstack([np.zeros((1, n), bool), np.ones((1, n), bool), inner])
    w = rng.normal(size=n)
    x = masks.astype(float)
    values = x @ w + 2.0 * x[:, 0] * x[:, 1] - 1.5 * x[:, 2] * x[:, 3] * x[:, 4]
    got = KernelSHAP().fit(masks, values).scores
    sizes = inner.sum(axis=1)
    exact = {int(s): Fraction(n - 1, comb(n, int(s)) * int(s) * (n - int(s))) for s in set(sizes.tolist())}
    top = max(exact.values())
    pi = np.array([float(exact[int(s)] / top) for s in sizes])
    a, y, delta = inner.astype(float), values[2:] - values[0], values[1] - values[0]
    basis = np.linalg.svd(np.ones((1, n)))[2][1:].T          # φ = Δ/n + basis·z: Σφ = Δ whatever z
    root = np.sqrt(pi)[:, None]

    def fit(r):
        z = np.linalg.lstsq(r * (a @ basis), (r * (y - a.sum(axis=1) * delta / n)[:, None]).ravel(), rcond=None)[0]
        return delta / n + basis @ z

    want = fit(root)
    assert got == pytest.approx(want, abs=1e-8)
    assert got.sum() == pytest.approx(delta)
    assert np.abs(fit(np.ones_like(root)) - want).max() > 1e-3   # the weights matter here


def test_a_sampled_target_s_sampling_is_part_of_the_settings(tmp_path):
    cold = ca.checks.regex(r"\bcold\b")
    a = ca.Sampled(cold, samples=4)
    b = ca.Sampled(cold, samples=4, sampling=ca.Sampling(temperature=1.5, top_p=0.5, max_tokens=8, seed=123))
    assert repr(a) != repr(b) and "temperature=1.5" in repr(b) and "seed=123" in repr(b)
    assert repr(a) == repr(ca.Sampled(cold, samples=4, sampling=ca.Sampling(temperature=0.7, top_p=0.95)))
    ca.Run(ca.ToyModel.demo(), [rio()], target=a, method="loo").execute(tmp_path / "r")
    with pytest.raises(ValueError, match="other settings"):
        ca.Run(ca.ToyModel.demo(), [rio()], target=b, method="loo").execute(tmp_path / "r")


def test_checks_are_told_apart_by_their_code_and_values():
    f, g = (lambda p, r: "cold" in r), (lambda p, r: "hot" in r)              # one line: one source text
    assert describe(f) != describe(g) and describe(f) == describe(f)
    assert repr(ca.Sampled(f)) != repr(ca.Sampled(g))
    run = lambda check: ca.Run(ca.ToyModel.demo(), [rio()], sample=ca.Sampling(n=2), check=check)  # noqa: E731
    assert run(f).settings()["check"] != run(g).settings()["check"]

    def word(w):
        return lambda p, r: w in r

    assert describe(word("cold")) != describe(word("hot")) and describe(word("cold")) == describe(word("cold"))
    assert describe(lambda p, r, w="cold": w in r) != describe(lambda p, r, w="hot": w in r)
    assert re.fullmatch(r".*<lambda> \[[0-9a-f]{12}\]", describe(lambda p, r: r in {"cold", "chilly", "icy"}))

    class Judge:
        def __init__(self, threshold, client):
            self.threshold, self.client = threshold, client

        def __call__(self, p, r):
            return len(r) > self.threshold

    assert describe(Judge(3, object())) == describe(Judge(3, object()))      # a client by its type, not its address
    assert describe(Judge(3, object())) != describe(Judge(4, object()))
    assert ca.checks.regex("x").__name__ != ca.checks.regex("x", flags=0).__name__
    assert describe(ca.checks.regex("x")) == "regex('x', negative=True)"


def test_resuming_a_changed_prompt_is_refused(tmp_path):
    ca.Run(ca.ToyModel.demo(), [rio()], method="loo").execute(tmp_path / "r")
    p = rio()
    content = list(p.messages[1].content)
    content[3] = ca.Unit("d3", "Summer is hot.", sep="\n", tag="doc")
    changed = ca.Prompt(p.id, messages=[p.messages[0], ca.Message("user", tuple(content))], responses=p.responses)
    with pytest.raises(ValueError, match="changed since"):
        ca.Run(ca.ToyModel.demo(), [changed], method="loo").execute(tmp_path / "r")
    renamed = ca.Prompt("rio-01b", messages=changed.messages, responses=changed.responses)
    assert len(ca.Run(ca.ToyModel.demo(), [renamed], method="loo").execute(tmp_path / "r")) == 1


def test_execute_gives_the_prompts_given_and_survives_a_cut_line(tmp_path):
    two = ca.Prompt("rio-02", messages=rio().messages, responses=rio().responses)
    first = ca.Run(ca.ToyModel.demo(), [rio(), two], method="loo").execute(tmp_path / "r")
    assert [e.prompt_id for e in ca.Run(ca.ToyModel.demo(), [two], method="loo").execute(tmp_path / "r").explanations] \
        == ["rio-02"]
    assert len(ca.Results.load(tmp_path / "r")) == 2                       # the directory keeps both
    lines = tmp_path / "r" / "explanations.jsonl"
    text = lines.read_text().splitlines()
    lines.write_text(text[0] + "\n" + text[1][: len(text[1]) // 2])         # a kill during the last append
    assert len(ca.Results.load(tmp_path / "r")) == 1                       # reading leaves the cut line out
    again = ca.Run(ca.ToyModel.demo(), [two, rio()], method="loo").execute(tmp_path / "r")
    assert [e.prompt_id for e in again.explanations] == ["rio-02", "rio-01"]
    assert again.explanations[0].scores == pytest.approx(first.explanations[1].scores)
    rows = [json.loads(line) for line in lines.read_text().splitlines()]
    assert len(rows) == 2 and all(r["seconds"] >= 0 and len(r["input"]) == 64 for r in rows)
    lines.write_text("{not a row}\n" + lines.read_text())
    with pytest.raises(ValueError, match="damaged"):
        ca.Run(ca.ToyModel.demo(), [rio()], method="loo").execute(tmp_path / "r")


def test_banzhaf_refuses_an_undetermined_fit():
    masks = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=bool)   # units 0 and 1 always together
    with pytest.raises(ValueError, match="undetermined"):
        Banzhaf().fit(masks, np.arange(4.0))
    refused = 0
    for seed in range(40):                                                  # small budgets: right, or refused
        try:
            e = ca.attribute(lambda m: m.astype(float) @ [1.0, -2.0, 0.5, 3.0, 0.25], 5, method="banzhaf",
                             budget=6, seed=seed, test_masks=0)
        except ValueError as error:
            assert "undetermined" in str(error)
            refused += 1
            continue
        assert e.surrogate.scores == pytest.approx([1.0, -2.0, 0.5, 3.0, 0.25], abs=1e-9)
    assert 0 < refused < 40


def test_the_command_line_checks_its_check(tmp_path):
    (tmp_path / "p.jsonl").write_text(RIO_LINE + "\n")
    base = 'prompts = "p.jsonl"\nout = "o"\n[model]\nuse = "contextattr:ToyModel.demo"\n'
    for check in ('reference = "contain"', 'reference = true', 'regex = "x"\nnegative = "false"', 'regex = 3'):
        (tmp_path / "c.toml").write_text(base + "[check]\n" + check + "\n")
        with pytest.raises(ValueError, match=r"\[check\]"):
            cli.load(tmp_path / "c.toml")
    for check, name in (('reference = "exact"', "reference(contains=False)"),
                        ('regex = "x"\nnegative = false', "regex('x', negative=False)")):
        (tmp_path / "c.toml").write_text(base + "[check]\n" + check + "\n[sample]\nn = 2\n")
        assert cli.load(tmp_path / "c.toml")[0].settings()["check"] == name


def test_recovery_ignores_zero_terms_and_shares_ties():
    s = Surrogate(intercept=0.0, scores=np.array([0, 0, 0, 0, 0, 0, 0.9, 0.0]))
    assert evaluate.recovery(s, {0, 1}, 3) == 0.0                          # no term of these units is non-zero
    assert evaluate.recovery(s, {6, 7}, 3) == pytest.approx(1 / 3)         # the one term, then two empty slots
    assert evaluate.recovery(s, {6}, 1) == 1.0
    tie = Surrogate(intercept=0.0, scores=np.array([1.0, 1.0, 1.0, 2.0]))
    assert evaluate.recovery(tie, {0, 3}, 2) == pytest.approx((1 + 1 / 3) / 2)   # one slot shared by three equal terms
    assert evaluate.recovery(tie, {2, 3}, 2) == evaluate.recovery(tie, {0, 3}, 2)
    with pytest.raises(ValueError):
        evaluate.recovery(s, {6}, 0)


def test_attribute_gives_f_its_own_copy():
    def f(m):
        out = m.astype(float) @ [1.0, -2.0, 0.5, 3.0]
        m[:, 0] = True                                                     # a function that writes into its input
        return out

    for method in ("banzhaf", "kernelshap"):
        e = ca.attribute(f, 4, method=method, budget=16, test_masks=16)
        assert e.surrogate.scores == pytest.approx([1.0, -2.0, 0.5, 3.0]) and e.r2 == pytest.approx(1.0)
