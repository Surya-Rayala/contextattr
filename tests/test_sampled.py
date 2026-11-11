"""Sampling's model-facing parts (interface §4–5): checks, sampled responses, the Sampled target against the toy model's
known rates (within their standard error), its noise from fresh seeds, and the intervention that samples again."""

import itertools

import numpy as np
import pytest
from prompt_helpers import rio

import contextattr as ca

ALL3 = np.array(list(itertools.product([0, 1], repeat=3)))
COLD = ca.checks.regex(r"\bcold\b")


def known_rate(d1, d2, d3):
    """The demo model's share of "cold" replies above temperature 0: uniform over the cues present (Brazil's summer in
    d2, its winter in d3), the default reply ("I do not know.") when neither is."""
    return d3 / (d2 + d3) if d2 + d3 else 0.0


def test_checks():
    p = rio()
    assert COLD(p, "It is cold today.") == ca.Verdict("negative", span=(6, 10))
    assert COLD(p, "It is hot.") == ca.Verdict("positive")
    assert ca.checks.regex("hot", negative=False)(p, "so hot") == ca.Verdict("positive", span=(3, 6))
    ref = ca.checks.reference()
    assert ref(p, "Well: hot and HUMID; it is summer in Brazil!").label == "positive"
    assert ref(p, "It is cold.").label == "negative"
    assert ca.checks.reference(contains=False)(p, "hot and humid it is summer in brazil").label == "positive"
    with pytest.raises(ValueError):
        ref(ca.Prompt("p", text=[ca.Unit("a", "A")]), "x")
    v = ca.responses.verdict
    assert v(True).label == "negative" and v(False).label == "positive" and v(None).value is None
    assert v(0.25).value == 0.25 and v("negative").value == 1.0 and v("positive").value == 0.0
    for bad in (lambda: v([1]), lambda: ca.Verdict("bad"), lambda: ca.Verdict(score=float("nan"))):
        with pytest.raises((TypeError, ValueError)):
            bad()


def test_sample_adds_labelled_replies():
    p = ca.sample(rio(), ca.ToyModel.demo(), ca.Sampling(n=8, temperature=0.7, seed=0), check=COLD)
    new = p.responses[1:]
    assert [r.id for r in new] == [f"sample-{k}" for k in range(8)]
    assert all((r.label == "negative") == ("cold" in r.text) for r in new)
    assert all(r.span == (6, 10) for r in new if r.label == "negative")
    again = ca.sample(rio(), ca.ToyModel.demo(), ca.Sampling(n=8, temperature=0.7, seed=0), check=COLD)
    assert again == p                                                     # the same seed, the same replies
    greedy = ca.sample(rio(), ca.ToyModel.demo(), ca.Sampling(n=2), check=None)
    assert [r.text for r in greedy.responses[1:]] == ["It is hot and humid: it is summer in Brazil."] * 2
    assert all(r.label is None for r in greedy.responses[1:])
    with pytest.raises(ValueError, match="already has"):
        ca.sample(p, ca.ToyModel.demo(), ca.Sampling(n=8, temperature=0.7, seed=0))


def test_sampled_values_match_the_known_rates():
    k = 400
    v = ca.Value(rio(), ca.ToyModel.demo(), ca.Sampled(COLD, samples=k))
    values = v(ALL3)
    for mask, value in zip(ALL3, values):
        p = known_rate(*mask)
        assert abs(value - p) <= 4 * np.sqrt(p * (1 - p) / k) + 1e-12, (mask, value)
    assert v.stats["requests"] == 8 and v.stats["generations"] == 8 * k
    v(ALL3)
    assert v.stats["requests"] == 8                                       # the replies are cached


def test_a_changed_check_asks_nothing_new():
    cache = ca.Cache()
    m = ca.ToyModel.demo()
    first = ca.Value(rio(), m, ca.Sampled(COLD, samples=50), cache=cache)
    first(ALL3)
    hot = ca.Value(rio(), m, ca.Sampled(ca.checks.regex(r"\bhot\b"), samples=50), cache=cache)
    hot_values = hot(ALL3)
    assert hot.stats["requests"] == 0
    assert hot_values[np.flatnonzero((ALL3 == [1, 1, 0]).all(axis=1))[0]] == 1.0


def test_sampled_noise_samples_with_fresh_seeds():
    v = ca.Value(rio(), ca.ToyModel.demo(), ca.Sampled(COLD, samples=100))
    n = v.noise(repeats=4)
    assert len(set(n["values"])) > 1 and 0 < n["std"] < 0.15            # binomial: about 0.05 at p = ½
    assert v.noise(repeats=4) == n                                         # and cached under their own keys
    assert ca.Sampled(COLD).sampling_for(2).seed == 2 and ca.Sampled(COLD).sampling_for(0).n == 4


def test_explain_with_a_sampled_target():
    e = ca.explain(rio(), ca.ToyModel.demo(), ca.Sampled(COLD, samples=200), method="kernelshap", budget=8)
    s = e.scores
    assert s["d3"] > 0.4 and s["d2"] < -0.2 and abs(s["d1"]) < 0.1
    assert e.requests["generations"] >= 8 * 200


def test_intervene():
    out = ca.intervene(rio(), ca.ToyModel.demo(), COLD, ["d3"], samples=400)
    assert abs(out["before"] - 0.5) <= 4 * 0.025 and out["after"] == 0.0
    assert out["change"] == pytest.approx(out["after"] - out["before"]) and 0 < out["se"] < 0.04
    assert out["dropped"] == ["d3"] and out["replies"] == [400, 400]
    assert ca.intervene(rio(), ca.ToyModel.demo(), COLD, "d2", samples=10)["after"] == 1.0


def test_sampled_needs_generate_and_a_check():
    class ScoreOnly:
        name = "s"

        def score(self, prompts, responses):
            return []

    with pytest.raises(ca.WrapperError, match="generate"):
        ca.Value(rio(), ScoreOnly(), ca.Sampled(COLD))
    with pytest.raises(TypeError):
        ca.Sampled("not a check")
    with pytest.raises(ValueError):
        ca.Sampled(COLD, samples=0)
    with pytest.raises(ValueError):
        ca.Value(rio(), ca.ToyModel.demo(), ca.Sampled(COLD), per_response=True)
    with pytest.raises(ValueError, match="labelled none"):
        ca.Value(rio(), ca.ToyModel.demo(), ca.Sampled(lambda p, r: None, samples=2))(ALL3[:1])
