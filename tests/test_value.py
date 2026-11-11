"""The value function (interface §1, §5): values equal the toy model's formula; each distinct request is asked once,
in batches, cached and resumed after a failure; a user's own function; the noise check; and every check."""

import itertools

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from prompt_helpers import masks_for, prompts, rio

from contextattr import (Cache, Choice, Contrast, LogProb, Prompt, Response, Scored, ToyModel, Unit, Value,
                         WrapperError)
from contextattr.wrapper.toy import log_sigmoid as ls

ALL3 = np.array(list(itertools.product([0, 1], repeat=3)))      # every mask of the three Rio documents
GOOD = Response("good-1", "It is hot and humid: it is summer in Brazil.", label="positive")


def rio2():
    """The Rio prompt with a positive response beside its negative one."""
    p = rio()
    return Prompt(p.id, messages=p.messages, reference=p.reference, responses=(*p.responses, GOOD))


def bad_mean(d1, d2, d3):
    """The demo model's mean log-probability of "It is cold, because Carnival is in winter.", by its formula:
    eight words at bias −1; "cold" and "winter" +2 each when document d3 (Brazil's winter) is kept."""
    return (6 * ls(-1.0) + 2 * ls(-1.0 + 2.0 * d3)) / 8


def good_mean(d1, d2, d3):
    """… of "It is hot and humid: it is summer in Brazil.": ten words; with d2 (Brazil's summer) "hot" and "summer"
    +2 and "humid" +1, and with d1 (February or March) as well "hot" and "summer" +1.5 more."""
    hot = ls(-1.0 + 2.0 * d2 + 1.5 * d1 * d2)
    return (7 * ls(-1.0) + 2 * hot + ls(-1.0 + 1.0 * d2)) / 10


class Counting(ToyModel):
    """The demo model, recording the size of every call."""

    def __init__(self, *args, fail_on_call=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.batches, self.fail_on_call = [], fail_on_call

    def score(self, prompts, responses):
        if self.fail_on_call == len(self.batches) + 1:
            raise RuntimeError("connection lost")
        self.batches.append(len(prompts))
        return super().score(prompts, responses)


def test_values_equal_the_formula():
    v = Value(rio(), ToyModel.demo(), LogProb())
    assert v(ALL3) == pytest.approx([bad_mean(*m) for m in ALL3], rel=1e-12)
    c = Value(rio2(), ToyModel.demo(), Contrast())
    assert c(ALL3) == pytest.approx([bad_mean(*m) - good_mean(*m) for m in ALL3], rel=1e-12)
    cols = Value(rio2(), ToyModel.demo(), Contrast(), per_response=True)(ALL3)
    assert cols.shape == (8, 2) and [i.id for i in c.items] == ["bad-1", "good-1"]
    assert cols[:, 1] == pytest.approx([good_mean(*m) for m in ALL3], rel=1e-12)


def test_choice_scores_each_text_in_full():
    v = Value(rio(), ToyModel.demo(), Choice("cold", "hot"))
    assert v(ALL3) == pytest.approx([ls(-1 + 2.0 * d3) - ls(-1 + 2.0 * d2 + 1.5 * d1 * d2) for d1, d2, d3 in ALL3])


def test_each_distinct_request_is_asked_once_in_batches():
    m = Counting.demo()
    v = Value(rio2(), m, Contrast(), batch_size=5)
    first = v(np.vstack([ALL3, ALL3]))                    # 16 masks, 8 distinct, 2 responses each
    assert m.batches == [5, 5, 5, 1] and v.stats == {"requests": 16, "reused": 0, "calls": 4}
    assert (v(ALL3) == first[:8]).all() and m.batches == [5, 5, 5, 1]   # all from the cache
    assert v.stats["reused"] == 16
    assert Value(rio2(), m, LogProb("bad-1"), cache=v.cache)(ALL3) == pytest.approx(
        [bad_mean(*x) for x in ALL3]) and m.batches == [5, 5, 5, 1]        # another target, the same answers


def test_a_failed_run_resumes_from_its_cache(tmp_path):
    path = tmp_path / "run" / "cache.sqlite"
    failing = Counting.demo()
    failing.fail_on_call = 2
    with pytest.raises(RuntimeError, match="connection lost"):
        Value(rio(), failing, LogProb(), cache=str(path), batch_size=3)(ALL3)
    healthy = Counting.demo()
    v = Value(rio(), healthy, LogProb(), cache=str(path), batch_size=3)
    assert v(ALL3) == pytest.approx([bad_mean(*m) for m in ALL3], rel=1e-12)
    assert healthy.batches == [3, 2] and v.stats["reused"] == 3      # only the five answers the failure left


def test_the_wrapper_cannot_spoil_the_cache_by_changing_its_inputs():
    class Mutating(ToyModel):
        def score(self, prompts, responses):
            out = super().score(prompts, responses)
            for p in prompts:
                p.append({"role": "user", "content": "winter in brazil"})
            return out

    v = Value(rio(), Mutating.demo(), LogProb())
    first = v(ALL3)
    assert (v(ALL3) == first).all() and first == pytest.approx([bad_mean(*m) for m in ALL3])


def test_a_function_gives_the_value_itself():
    f = lambda ps: [p[1]["content"].count("Brazil") for p in ps]        # noqa: E731
    v = Value(rio(), function=f)
    assert v(ALL3).tolist() == [d2 + d3 for _, d2, d3 in ALL3]
    v(ALL3)
    assert v.stats["requests"] == 16                                     # unnamed: computed afresh
    named = Value(rio(), function=f, name="count-brazil")
    named(ALL3)
    named(ALL3)
    assert named.stats == {"requests": 8, "reused": 8, "calls": 1}


@pytest.mark.parametrize("f, error", [(lambda ps: [1.0], ValueError), (lambda ps: ["x"] * len(ps), TypeError),
                                      (lambda ps: [float("nan")] * len(ps), ValueError), (lambda ps: 3, TypeError)])
def test_a_bad_function_is_refused(f, error):
    with pytest.raises(error):
        Value(rio(), function=f)(ALL3)


def test_replace_reaches_the_rendering():
    v = Value(rio(), function=lambda ps: [p[1]["content"].count("[UNK]") for p in ps], replace="[UNK]")
    assert v(ALL3).tolist() == [3 - sum(m) for m in ALL3]


def test_noise():
    v = Value(rio(), ToyModel.demo(), LogProb())
    n = v.noise()
    assert n["std"] == 0.0 and n["values"] == pytest.approx([bad_mean(1, 1, 1)] * 4, rel=1e-12)

    class Drifting(ToyModel):
        calls = 0

        def score(self, prompts, responses):
            self.calls += 1
            return [Scored(tuple(x - 0.01 * self.calls for x in s.logprobs), s.tokens)
                    for s in super().score(prompts, responses)]

    m = Drifting.demo()
    v = Value(rio(), m, LogProb())
    first = v.noise(repeats=2)
    assert first["std"] > 0 and len(first["values"]) == 3
    assert v.noise(repeats=2) == first and m.calls == 3            # the repeats are cached under their own keys


def test_values_must_be_finite():
    sure = ToyModel(bias=800.0)                                    # every word has probability 1
    with pytest.raises(ValueError, match="not finite"):
        Value(rio(), sure, LogProb(reduce="logodds"))(ALL3)


@pytest.mark.parametrize("masks", [np.ones((2, 2)), np.ones(3), [[1, 2, 0]], [["1", "1", "1"]], np.full((1, 3), 0.5)])
def test_bad_masks_are_refused(masks):
    with pytest.raises(ValueError):
        Value(rio(), ToyModel.demo(), LogProb())(masks)


def test_settings_are_checked():
    p, m = rio(), ToyModel.demo()

    class GenerateOnly:
        name = "gen"

        def generate(self, prompts, sampling):
            return []

    with pytest.raises(WrapperError, match="needs the wrapper's score"):
        Value(p, GenerateOnly(), LogProb())
    with pytest.raises(TypeError):
        Value(p, m, "logprob")
    with pytest.raises(ValueError):
        Value(p, m, LogProb(), function=len)
    with pytest.raises(ValueError):
        Value(p)
    with pytest.raises(ValueError):
        Value(p, function=len, per_response=True)
    with pytest.raises(ValueError):
        Value(p, m, LogProb(), batch_size=0)
    with pytest.raises(TypeError):
        Value(p, m, LogProb(), replace=1)
    with pytest.raises(TypeError):
        Value("not a prompt", m, LogProb())
    assert Value(p, m, LogProb())(np.zeros((0, 3))).shape == (0,)


@given(prompts(), st.data(), st.floats(-3, 3), st.lists(st.floats(-4, 4), min_size=1, max_size=12))
def test_any_prompt_and_mask_give_the_formula(prompt, data, bias, weights):
    """Rules keyed on the units' own texts: the value is the formula applied to the rendered prompt."""
    rules = {}
    for u, w in zip(prompt.units, itertools.cycle(weights)):
        rules[u.text] = {"alpha": w}
    model = ToyModel(rules=rules, bias=bias)
    responded = Prompt(prompt.id, messages=prompt.messages, text=prompt.text, responses=[Response("r", "alpha beta")])
    masks = np.array([data.draw(masks_for(prompt)) for _ in range(3)])
    values = Value(responded, model, LogProb())(masks)
    for mask, value in zip(masks, values):
        rendered = responded.render(mask)
        text = rendered if isinstance(rendered, str) else "\n".join(m["content"] for m in rendered)
        x = bias + sum(w["alpha"] for cue, w in model.rules.items() if cue in text.lower())
        assert value == pytest.approx((ls(x) + ls(bias)) / 2, rel=1e-9, abs=1e-12)
