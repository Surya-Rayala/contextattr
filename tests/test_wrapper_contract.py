"""The wrapper contract (docs/wrapper.md): Sampling and Scored, and the checks on every answer a wrapper gives."""

import math

import numpy as np
import pytest

from contextattr import Sampling, Scored, ToyModel, WrapperError
from contextattr.wrapper import (Generator, Scorer, call_generate, call_score, can_generate, can_score,
                                 model_name)


class Fake:
    """A wrapper made of the given functions."""

    def __init__(self, score=None, generate=None, name="fake"):
        self.name = name
        if score:
            self.score = score
        if generate:
            self.generate = generate


def test_sampling_defaults_and_normalisation():
    s = Sampling()
    assert (s.n, s.temperature, s.top_p, s.max_tokens, s.seed, s.stop) == (1, 0.0, 1.0, 256, 0, ())
    assert Sampling(stop=["\n", "END"]).stop == ("\n", "END")
    assert Sampling(stop="\n").stop == ("\n",)
    assert Sampling(seed=None).seed is None
    assert Sampling(n=3, temperature=0.7).as_dict() == {"n": 3, "temperature": 0.7, "top_p": 1.0,
                                                        "max_tokens": 256, "seed": 0, "stop": []}


@pytest.mark.parametrize("kwargs", [{"n": 0}, {"n": 1.0}, {"n": True}, {"temperature": -0.1},
                                    {"temperature": math.inf}, {"temperature": "0"}, {"top_p": 0},
                                    {"top_p": 1.5}, {"max_tokens": 0}, {"seed": 1.5}, {"stop": [""]},
                                    {"stop": [3]}])
def test_sampling_refuses(kwargs):
    with pytest.raises(ValueError):
        Sampling(**kwargs)


def test_scored_accepts_sequences_and_arrays():
    s = Scored(np.array([-0.5, -1.0], dtype=np.float32), ["a", " b"])
    assert s.logprobs == (-0.5, -1.0) and s.tokens == ("a", " b")
    assert Scored([0.0]).logprobs == (0.0,)
    assert Scored([5e-4]).logprobs == (5e-4,)        # rounding a hair above 0 is tolerated
    assert Scored([]).logprobs == ()


@pytest.mark.parametrize("logprobs, tokens, error", [
    ([0.5], None, ValueError), ([math.nan], None, ValueError), ([-math.inf], None, ValueError),
    (["x"], None, TypeError), ("abc", None, TypeError), (5, None, TypeError),
    ([-1.0], ["a", "b"], ValueError), ([-1.0], [1], TypeError), ([-1.0], "a", TypeError),
])
def test_scored_refuses(logprobs, tokens, error):
    with pytest.raises(error):
        Scored(logprobs, tokens)


def test_capabilities_and_name():
    both = Fake(score=lambda p, r: [], generate=lambda p, s: [])
    assert can_score(both) and can_generate(both)
    only = Fake(generate=lambda p, s: [])
    assert not can_score(only) and can_generate(only)
    assert isinstance(ToyModel(), Scorer) and isinstance(ToyModel(), Generator)
    assert model_name(ToyModel(name="x")) == "x"
    for bad in (object(), Fake(name=""), Fake(name=3)):
        with pytest.raises(WrapperError):
            model_name(bad)


@pytest.mark.parametrize("answer, match", [
    ("text", "not a list"), ({"a": 1}, "not a list"), (None, "not a list"), (iter([]), "not a list"),
    ([Scored([-1.0])], "1 answers for 2 requests"),
    ([Scored([-1.0]), [-1.0]], "answer 1 is list, not Scored"),
])
def test_call_score_refuses_bad_answers(answer, match):
    with pytest.raises(WrapperError, match=match):
        call_score(Fake(score=lambda p, r: answer), ["a", "b"], ["x", "y"])


def test_call_score_takes_a_tuple():
    m = Fake(score=lambda p, r: tuple(Scored([-1.0]) for _ in p))
    assert call_score(m, ["a", "b"], ["x", "y"]) == [Scored([-1.0]), Scored([-1.0])]


@pytest.mark.parametrize("answer, match", [
    ("text", "not a list"), ([["x"]], "replies for 1 prompts, not 2"), ([["x"], "y"], "answer 1 is str"),
    ([["x"], ["y", "z"]], "must hold 1 strings"), ([["x"], [3]], "must hold 1 strings"),
])
def test_call_generate_refuses_bad_answers(answer, match):
    with pytest.raises(WrapperError, match=match):
        call_generate(Fake(generate=lambda p, s: answer), ["a", "b"], Sampling())


def test_call_generate_passes_good_answers():
    m = Fake(generate=lambda p, s: [("r",) * s.n for _ in p])
    assert call_generate(m, ["a", "b"], Sampling(n=2)) == [["r", "r"], ["r", "r"]]
