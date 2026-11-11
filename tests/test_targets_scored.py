"""The scored targets (interface §5): the texts each scores, how their numbers combine, reductions and spans."""

import math

import numpy as np
import pytest
from prompt_helpers import rio

from contextattr import Choice, Contrast, LogProb, Pooled, Prompt, Response, Scored, Unit
from contextattr.targets import Item, logodds, reduce_scored


def prompt_with(*responses):
    return Prompt("p", text=[Unit("a", "A")], responses=responses)


def test_reductions():
    s = Scored([-1.0, -2.0, -3.0], ["a", "b", "c"])
    assert reduce_scored(s, Item("r", "abc")) == -2.0
    assert reduce_scored(s, Item("r", "abc", reduce="sum")) == -6.0
    assert reduce_scored(s, Item("r", "abc", reduce="logodds")) == pytest.approx(-6.0 - math.log(1 - math.exp(-6.0)))
    assert logodds(math.log(0.25)) == pytest.approx(math.log(1 / 3))
    assert logodds(0.0) == math.inf


def test_a_span_keeps_the_tokens_that_overlap_it():
    s = Scored([-1.0, -2.0, -4.0, -8.0], ["It", " is", " cold", "."])     # characters 0-2, 2-5, 5-10, 10-11
    text = "It is cold."
    assert reduce_scored(s, Item("r", text, span=(6, 10), reduce="sum")) == -4.0
    assert reduce_scored(s, Item("r", text, span=(4, 6), reduce="sum")) == -6.0      # touches " is" and " cold"
    assert reduce_scored(s, Item("r", text, span=(10, 11), reduce="sum")) == -8.0
    assert reduce_scored(s, Item("r", text, span=(0, 11))) == -15.0 / 4


def test_a_span_needs_token_texts_that_join_back():
    with pytest.raises(ValueError, match="needs token texts"):
        reduce_scored(Scored([-1.0]), Item("r", "x", span=(0, 1)))
    with pytest.raises(ValueError, match="join back"):
        reduce_scored(Scored([-1.0], ["y"]), Item("r", "x", span=(0, 1)))
    with pytest.raises(ValueError, match="no tokens"):
        reduce_scored(Scored([]), Item("r", "x"))


def test_logprob_finds_its_response():
    p = rio()
    assert LogProb().items(p) == [Item("bad-1", "It is cold, because Carnival is in winter.", "negative")]
    assert LogProb("bad-1").items(p) == LogProb().items(p)
    own = Response("mine", "Hot.", span=(0, 3))
    assert LogProb(own, reduce="sum").items(p) == [Item("mine", "Hot.", None, (0, 3), "sum")]
    with pytest.raises(ValueError, match="no response 'nope'"):
        LogProb("nope").items(p)
    with pytest.raises(ValueError, match="2 responses"):
        LogProb().items(prompt_with(Response("a", "x"), Response("b", "y")))


def test_pooled_and_contrast_combine_by_label():
    p = prompt_with(Response("n1", "x", "negative"), Response("p1", "y", "positive"),
                    Response("n2", "z", "negative"), Response("u", "w"))
    items = Contrast().items(p)
    assert [i.id for i in items] == ["n1", "n2", "p1"]
    assert Contrast().combine(np.array([[-1.0, -4.0, -3.0]]), items) == pytest.approx([(-1.0 - 4.0) / 2 + 3.0])
    pooled = Pooled("negative").items(p)
    assert [i.id for i in pooled] == ["n1", "n2"]
    assert Pooled().combine(np.array([[-1.0, -4.0]]), pooled) == pytest.approx([-2.5])
    with pytest.raises(ValueError, match="no positive response"):
        Contrast().items(prompt_with(Response("n", "x", "negative")))
    with pytest.raises(ValueError):
        Pooled("neutral")


def test_choice():
    items = Choice("True", "False").items(rio())
    assert [(i.text, i.reduce) for i in items] == [("True", "sum"), ("False", "sum")]
    assert Choice("True", "False").combine(np.array([[-0.1, -2.3]]), items) == pytest.approx([2.2])
    for a, b in [("", "x"), ("x", "x"), ("x", None)]:
        with pytest.raises(ValueError):
            Choice(a, b)


def test_reduce_is_checked():
    for make in (lambda: LogProb(reduce="max"), lambda: Pooled(reduce="avg"), lambda: Contrast(reduce="")):
        with pytest.raises(ValueError):
            make()
