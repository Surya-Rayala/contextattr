"""explain and attribute end to end (interface §1, §8): the toy model's known answers through every method, R² on
shared held-out masks, the requests each step costs, the noise, and the result's views."""

import json

import numpy as np
import pytest
from prompt_helpers import rio

import contextattr as ca
from contextattr.evaluate import faithfulness, held_out_masks
from contextattr.wrapper.toy import log_sigmoid as ls

D3 = 2 * (ls(1.0) - ls(-1.0)) / 8      # the bad response: "cold" and "winter" rise by 2 when d3 is kept, 8 words


def rio2():
    p = rio()
    good = ca.Response("good-1", "It is hot and humid: it is summer in Brazil.", label="positive")
    return ca.Prompt(p.id, messages=p.messages, responses=(*p.responses, good))


@pytest.mark.parametrize("method", ["loo", "banzhaf", "kernelshap"])
def test_the_toy_model_s_known_answer(method):
    e = ca.explain(rio(), ca.ToyModel.demo(), ca.LogProb(), method=method, budget=8)
    assert e.scores == pytest.approx({"d1": 0.0, "d2": 0.0, "d3": D3}, abs=1e-12)
    assert e.r2 == pytest.approx(1.0) and e.top(1) == [("d3", pytest.approx(D3))]


def test_contrast_shows_what_tips_the_answer():
    e = ca.explain(rio2(), ca.ToyModel.demo(), ca.Contrast(), method="kernelshap", budget=8)
    s = e.scores
    assert s["d2"] > 0 > s["d3"] and s["d1"] > 0       # summer (with d1) pushes to the good answer, winter to the bad
    assert s["d1"] < s["d2"]                            # d1 matters only together with d2


def test_requests_and_the_shared_test_masks():
    cache = ca.Cache()
    m = ca.ToyModel.demo()
    first = ca.explain(rio(), m, ca.LogProb(), method="contextcite", budget=6, test_masks=20, cache=cache)
    assert first.masks == 6 and first.test_masks == 20
    # three units have only 8 distinct masks: the 6 + 20 masks and the 3 noise repeats ask at most 8 + 3 requests
    assert first.requests["requests"] <= 11
    second = ca.explain(rio(), m, ca.LogProb(), method="banzhaf", budget=8, test_masks=20, cache=cache)
    assert second.requests["requests"] == 0                # every mask and the noise repeats came from the cache
    assert (held_out_masks(3, 20, 0) == held_out_masks(3, 20, 0)).all()
    assert not (held_out_masks(12, 20, 0) == held_out_masks(12, 20, 1)).all()


def test_noise_is_measured_and_can_be_skipped():
    e = ca.explain(rio(), ca.ToyModel.demo(), method="loo")
    assert e.noise["std"] == 0.0 and len(e.noise["values"]) == 4
    assert ca.explain(rio(), ca.ToyModel.demo(), method="loo", noise_repeats=0).noise is None


def test_a_value_function_and_attribute():
    count = lambda ps: [p[1]["content"].count("Brazil") for p in ps]        # noqa: E731
    e = ca.explain(rio(), value=count, name="brazil", method="kernelshap", budget=8)
    assert e.scores == pytest.approx({"d1": 0.0, "d2": 1.0, "d3": 1.0})
    a = ca.attribute(lambda m: 3 * m[:, 0] - m[:, 1], 2, method="kernelshap", budget=4, unit_ids=["x", "y"])
    assert a.scores == pytest.approx({"x": 3.0, "y": -1.0}) and a.prompt_id is None


def test_views_of_an_explanation():
    e = ca.explain(rio(), ca.ToyModel.demo(), method="kernelshap", budget=8)
    d = e.as_dict()
    assert json.loads(json.dumps(d))["units"] == ["d1", "d2", "d3"] and d["interactions"] == []
    assert d["target"].startswith("LogProb") and d["masks"] == 8
    text = e.table()
    assert text.splitlines()[2].strip().startswith("d3") and "Winter in Brazil" in text and "R² 1" in text
    assert text.splitlines()[1].strip().startswith("above 0: the unit")
    assert e.surrogate.predict(np.ones((1, 3)))[0] == pytest.approx(e.surrogate.intercept + D3)


def test_faithfulness():
    v = np.array([1.0, 2.0, 3.0])
    assert faithfulness(v, v) == 1.0 and faithfulness(np.full(3, 2.0), v) == 0.0
    assert faithfulness(v[::-1], v) < 0 and np.isnan(faithfulness(v, np.ones(3)))


def test_explain_checks_its_inputs():
    with pytest.raises(ValueError):
        ca.explain(rio())
    with pytest.raises(ValueError):
        ca.explain(rio(), ca.ToyModel.demo(), value=len)
    with pytest.raises(ValueError):
        ca.explain(rio(), ca.ToyModel.demo(), method="nope")
    with pytest.raises(ValueError):
        ca.attribute(lambda m: m[:, 0], 0)
    with pytest.raises(ValueError):
        ca.attribute(lambda m: np.ones(2), 3, method="loo")
    with pytest.raises(ValueError):
        ca.attribute(lambda m: m[:, 0], 2, unit_ids=["only one"])


def test_the_second_line_says_how_to_read_the_sign():
    """Every target: a score above 0 pushes towards what is explained, and the printed result says what that is."""
    from contextattr.result import sign_meaning

    said = {"LogProb(response=None, reduce='mean')": "makes the answer more likely",
            "Pooled(label='negative', reduce='mean')": "makes the negative answers more likely",
            "Pooled(label='positive', reduce='mean')": "makes the positive answers more likely",
            "Contrast(reduce='mean')": "pushes towards the positive (wanted) answers",
            "Choice(a='450', b='310')": "pushes towards '450' · below 0: towards '310'",
            "a function": "raises the value", None: "raises the value"}
    for target, words in said.items():
        assert sign_meaning(target).startswith("above 0: the unit ") and words in sign_meaning(target), target
    for target in (ca.LogProb(), ca.Pooled("negative"), ca.Contrast(), ca.Choice("450", "310")):
        assert not sign_meaning(repr(target)).endswith("lowers it"), repr(target)       # each has its own line
    model = ca.ToyModel.demo()
    e = ca.explain(rio(), model, ca.Sampled(ca.checks.regex(r"cold"), samples=2), method="loo", test_masks=0,
                   noise_repeats=0)
    assert "more replies get flagged by the check" in e.table().splitlines()[1]
    assert ca.Explanation.from_dict(e.as_dict()).table().splitlines()[1] == e.table().splitlines()[1]


def test_the_table_is_the_same_on_every_machine():
    """Scores that print alike keep the prompt's order, and a zero never prints as -0.0000: rounding noise of the
    last bits, which differs between machines, must not change what is printed."""
    def f(masks):
        return 2.0 * np.asarray(masks)[:, 1] + 1e-17 * np.asarray(masks)[:, 0] - 1e-17 * np.asarray(masks)[:, 2]

    lines = ca.attribute(f, 4, method="banzhaf", budget=16, unit_ids=["a", "b", "c", "d"]).table().splitlines()
    assert [line.split()[0] for line in lines[2:6]] == ["b", "a", "c", "d"]
    assert all("-0.0000" not in line for line in lines)
