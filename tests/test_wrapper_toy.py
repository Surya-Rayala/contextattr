"""ToyModel: its answers against its formula worked by hand, its token texts, its replies and its checks."""

import math

import pytest

from contextattr import Sampling, ToyModel
from contextattr.wrapper.toy import log_sigmoid


def ls(x):
    """log σ(x), written directly (fine for moderate x)."""
    return -math.log(1 + math.exp(-x))


def test_log_sigmoid_is_exact_and_stable():
    for x in (-5.0, -1.0, 0.0, 0.3, 4.0):
        assert log_sigmoid(x) == pytest.approx(ls(x), rel=1e-12)
    assert log_sigmoid(800.0) == 0.0 and log_sigmoid(-800.0) == -800.0


def test_scores_follow_the_rules_word_by_word():
    m = ToyModel(rules={"Winter": {"cold": 2.0}, "summer": {"hot": 1.5, "cold": -1.0}},
                 pairs={("winter", "summer"): {"cold": 0.5}}, bias=-1.0)
    (s,) = m.score(["Docs: winter and SUMMER."], ["It is cold, hot!"])
    assert s.tokens == ("It", " is", " cold,", " hot!")
    assert s.logprobs == pytest.approx([ls(-1.0), ls(-1.0), ls(-1.0 + 2.0 - 1.0 + 0.5), ls(-1.0 + 1.5)], rel=1e-12)
    (s,) = m.score(["only winter"], ["COLD"])
    assert s.logprobs == pytest.approx([ls(1.0)], rel=1e-12)


def test_chat_prompts_join_their_messages():
    m = ToyModel(rules={"winter": {"cold": 3.0}})
    chat = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "It is winter"}]
    assert m.score([chat], ["cold"])[0].logprobs == m.score(["be brief\nIt is winter"], ["cold"])[0].logprobs
    apart = [{"role": "user", "content": "win"}, {"role": "user", "content": "ter"}]
    assert m.score([apart], ["cold"])[0].logprobs == pytest.approx([ls(0.0)])   # a cue cut by a message is absent


def test_tokens_join_back_exactly():
    m = ToyModel()
    for r in ["a", "  lead", "trail  ", " both ", "x\ny\tz", " "]:
        (s,) = m.score(["p"], [r])
        assert "".join(s.tokens) == r and len(s.tokens) == len(s.logprobs)
    assert m.score(["p"], [""])[0].logprobs == ()


def test_generate_replies_by_the_first_cue_present():
    m = ToyModel(replies={"summer": "hot", "winter": "cold"}, default_reply="?")
    assert m.generate(["Summer and winter", "winter", "spring"], Sampling(n=2)) == [
        ["hot", "hot"], ["cold", "cold"], ["?", "?"]]


def test_the_demo_tells_the_rio_story():
    m = ToyModel.demo()
    winter, summer = "Winter in Brazil runs from June to September.", "Summer in Brazil runs from December to March."
    cold = lambda text: m.score([text], ["cold"])[0].logprobs[0]
    assert cold(winter) > cold(summer)
    assert m.generate([winter], Sampling())[0][0] == "It is cold, because Carnival is in winter."
    assert m.generate([winter + summer], Sampling())[0][0] == "It is hot and humid: it is summer in Brazil."


@pytest.mark.parametrize("kwargs", [{"rules": {"": {"a": 1}}}, {"rules": {"c": {"a": "x"}}}, {"rules": {"c": [1]}},
                                    {"rules": {"c": {"a": math.nan}}}, {"rules": {"c": {"a": True}}},
                                    {"pairs": {"ab": {"a": 1}}}, {"bias": "x"}, {"bias": math.inf},
                                    {"replies": {"c": 3}}])
def test_bad_settings_are_refused(kwargs):
    with pytest.raises(TypeError):
        ToyModel(**kwargs)


def test_bad_inputs_are_refused():
    m = ToyModel()
    with pytest.raises(ValueError):
        m.score(["a"], ["x", "y"])
    for prompt in ({"role": "user", "content": "x"}, [{"role": "user", "content": 3}], [{"content": "x"}], 5):
        with pytest.raises(TypeError):
            m.score([prompt], ["x"])
    with pytest.raises(TypeError):
        m.score(["a"], [3])
