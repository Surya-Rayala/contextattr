"""Prompt, Message, Unit, Group and Response: construction, normalisation and every check (interface §2)."""

import pytest
from prompt_helpers import rio

from contextattr import Group, Message, Prompt, Response, Unit


def test_dict_messages_and_responses_are_normalised():
    p = rio()
    assert p.is_chat and p.text is None
    assert all(isinstance(m, Message) for m in p.messages)
    assert p.messages[0].content == ("Answer in one sentence.",)
    assert p.responses == (Response("bad-1", "It is cold, because Carnival is in winter.", label="negative"),)
    assert p.n == 3 and p.unit_ids == ("d1", "d2", "d3")
    assert [u.tag for u in p.units] == ["doc"] * 3
    assert p.index("d2") == 1 and dict(p.groups) == {}


def test_equal_prompts_are_equal_and_hashable():
    assert rio() == rio() and hash(rio()) == hash(rio())
    assert len({rio(), rio()}) == 1


def test_units_are_in_order_across_messages_and_groups():
    g = Group("g", (Unit("a", "A"), Unit("b", "B")), tag="t")
    p = Prompt("p", messages=[{"role": "user", "content": [Unit("x", "X"), g]},
                              {"role": "assistant", "content": ["fixed", Unit("y", "Y")]}])
    assert p.unit_ids == ("x", "a", "b", "y")
    assert dict(p.groups) == {"g": (1, 2)}
    assert [u.group for u in p.units] == [None, "g", "g", None]
    assert [u.tag for u in p.units] == [None, "t", "t", None]


def test_text_prompt():
    p = Prompt("t", text=["Review: ", Unit("w1", "great", sep=" "), Unit("w2", "film")])
    assert not p.is_chat and p.messages is None and p.n == 2
    assert Prompt("t", text=[Unit("only", "x")]).n == 1


@pytest.mark.parametrize("kwargs, error", [
    ({}, ValueError),                                                            # neither messages nor text
    ({"messages": [{"role": "user", "content": [Unit("a", "A")]}], "text": [Unit("b", "B")]}, ValueError),  # both
    ({"messages": []}, ValueError),
    ({"messages": "hello"}, TypeError),
    ({"messages": [{"role": "user", "content": "no units"}]}, ValueError),
    ({"messages": [{"role": "user", "content": [Unit("a", "A")], "name": "x"}]}, ValueError),   # unknown field
    ({"messages": [{"role": "", "content": [Unit("a", "A")]}]}, ValueError),     # empty role
    ({"messages": [("user", "content")]}, TypeError),
    ({"text": [Unit("a", "A"), 5]}, TypeError),                                  # not a piece
    ({"text": "only fixed text"}, ValueError),                                   # no units
])
def test_prompt_shape_checks(kwargs, error):
    with pytest.raises(error):
        Prompt("p", **kwargs)


@pytest.mark.parametrize("content", [
    [Unit("a", "A"), Unit("a", "B")],                                            # two units
    [Group("a", (Unit("b", "B"),)), Unit("a", "A")],                             # a group and a unit
    [Group("g", (Unit("a", "A"),)), Unit("a", "B")],                             # a group's unit and a unit
    [Group("g", (Unit("a", "A"),)), Group("g", (Unit("b", "B"),))],              # two groups
    [Group("g", (Unit("g", "A"),))],                                             # a group and its own unit
])
def test_ids_must_be_unique(content):
    with pytest.raises(ValueError, match="used twice"):
        Prompt("p", text=content)


def test_ids_must_be_unique_across_messages():
    with pytest.raises(ValueError, match="used twice"):
        Prompt("p", messages=[{"role": "user", "content": [Unit("a", "A")]},
                              {"role": "user", "content": [Unit("a", "B")]}])


@pytest.mark.parametrize("make, error", [
    (lambda: Unit("", "A"), ValueError),
    (lambda: Unit("a", ""), ValueError),
    (lambda: Unit("a", None), TypeError),
    (lambda: Unit(1, "A"), TypeError),
    (lambda: Unit("a", "A", sep=None), TypeError),
    (lambda: Unit("a", "A", tag=3), TypeError),
    (lambda: Unit("a", "A", replace=b"x"), TypeError),
    (lambda: Group("g", ()), ValueError),
    (lambda: Group("g", "AB"), TypeError),
    (lambda: Group("g", ("A",)), TypeError),
    (lambda: Group("g", (Unit("a", "A", group="other"),)), ValueError),
    (lambda: Group("", (Unit("a", "A"),)), ValueError),
    (lambda: Group("g", (Unit("a", "A"),), lead=None), TypeError),
])
def test_unit_and_group_checks(make, error):
    with pytest.raises(error):
        make()


def test_group_gives_its_tag_and_replace_to_units_without_their_own():
    g = Group("g", (Unit("a", "A"), Unit("b", "B", tag="own", replace="R")), tag="t", replace="[X]")
    assert [(u.tag, u.replace, u.group) for u in g.units] == [("t", "[X]", "g"), ("own", "R", "g")]
    assert Group("g", g.units, tag="t", replace="[X]") == g      # building it again changes nothing


def test_a_unit_naming_a_group_must_be_inside_it():
    with pytest.raises(ValueError, match="not inside it"):
        Prompt("p", text=[Unit("a", "A", group="g")])


@pytest.mark.parametrize("make, error", [
    (lambda: Response("", "x"), ValueError),
    (lambda: Response("r", ""), ValueError),
    (lambda: Response("r", "x", label="bad"), ValueError),
    (lambda: Response("r", "abc", span=(0, 4)), ValueError),
    (lambda: Response("r", "abc", span=(2, 2)), ValueError),
    (lambda: Response("r", "abc", span=(-1, 2)), ValueError),
    (lambda: Response("r", "abc", span=(0, 1, 2)), TypeError),
    (lambda: Response("r", "abc", span=(True, 2)), TypeError),
    (lambda: Response("r", "abc", span=(0.0, 2)), TypeError),
    (lambda: Response("r", "abc", span="ab"), TypeError),
])
def test_response_checks(make, error):
    with pytest.raises(error):
        make()


def test_response_span_and_ids():
    assert Response("r", "abc", span=[1, 3]).span == (1, 3)
    with pytest.raises(ValueError, match="unique"):
        Prompt("p", text=[Unit("a", "A")], responses=[Response("r", "x"), Response("r", "y")])
    with pytest.raises(TypeError):
        Prompt("p", text=[Unit("a", "A")], responses=["just text"])


def test_mask_from_ids():
    p = Prompt("p", text=[Unit("x", "X"), Group("g", (Unit("a", "A"), Unit("b", "B"))), Unit("y", "Y")])
    assert p.mask(keep=["x"]) == (True, False, False, False)
    assert p.mask(keep="g") == (False, True, True, False)
    assert p.mask(drop=["g", "y"]) == (True, False, False, False)
    assert p.mask(drop=[]) == (True,) * 4
    assert p.mask(keep=[]) == (False,) * 4
    with pytest.raises(KeyError):
        p.mask(drop=["nope"])
    with pytest.raises(ValueError):
        p.mask()
    with pytest.raises(ValueError):
        p.mask(keep=["x"], drop=["y"])
    with pytest.raises(KeyError):
        p.index("nope")
