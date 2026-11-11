"""Rendering a mask (interface §2, §6): exact text, deletion with the separator, replacement, groups, and a
reference renderer written another way that every random prompt and mask must agree with."""

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from prompt_helpers import masks_for, prompts, rio

from contextattr import Group, Prompt, Unit
from contextattr.prompt import as_mask, render

QUESTION = "Question: What is the weather like during Rio Carnival?"
D1 = "Rio Carnival takes place before Lent, in February or March.\n"
D2 = "Summer in Brazil runs from December to March.\n"
D3 = "Winter in Brazil runs from June to September.\n"


def test_interface_example_renders_exactly():
    p = rio()
    full = p.render()
    assert full == [{"role": "system", "content": "Answer in one sentence."},
                    {"role": "user", "content": "Documents:\n" + D1 + D2 + D3 + QUESTION}]
    assert p.render(p.mask(drop="d3"))[1]["content"] == "Documents:\n" + D1 + D2 + QUESTION
    assert p.render([0, 0, 0])[1]["content"] == "Documents:\n" + QUESTION
    assert p.render([0, 1, 0], replace="[UNK]")[1]["content"] == ("Documents:\n[UNK]\n" + D2 + "[UNK]\n" + QUESTION)
    assert p.render([1, 1, 1], replace="[UNK]") == full


def test_text_prompt_gives_a_string():
    p = Prompt("t", text=["Review: ", Unit("w1", "great", sep=" "), Unit("w2", "film")])
    assert p.render() == "Review: great film"
    assert p.render([0, 1]) == "Review: film"
    assert p.render([1, 0]) == "Review: great "      # a kept unit keeps its own separator
    assert p.render([0, 0], replace="_") == "Review: _ _"


def test_a_unit_s_own_replacement_overrides_the_run_setting():
    p = Prompt("t", text=[Unit("a", "A", sep=" ", replace="<a>"), Unit("b", "B", sep=" "), "end"])
    assert p.render([0, 0]) == "<a> end"                 # delete for b, its own text for a
    assert p.render([0, 0], replace="?") == "<a> ? end"
    assert p.render([0, 0], replace="") == "<a>  end"   # an empty replacement keeps the separator


def test_group_lead_and_sep_stay_while_any_unit_shows():
    g = Group("d", (Unit("d.0", "One.", sep=" "), Unit("d.1", "Two.", sep="")), lead="> ", sep="\n")
    p = Prompt("t", text=["Docs:\n", g, "Q?"])
    assert p.render() == "Docs:\n> One. Two.\nQ?"
    assert p.render([1, 0]) == "Docs:\n> One. \nQ?"
    assert p.render([0, 1]) == "Docs:\n> Two.\nQ?"
    assert p.render([0, 0]) == "Docs:\nQ?"               # the last unit deleted takes the lead and sep with it
    assert p.render([0, 0], replace="[UNK]") == "Docs:\n> [UNK] [UNK]\nQ?"
    g2 = Group("d", (Unit("d.0", "One.", sep=" ", replace="…"), Unit("d.1", "Two.")), lead="> ", sep="\n")
    assert Prompt("t", text=[g2]).render([0, 0]) == "> … \n"   # a unit replaced by its own text still shows


def test_an_emptied_message_stays_in_place():
    p = Prompt("p", messages=[{"role": "system", "content": [Unit("s", "Be brief.")]},
                              {"role": "user", "content": "Hi"}])
    assert p.render([0]) == [{"role": "system", "content": ""}, {"role": "user", "content": "Hi"}]


@pytest.mark.parametrize("mask, error", [
    ([1, 1], ValueError),               # too short
    ([1, 1, 1, 1], ValueError),         # too long
    ([1, 2, 0], ValueError),
    ([1, 0.5, 0], ValueError),
    (["1", 1, 1], ValueError),
    ([None, 1, 1], ValueError),
    ("111", TypeError),
    (7, TypeError),
    ([[1, 1, 1], [1, 1, 1], [1, 1, 1]], ValueError),   # a matrix of the right length is not a mask
])
def test_bad_masks_are_refused(mask, error):
    with pytest.raises(error):
        rio().render(mask)


def test_mask_values():
    assert as_mask([1, 0, True, False, 1.0], 5) == (True, False, True, False, True)
    assert as_mask(None, 3) == (True, True, True)
    assert as_mask((x for x in [0, 1]), 2) == (False, True)
    with pytest.raises(TypeError):
        render(rio(), None, replace=5)


# -- every random prompt and mask against a reference renderer -------------------------------------------------------

def reference_render(prompt, mask, replace):
    """Flatten the prompt into segments, each shown or not by a rule on the mask, then join: a different route to the
    same specification (interface §2) than render.py's single pass."""
    segments, k = [], 0

    def unit(u):
        nonlocal k
        removed = u.replace if u.replace is not None else replace
        segments.append(("unit", k, u.text + u.sep, None if removed is None else removed + u.sep))
        k += 1

    def flatten(pieces):
        for p in pieces:
            if isinstance(p, str):
                segments.append(("fixed", p))
            elif isinstance(p, Unit):
                unit(p)
            else:
                first = k
                lead = len(segments)
                segments.append(None)
                for u in p.units:
                    unit(u)
                members = [s for s in segments[lead + 1:]]
                segments[lead] = ("edge", p.lead, members)
                segments.append(("edge", p.sep, members))
                assert k - first == len(p.units)

    def show(seg):
        if seg[0] == "fixed":
            return seg[1]
        if seg[0] == "unit":
            return seg[2] if mask[seg[1]] else (seg[3] or "")
        visible = any(mask[i] or alt is not None for _, i, _, alt in seg[2])
        return seg[1] if visible else ""

    out = []
    for content in ([m.content for m in prompt.messages] if prompt.is_chat else [prompt.text]):
        start = len(segments)
        flatten(content)
        out.append("".join(show(s) for s in segments[start:]))
    assert k == prompt.n
    if prompt.is_chat:
        return [{"role": m.role, "content": c} for m, c in zip(prompt.messages, out)]
    return out[0]


def original(prompt):
    """The prompt as written: every piece's full text, by its fields alone."""
    def full(pieces):
        return "".join(p if isinstance(p, str) else p.text + p.sep if isinstance(p, Unit) else
                       p.lead + "".join(u.text + u.sep for u in p.units) + p.sep for p in pieces)
    if prompt.is_chat:
        return [{"role": m.role, "content": full(m.content)} for m in prompt.messages]
    return full(prompt.text)


@given(prompts(), st.data(), st.none() | st.text(max_size=3))
def test_render_agrees_with_the_reference(prompt, data, replace):
    mask = data.draw(masks_for(prompt))
    assert prompt.render(mask, replace) == reference_render(prompt, mask, replace)


@given(prompts(), st.none() | st.text(max_size=3))
def test_keeping_every_unit_gives_the_prompt_as_written(prompt, replace):
    assert prompt.render() == original(prompt)
    assert prompt.render([1] * prompt.n, replace) == original(prompt)


@given(prompts(), st.data())
def test_removing_one_unit_removes_exactly_its_text(prompt, data):
    """From the full prompt, deleting unit k takes away its text and separator, and its group's lead and sep when
    nothing else of the group shows; nothing else changes."""
    k = data.draw(st.integers(0, prompt.n - 1))
    u = prompt.units[k]
    assume(u.replace is None)
    mask = [True] * prompt.n
    mask[k] = False
    full, cut = prompt.render(), prompt.render(mask)
    if prompt.is_chat:
        full = "\x00".join(m["content"] for m in full)
        cut = "\x00".join(m["content"] for m in cut)
    removed = len(u.text) + len(u.sep)
    if u.group is not None and len(prompt.groups[u.group]) == 1:
        g = next(p for p in _pieces(prompt) if getattr(p, "id", None) == u.group)
        removed += len(g.lead) + len(g.sep)
    assert len(full) - len(cut) == removed


@given(prompts())
def test_roles_and_message_count_never_change(prompt):
    if prompt.is_chat:
        for mask in ([0] * prompt.n, [1] * prompt.n):
            assert [m["role"] for m in prompt.render(mask)] == [m.role for m in prompt.messages]


def _pieces(prompt):
    return [p for m in prompt.messages for p in m.content] if prompt.is_chat else list(prompt.text)
