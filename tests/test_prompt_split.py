"""Splitting (interface §2): each splitter's rules on worked examples, the exact round trip of every splitter on any
text, and a splitter given as a function."""

import re

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from prompt_helpers import SPLIT_TEXT

from contextattr import Prompt, split
from contextattr.prompt import KINDS

LINE_BREAK = re.compile(r"\r\n|\r|\n")


def texts(g):
    return [u.text for u in g.units]


def joined(g):
    return g.lead + "".join(u.text + u.sep for u in g.units)


SENTENCES = [
    ("Rio is hot. It is summer.", ["Rio is hot.", "It is summer."]),
    ("Dr. Smith arrived. He sat down.", ["Dr. Smith arrived.", "He sat down."]),
    ("See Fig. 3 for details. Then stop.", ["See Fig. 3 for details.", "Then stop."]),
    ("As Eq. (2) shows. Done.", ["As Eq. (2) shows.", "Done."]),
    ("Pi is 3.14. That is all.", ["Pi is 3.14.", "That is all."]),
    ("J. K. Rowling wrote it. Many read it.", ["J. K. Rowling wrote it.", "Many read it."]),
    ("The U.S. Army left. Others stayed.", ["The U.S. Army left.", "Others stayed."]),
    ('He said "Stop." Then he left.', ['He said "Stop."', "Then he left."]),
    ("Is it? Yes! Fine.", ["Is it?", "Yes!", "Fine."]),
    ("Wait... what? No.", ["Wait... what?", "No."]),
    ("Wait… What now?", ["Wait…", "What now?"]),
    ("e.g. this one. And that.", ["e.g. this one.", "And that."]),
    ("Smith et al. (2020) agree. We do too.", ["Smith et al. (2020) agree.", "We do too."]),
    ("It costs $5. 3 people paid.", ["It costs $5.", "3 people paid."]),
    ('Stop. "Go!" he said.', ["Stop.", '"Go!" he said.']),
    ("(See below.) Next part.", ["(See below.)", "Next part."]),
    ("Line one\nLine two", ["Line one", "Line two"]),
    ("1. Mix the flour.\n2. Add water.", ["1. Mix the flour.", "2. Add water."]),
    ("a. First option\nb. Second option", ["a. First option", "b. Second option"]),
    ("Pick item b. Then go.", ["Pick item b.", "Then go."]),             # a lower-case letter is not an initial
    ("A b. C d.", ["A b.", "C d."]),
    ("你好。我很好。", ["你好。", "我很好。"]),
    ("No end punctuation", ["No end punctuation"]),
    ("We met in Jan. The rest followed.", ["We met in Jan.", "The rest followed."]),
    ("Due Jan. 5 at noon. Late is fine.", ["Due Jan. 5 at noon.", "Late is fine."]),
    # Known limits of the rules, kept so that any change to them shows: a sentence that ends with an initial or a
    # dotted abbreviation is not cut there.
    ("He got an A. Then he left.", ["He got an A. Then he left."]),
    ("He lives in the U.S. She does not.", ["He lives in the U.S. She does not."]),
]


@pytest.mark.parametrize("text, expected", SENTENCES)
def test_sentence_rules(text, expected):
    g = split(text, "sentence", id="s")
    assert texts(g) == expected
    assert joined(g) == text


def test_sentence_spaces_go_to_lead_and_separators():
    g = split("  One here.  Two there.\t\n", "sentence", id="s")
    assert g.lead == "  "
    assert [(u.id, u.text, u.sep) for u in g.units] == [("s.0", "One here.", "  "), ("s.1", "Two there.", "\t\n")]


def test_words():
    g = split("  a  b\tc\n", "word", id="w")
    assert (g.lead, [(u.text, u.sep) for u in g.units]) == ("  ", [("a", "  "), ("b", "\t"), ("c", "\n")])


def test_lines_keep_their_indentation():
    g = split("  a\n\n  b  \r\nc", "line", id="l")
    assert (g.lead, [(u.text, u.sep) for u in g.units]) == ("", [("  a", "\n\n"), ("  b", "  \r\n"), ("c", "")])
    assert split("\n\n x", "line", id="l").lead == "\n\n"


def test_paragraphs():
    g = split("P1 l1\nP1 l2\n\n  P2\n \nP3", "paragraph", id="p")
    assert [(u.text, u.sep) for u in g.units] == [("P1 l1\nP1 l2", "\n\n"), ("  P2", "\n \n"), ("P3", "")]


def test_split_settings_reach_the_group():
    g = split("A b. C d.", id="doc", sep="\n", tag="doc", replace="[UNK]")
    assert (g.id, g.sep, g.tag, g.replace) == ("doc", "\n", "doc", "[UNK]")
    assert [(u.id, u.group, u.tag, u.replace) for u in g.units] == [
        ("doc.0", "doc", "doc", "[UNK]"), ("doc.1", "doc", "doc", "[UNK]")]


def test_a_function_can_split():
    g = split("abc def", lambda t: [(0, 3), (4, 7)], id="f")
    assert (g.lead, [(u.text, u.sep) for u in g.units]) == ("", [("abc", " "), ("def", "")])
    g = split("xxabcyy", lambda t: iter([(2, 5)]), id="f")
    assert (g.lead, g.units[0].text, g.units[0].sep) == ("xx", "abc", "yy")


@pytest.mark.parametrize("spans, error", [
    ([(0, 3), (2, 5)], ValueError),       # overlapping
    ([(3, 3)], ValueError),               # empty
    ([(4, 2)], ValueError),               # backwards
    ([(0, 99)], ValueError),              # past the end
    ([(2, 4), (0, 1)], ValueError),       # not sorted
    ([(0.0, 2)], TypeError),
    ([(True, 2)], TypeError),
    ([(0, 1, 2)], TypeError),
    ([3], TypeError),
    ([], ValueError),                     # no units
])
def test_bad_spans_are_refused(spans, error):
    with pytest.raises(error):
        split("abcdefg", lambda t: spans, id="f")


@pytest.mark.parametrize("text", ["", "   ", "\n\t\n"])
@pytest.mark.parametrize("by", KINDS)
def test_nothing_to_split(text, by):
    with pytest.raises(ValueError, match="no units"):
        split(text, by, id="x")


def test_bad_arguments():
    with pytest.raises(ValueError):
        split("a b", "words", id="x")
    with pytest.raises(TypeError):
        split(b"a b", "word", id="x")
    with pytest.raises(ValueError):
        split("a b", "word", id="")


# -- properties on any text -------------------------------------------------------------------------------------------

@given(SPLIT_TEXT, st.sampled_from(KINDS))
def test_every_splitter_keeps_the_text_exactly(text, by):
    assume(text.strip())
    g = split(text, by, id="x")
    assert joined(g) == text
    assert Prompt("p", text=[g]).render() == text
    assert Prompt("p", text=[g]).render([0] * len(g.units)) == ""      # all deleted: nothing of the group remains
    assert [u.id for u in g.units] == [f"x.{k}" for k in range(len(g.units))]
    for u in g.units:
        assert u.text.strip(), "a unit always has text"
        assert u.text == u.text.rstrip(), "trailing spaces belong to the separator"
        if by in ("word", "sentence"):
            assert u.text == u.text.lstrip()
        if by in ("sentence", "line"):
            assert not LINE_BREAK.search(u.text), "a line break always ends a sentence or a line"
        if by == "word":
            assert not any(c.isspace() for c in u.text)
        if by == "paragraph":
            assert all(line.strip() for line in LINE_BREAK.split(u.text)), "no blank line inside a paragraph"
    assert not g.lead.strip(), "only spaces come before the first unit"
    assert all(not u.sep.strip() for u in g.units), "only spaces separate units"


@given(SPLIT_TEXT)
def test_coarser_splitters_contain_finer_ones(text):
    """Every sentence lies inside one line and every line inside one paragraph; every word inside one sentence,
    unless full-width punctuation cuts a sentence where no space is."""
    assume(text.strip())
    spans = {by: _spans(split(text, by, id="x"), text) for by in KINDS}
    pairs = [("sentence", "line"), ("line", "paragraph")]
    if not any(c in text for c in "。！？"):
        pairs.append(("word", "sentence"))
    for fine, coarse in pairs:
        for a, b in spans[fine]:
            assert any(c <= a and b <= d for c, d in spans[coarse]), (fine, text[a:b])


@given(SPLIT_TEXT, st.data())
def test_any_valid_spans_round_trip(text, data):
    assume(text)
    cuts = sorted(data.draw(st.sets(st.integers(0, len(text)), min_size=2, max_size=8)))
    spans = [(a, b) for a, b in zip(cuts[::2], cuts[1::2])]
    g = split(text, lambda t: spans, id="f")
    assert joined(g) == text
    assert _spans(g, text) == spans


def _spans(group, text):
    out, pos = [], len(group.lead)
    for u in group.units:
        out.append((pos, pos + len(u.text)))
        pos += len(u.text) + len(u.sep)
    assert pos == len(text)
    return out
