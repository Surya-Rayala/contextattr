"""Cutting a text into units that join back to it exactly (docs/interface.md §2).

Every splitter only chooses where units start and end: the text before the first unit becomes the group's `lead`,
and the text after each unit, up to the next one, becomes that unit's `sep`, so nothing is ever lost or added.
"""

from __future__ import annotations

import operator
import re
from collections.abc import Callable, Iterable

from .units import Group, Unit

KINDS = ("word", "sentence", "line", "paragraph")

_LINE_BREAK = re.compile(r"\r\n|\r|\n")
_WORD = re.compile(r"\S+")
_CLOSERS = "\"'”’»)]"
_OPENERS = "\"'“‘«([{"
_LATIN_END = re.compile(r"[.!?…]+[\"'”’»)\]]*(?=\s)")  # terminal punctuation, closing quotes or brackets, a space
_CJK_END = re.compile(r"[。！？]+[」』”’）)]*")  # full-width terminals need no space after them
_NUMBER_NEXT = re.compile(r"\(?\d")
_DOTTED = re.compile(r"(?:[^\W\d_]\.)+[^\W\d_]")  # U.S, e.g, i.e, a.m: what precedes the final period
# A period after these never ends a sentence: titles and short forms that precede a name or a word.
_ABBREV = frozenset("mr mrs ms mx dr prof st mt gen gov sen rep rev lt col sgt capt cmdr adm maj vs cf al".split())
# A period after these does not end a sentence when a number follows: "Fig. 3", "No. 5", "Eq. (2)", "Jan. 5".
_ABBREV_NUMBER = frozenset("no nos fig figs eq eqs sec secs vol vols pp p ch art ref tab approx "
                           "jan feb mar apr jun jul aug sep sept oct nov dec".split())


def split(text: str, by: str | Callable[[str], Iterable[tuple[int, int]]] = "sentence", *, id: str,
          sep: str = "", tag: str | None = None, replace: str | None = None) -> Group:
    """Cut `text` into units `<id>.0`, `<id>.1`, … held in a Group with `sep` after it. The units with their
    separators give back `text` exactly. `by`: "word", "sentence", "line", "paragraph", or a function returning the
    units' (start, end) character spans, sorted and not overlapping."""
    if not isinstance(text, str):
        raise TypeError(f"split: text must be a string, not {type(text).__name__}")
    if callable(by):
        spans = _custom_spans(text, by)
    elif by in _SPLITTERS:
        spans = _SPLITTERS[by](text)
    else:
        raise ValueError(f"split: by must be one of {', '.join(KINDS)} or a function, not {by!r}")
    if not spans:
        raise ValueError(f"split: no units in {text[:40]!r} (by {by if isinstance(by, str) else 'the function'})")
    units = []
    for k, (start, end) in enumerate(spans):
        nxt = spans[k + 1][0] if k + 1 < len(spans) else len(text)
        units.append(Unit(f"{id}.{k}", text[start:end], sep=text[end:nxt]))
    return Group(id, tuple(units), lead=text[: spans[0][0]], sep=sep, tag=tag, replace=replace)


def _word_spans(text: str) -> list[tuple[int, int]]:
    """Runs of non-space characters."""
    return [(m.start(), m.end()) for m in _WORD.finditer(text)]


def _line_spans(text: str) -> list[tuple[int, int]]:
    """Lines with text, each from the line's start (its indentation included) to its last non-space character."""
    spans = []
    start = 0
    for m in [*_LINE_BREAK.finditer(text), None]:
        end = m.start() if m else len(text)
        content = text[start:end].rstrip()
        if content.strip():
            spans.append((start, start + len(content)))
        if m:
            start = m.end()
    return spans


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    """Runs of lines with text that no blank line separates, from the first line's start to the last's last
    non-space character."""
    spans: list[tuple[int, int]] = []
    for a, b in _line_spans(text):
        if spans and len(_LINE_BREAK.findall(text, spans[-1][1], a)) == 1:
            spans[-1] = (spans[-1][0], b)
        else:
            spans.append((a, b))
    return spans


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Sentences within each line (a line break always ends one), without surrounding spaces."""
    spans = []
    for a, b in _line_spans(text):
        a = _skip_space(text, a, b)
        start = a
        for end in _sentence_ends(text, a, b):
            spans.append((start, end))
            start = _skip_space(text, end, b)
        spans.append((start, b))
    return spans


def _sentence_ends(text: str, a: int, b: int) -> list[int]:
    """Positions inside text[a:b] where a sentence ends and another begins on the same line."""
    ends = set()
    for m in _LATIN_END.finditer(text, a, b):
        nxt = _skip_space(text, m.end(), b)
        if nxt < b and not text[nxt].islower() and not _abbreviation(text, a, m.start(), m.group(), nxt, b):
            ends.add(m.end())
    for m in _CJK_END.finditer(text, a, b):
        if _skip_space(text, m.end(), b) < b:
            ends.add(m.end())
    return sorted(ends)


def _abbreviation(text: str, line_start: int, p: int, run: str, nxt: int, b: int) -> bool:
    """Whether the single period at p closes an abbreviation rather than a sentence."""
    if run.rstrip(_CLOSERS) != ".":
        return False
    q = p
    while q > line_start and not text[q - 1].isspace():
        q -= 1
    word = text[q:p].lstrip(_OPENERS)
    if not word:
        return False
    return ((len(word) == 1 and word.isupper())                                    # an initial: "J. K. Rowling"
            or _DOTTED.fullmatch(word) is not None                                 # "U.S.", "e.g.", "a.m."
            or word.lower() in _ABBREV                                             # "Dr. Smith", "et al. (2020)"
            or (word.lower() in _ABBREV_NUMBER and _NUMBER_NEXT.match(text, nxt, b) is not None)   # "Fig. 3"
            or ((word.isdigit() or (len(word) == 1 and word.isalpha())) and q == line_start))   # "1. Mix", "a. Add"


def _skip_space(text: str, i: int, b: int) -> int:
    while i < b and text[i].isspace():
        i += 1
    return i


def _custom_spans(text: str, fn: Callable[[str], Iterable[tuple[int, int]]]) -> list[tuple[int, int]]:
    spans = []
    prev = 0
    for k, span in enumerate(fn(text)):
        try:
            start, end = span
            if isinstance(start, bool) or isinstance(end, bool):
                raise TypeError
            start, end = operator.index(start), operator.index(end)
        except (TypeError, ValueError):
            raise TypeError(f"split: the function gave {span!r} as span {k}; a span is (start, end) integers") from None
        if not prev <= start < end <= len(text):
            raise ValueError(f"split: span {k} is ({start}, {end}); spans must be sorted, not overlapping and not "
                             f"empty, within 0..{len(text)}")
        spans.append((start, end))
        prev = end
    return spans


_SPLITTERS = {"word": _word_spans, "sentence": _sentence_spans, "line": _line_spans, "paragraph": _paragraph_spans}
