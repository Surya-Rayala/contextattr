"""Checks: what labels a response (docs/interface.md §4). A check is any function check(prompt, response) giving a
Verdict, a label ("positive", "negative" or None), True for negative and False for positive, or a number (a judge's
score, a classifier's probability); `verdict` reads any of these; `describe` names a check in a run's settings."""

from __future__ import annotations

import functools
import hashlib
import math
import numbers
import re
import string
import types
from dataclasses import dataclass

import numpy as np

from ..prompt import LABELS, Prompt


@dataclass(frozen=True)
class Verdict:
    """A check's answer: a label, or a number, and optionally the span [start, end) of the response that matters."""

    label: str | None = None
    score: float | None = None
    span: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.label is not None and self.label not in LABELS:
            raise ValueError(f"a verdict's label is positive, negative or None, not {self.label!r}")
        if self.score is not None and not math.isfinite(self.score):
            raise ValueError("a verdict's score must be finite")

    @property
    def value(self) -> float | None:
        """What Sampled averages: 1 for negative, 0 for positive, else the score; None when there is neither."""
        if self.score is not None:
            return float(self.score)
        return None if self.label is None else float(self.label == "negative")


def verdict(result) -> Verdict:
    """Any check's answer as a Verdict."""
    if isinstance(result, Verdict):
        return result
    if result is None or isinstance(result, str):
        return Verdict(label=result)
    if isinstance(result, bool):
        return Verdict(label="negative" if result else "positive")
    if isinstance(result, numbers.Real):
        return Verdict(score=float(result))
    raise TypeError(f"a check gives a Verdict, a label, a bool or a number, not {type(result).__name__}")


def regex(pattern: str, *, negative: bool = True, flags: int = re.IGNORECASE):
    """A check by pattern: a response that matches it is negative (or positive, with negative=False) and the match is
    its span; one that does not match is the other label."""
    compiled = re.compile(pattern, flags)

    def check(prompt: Prompt, response: str) -> Verdict:
        m = compiled.search(response)
        hit, miss = ("negative", "positive") if negative else ("positive", "negative")
        if m is None:
            return Verdict(label=miss)
        return Verdict(label=hit, span=(m.start(), m.end()) if m.end() > m.start() else None)

    shown = "" if flags == re.IGNORECASE else f", flags={re.RegexFlag(flags)!r}"
    check.__name__ = f"regex({pattern!r}, negative={negative}{shown})"
    check.named = True
    return check


def reference(*, contains: bool = True):
    """A check against the prompt's `reference`: positive when the response, normalised (lower case, no punctuation,
    single spaces), contains the normalised reference (or equals it, with contains=False); negative otherwise."""

    def check(prompt: Prompt, response: str) -> Verdict:
        if prompt.reference is None:
            raise ValueError(f"prompt {prompt.id!r} has no reference to check against")
        ref, got = normalise(prompt.reference), normalise(response)
        ok = (ref in got) if contains else (ref == got)
        return Verdict(label="positive" if ok else "negative")

    check.__name__ = f"reference(contains={contains})"
    check.named = True
    return check


def describe(check) -> str:
    """A check's name in a run's settings and an explanation's target, so that a resumed run cannot mix two checks.
    regex's and reference's names carry their arguments. Any other check (a function, a bound method, a callable
    object) is named by its name and a digest of its code (bytecode, constants and names, not line numbers) and of the
    values it holds: its closure's and defaults', or its object's attributes, with numbers, strings and patterns by
    value and other objects (a client, a model) by type. A check that depends on anything else should carry it in its
    __name__ and set `named = True`."""
    if getattr(check, "named", False) is True:
        return check.__name__
    if isinstance(check, functools.partial):
        return f"partial({describe(check.func)}, {_held(check.args)}, {_held(check.keywords)})"
    fn = getattr(check, "__func__", check)                       # a bound method: its function, and its object
    if isinstance(fn, types.FunctionType):
        owner = getattr(check, "__self__", None)
        state = [_code(fn.__code__), [_cell(c) for c in fn.__closure__ or ()], _held(fn.__defaults__),
                 _held(fn.__kwdefaults__), _held(getattr(owner, "__dict__", None))]
        name = fn.__qualname__
    else:
        call = getattr(type(check).__call__, "__code__", None)
        state = [_code(call) if call is not None else "", _held(getattr(check, "__dict__", None))]
        name = type(check).__qualname__
    return f"{name} [{hashlib.sha256(repr(state).encode()).hexdigest()[:12]}]"


def _code(code: types.CodeType) -> str:
    def const(c):
        if isinstance(c, types.CodeType):
            return _code(c)
        if isinstance(c, (tuple, frozenset)):                     # a set's order changes with the hash seed
            parts = [const(x) for x in c]
            return repr(sorted(parts) if isinstance(c, frozenset) else parts)
        return repr(c)

    return repr((code.co_code.hex(), [const(c) for c in code.co_consts], code.co_names, code.co_varnames))


def _cell(cell) -> str:
    try:
        return _held(cell.cell_contents)
    except ValueError:                                           # a cell not yet filled
        return "<empty>"


def _held(value) -> str:
    if value is None or isinstance(value, (str, bytes, numbers.Number)):
        return repr(value)
    if isinstance(value, re.Pattern):
        return f"re.compile({value.pattern!r}, {int(value.flags)})"
    if isinstance(value, np.ndarray):
        digest = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        return f"array({value.dtype}, {value.shape}, {digest})"
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(_held(v) for v in value) + "]"
    if isinstance(value, (set, frozenset)):
        return "{" + ", ".join(sorted(_held(v) for v in value)) + "}"
    if isinstance(value, dict):
        return "{" + ", ".join(sorted(f"{_held(k)}: {_held(v)}" for k, v in value.items())) + "}"
    if isinstance(value, types.FunctionType):                    # a function it calls: by its code, not recursing
        return value.__name__ if getattr(value, "named", False) is True else _code(value.__code__)
    return f"<{type(value).__qualname__}>"


_PUNCTUATION = str.maketrans({c: " " for c in string.punctuation})


def normalise(text: str) -> str:
    return " ".join(text.lower().translate(_PUNCTUATION).split())
