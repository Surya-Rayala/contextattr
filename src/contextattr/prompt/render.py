"""A mask → the prompt as the model sees it: chat messages or text, removed units deleted or replaced (§2, §6)."""

from __future__ import annotations

from .units import Group, Prompt, Unit


def as_mask(mask, n: int) -> tuple[bool, ...]:
    """Check a mask: n entries, each 1 or True (keep) or 0 or False (remove). None keeps every unit."""
    if mask is None:
        return (True,) * n
    if isinstance(mask, (str, bytes)):
        raise TypeError("a mask is a sequence of 0/1 or True/False, not a string")
    try:
        values = list(mask)
    except TypeError:
        raise TypeError(f"a mask is a sequence of 0/1 or True/False, not {type(mask).__name__}") from None
    if len(values) != n:
        raise ValueError(f"the mask has {len(values)} entries, the prompt {n} units")
    out = []
    for k, v in enumerate(values):
        if v is True or v is False:
            out.append(v)
            continue
        try:
            ok = not isinstance(v, (str, bytes)) and bool(v == 0 or v == 1)
        except (TypeError, ValueError):  # e.g. a row of a 2-D array
            ok = False
        if not ok:
            raise ValueError(f"mask entry {k} is {v!r}: use 0/1 or True/False")
        out.append(bool(v == 1))
    return tuple(out)


def render(prompt: Prompt, mask=None, replace: str | None = None):
    """The prompt with only the units `mask` keeps. A removed unit is deleted together with its `sep`, or, when it
    has its own `replace` text or `replace` is given, becomes that text followed by its `sep`. A group's `lead` and
    `sep` stay while any of its units shows. Chat prompts give [{"role", "content"}] dicts, text prompts a string."""
    if replace is not None and not isinstance(replace, str):
        raise TypeError(f"replace must be a string or None, not {type(replace).__name__}")
    keep = iter(as_mask(mask, prompt.n))
    if prompt.messages is not None:
        return [{"role": m.role, "content": _join(m.content, keep, replace)} for m in prompt.messages]
    return _join(prompt.text, keep, replace)


def _join(pieces, keep, replace: str | None) -> str:
    parts: list[str] = []
    for p in pieces:
        if isinstance(p, str):
            parts.append(p)
        elif isinstance(p, Unit):
            parts.append(_unit(p, next(keep), replace)[0])
        elif isinstance(p, Group):
            shown = [_unit(u, next(keep), replace) for u in p.units]
            if any(visible for _, visible in shown):
                parts.append(p.lead)
                parts.extend(text for text, _ in shown)
                parts.append(p.sep)
    return "".join(parts)


def _unit(u: Unit, kept: bool, replace: str | None) -> tuple[str, bool]:
    """The unit's text in the rendered prompt, and whether it shows (kept, or replaced)."""
    if kept:
        return u.text + u.sep, True
    r = u.replace if u.replace is not None else replace
    if r is None:
        return "", False
    return r + u.sep, True
