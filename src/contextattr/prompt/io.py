"""Reading and writing prompts.jsonl, one prompt per line (docs/interface.md §2).

Reading checks every field and names the line and the place of any error. Writing gives each group its units
explicitly, so a file read back gives equal prompts whatever splitter made the groups.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable

from .split import KINDS, split
from .units import Group, Message, Prompt, Response, Unit

_PROMPT_FIELDS = {"id", "messages", "text", "reference", "responses"}
_MESSAGE_FIELDS = {"role", "content"}
_UNIT_FIELDS = {"unit", "text", "sep", "tag", "replace", "split"}
_GROUP_FIELDS = {"group", "units", "lead", "sep", "tag", "replace"}
_RESPONSE_FIELDS = {"id", "text", "label", "span"}


class _FormatError(ValueError):
    """An error already prefixed with its place in the prompt."""


def read_prompts(path) -> list[Prompt]:
    """Every prompt of a JSON Lines file; blank lines are skipped and prompt ids must be unique."""
    prompts, lines = [], {}
    with open(path, encoding="utf-8-sig") as f:     # -sig: a byte-order mark some editors add is skipped
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            where = f"{os.fspath(path)} line {number}"
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{where}: not valid JSON ({e.msg}, column {e.colno})") from None
            prompt = prompt_from_dict(data, where)
            if prompt.id in lines:
                raise ValueError(f"{where}: prompt id {prompt.id!r} is already used on line {lines[prompt.id]}")
            lines[prompt.id] = number
            prompts.append(prompt)
    return prompts


def write_prompts(prompts: Iterable[Prompt], path) -> None:
    """Write prompts as JSON Lines, replacing the file only once every line is written."""
    rows, seen = [], set()
    for p in prompts:
        if not isinstance(p, Prompt):
            raise TypeError(f"write_prompts takes Prompt objects, not {type(p).__name__}")
        if p.id in seen:
            raise ValueError(f"prompt id {p.id!r} is used twice")
        seen.add(p.id)
        rows.append(json.dumps(prompt_to_dict(p), ensure_ascii=False))
    tmp = f"{os.fspath(path)}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(row + "\n" for row in rows)
    os.replace(tmp, path)


def prompt_from_dict(data, where: str = "prompt") -> Prompt:
    """One prompt from its JSON object, every field checked."""
    try:
        _object(data, "the prompt")
        _fields(data, _PROMPT_FIELDS, {"id"}, "the prompt")
        messages = text = None
        if "messages" in data:
            if not isinstance(data["messages"], list):
                raise _FormatError("messages must be a list")
            messages = [_at(f"messages[{k}]", _message, m, f"messages[{k}]") for k, m in enumerate(data["messages"])]
        if "text" in data:
            text = _content(data["text"], "text")
        responses = data.get("responses", [])
        if not isinstance(responses, list):
            raise _FormatError("responses must be a list")
        return Prompt(data["id"], messages=messages, text=text, reference=data.get("reference"),
                      responses=tuple(_at(f"responses[{k}]", _response, r, f"responses[{k}]")
                                      for k, r in enumerate(responses)))
    except (TypeError, ValueError) as e:
        raise ValueError(f"{where}: {e}") from None


def prompt_to_dict(prompt: Prompt) -> dict:
    """The JSON object of a prompt; prompt_from_dict gives it back equal."""
    d: dict = {"id": prompt.id}
    if prompt.messages is not None:
        d["messages"] = [{"role": m.role, "content": _content_out(m.content)} for m in prompt.messages]
    else:
        d["text"] = _content_out(prompt.text)
    if prompt.reference is not None:
        d["reference"] = prompt.reference
    if prompt.responses:
        d["responses"] = [_response_out(r) for r in prompt.responses]
    return d


# -- reading ----------------------------------------------------------------------------------------------------------

def _at(where: str, fn, *args):
    try:
        return fn(*args)
    except _FormatError:
        raise
    except (TypeError, ValueError) as e:
        raise _FormatError(f"{where}: {e}") from None


def _object(value, where: str) -> None:
    if not isinstance(value, dict):
        raise _FormatError(f"{where} must be a JSON object, not {type(value).__name__}")


def _fields(obj: dict, allowed: set, required: set, where: str) -> None:
    unknown = sorted(set(obj) - allowed)
    if unknown:
        raise _FormatError(f"{where}: unknown field {', '.join(map(repr, unknown))} "
                           f"(allowed: {', '.join(sorted(allowed))})")
    missing = sorted(required - set(obj))
    if missing:
        raise _FormatError(f"{where}: missing field {', '.join(map(repr, missing))}")


def _message(value, where: str) -> Message:
    _object(value, where)
    _fields(value, _MESSAGE_FIELDS, _MESSAGE_FIELDS, where)
    return Message(value["role"], _content(value["content"], f"{where}.content"))


def _content(value, where: str):
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise _FormatError(f"{where} must be a string or a list, not {type(value).__name__}")
    return [_at(f"{where}[{k}]", _piece, p, f"{where}[{k}]") for k, p in enumerate(value)]


def _piece(value, where: str):
    if isinstance(value, str):
        return value
    _object(value, where)
    if "unit" in value and "group" in value:
        raise _FormatError(f"{where}: a piece is a unit or a group, not both")
    if "unit" in value:
        _fields(value, _UNIT_FIELDS, {"unit", "text"}, where)
        if "split" in value:
            if value["split"] not in KINDS:
                raise _FormatError(f"{where}: split must be one of {', '.join(KINDS)}, not {value['split']!r}")
            return split(value["text"], value["split"], id=value["unit"], sep=value.get("sep", ""),
                         tag=value.get("tag"), replace=value.get("replace"))
        return _unit(value)
    if "group" in value:
        _fields(value, _GROUP_FIELDS, {"group", "units"}, where)
        if not isinstance(value["units"], list):
            raise _FormatError(f"{where}: units must be a list")
        units = []
        for k, u in enumerate(value["units"]):
            _object(u, f"{where}.units[{k}]")
            _fields(u, _UNIT_FIELDS - {"split"}, {"unit", "text"}, f"{where}.units[{k}]")
            units.append(_at(f"{where}.units[{k}]", _unit, u))
        return Group(value["group"], tuple(units), lead=value.get("lead", ""), sep=value.get("sep", ""),
                     tag=value.get("tag"), replace=value.get("replace"))
    raise _FormatError(f"{where}: an object piece needs a 'unit' or a 'group' field")


def _unit(value: dict) -> Unit:
    return Unit(value["unit"], value["text"], sep=value.get("sep", ""), tag=value.get("tag"),
                replace=value.get("replace"))


def _response(value, where: str) -> Response:
    _object(value, where)
    _fields(value, _RESPONSE_FIELDS, {"id", "text"}, where)
    span = value.get("span")
    if span is not None and not isinstance(span, list):
        raise _FormatError(f"{where}: span must be a list [start, end]")
    return Response(value["id"], value["text"], label=value.get("label"),
                    span=tuple(span) if span is not None else None)


# -- writing ----------------------------------------------------------------------------------------------------------

def _content_out(pieces):
    if len(pieces) == 1 and isinstance(pieces[0], str):
        return pieces[0]
    return [_piece_out(p) for p in pieces]


def _piece_out(p):
    if isinstance(p, str):
        return p
    if isinstance(p, Unit):
        return _unit_out(p, None, None)
    d = {"group": p.id, "units": [_unit_out(u, p.tag, p.replace) for u in p.units]}
    for name in ("lead", "sep"):
        if getattr(p, name):
            d[name] = getattr(p, name)
    for name in ("tag", "replace"):
        if getattr(p, name) is not None:
            d[name] = getattr(p, name)
    return d


def _unit_out(u: Unit, group_tag, group_replace) -> dict:
    d = {"unit": u.id, "text": u.text}
    if u.sep:
        d["sep"] = u.sep
    if u.tag is not None and u.tag != group_tag:
        d["tag"] = u.tag
    if u.replace is not None and u.replace != group_replace:
        d["replace"] = u.replace
    return d


def _response_out(r: Response) -> dict:
    d = {"id": r.id, "text": r.text}
    if r.label is not None:
        d["label"] = r.label
    if r.span is not None:
        d["span"] = list(r.span)
    return d
