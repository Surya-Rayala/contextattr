"""The input format of docs/interface.md §2: Prompt, Message, Unit, Group and Response, with their checks."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType
from typing import Union

LABELS = ("positive", "negative")


def _need_str(value, what: str, *, empty: bool = True) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{what} must be a string, not {type(value).__name__}")
    if not empty and not value:
        raise ValueError(f"{what} is empty")


def _need_opt_str(value, what: str) -> None:
    if value is not None:
        _need_str(value, what)


def _need_sequence(value, what: str) -> None:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Iterable):
        raise TypeError(f"{what} must be a list, not {type(value).__name__}")


@dataclass(frozen=True)
class Unit:
    """A removable piece of a prompt: `text`, then `sep`, which goes with it when it is deleted and stays when it is
    replaced. `replace` is this unit's replacement text, overriding the run's removal setting."""

    id: str
    text: str
    sep: str = ""
    tag: str | None = None
    replace: str | None = None
    group: str | None = None  # the id of the Group holding it, set by the Group

    def __post_init__(self) -> None:
        _need_str(self.id, "unit id", empty=False)
        _need_str(self.text, f"unit {self.id!r}: text", empty=False)
        _need_str(self.sep, f"unit {self.id!r}: sep")
        _need_opt_str(self.tag, f"unit {self.id!r}: tag")
        _need_opt_str(self.replace, f"unit {self.id!r}: replace")
        _need_opt_str(self.group, f"unit {self.id!r}: group")


@dataclass(frozen=True)
class Group:
    """Units that belong together, such as a document cut into sentences. `lead` comes before them and `sep` after;
    both stay while any of the units shows (kept, or replaced) and go with the last one deleted. `tag` and `replace`
    are given to each unit that has none of its own."""

    id: str
    units: tuple[Unit, ...]
    lead: str = ""
    sep: str = ""
    tag: str | None = None
    replace: str | None = None

    def __post_init__(self) -> None:
        _need_str(self.id, "group id", empty=False)
        _need_str(self.lead, f"group {self.id!r}: lead")
        _need_str(self.sep, f"group {self.id!r}: sep")
        _need_opt_str(self.tag, f"group {self.id!r}: tag")
        _need_opt_str(self.replace, f"group {self.id!r}: replace")
        _need_sequence(self.units, f"group {self.id!r}: units")
        units = []
        for u in self.units:
            if not isinstance(u, Unit):
                raise TypeError(f"group {self.id!r}: units must be Unit objects, not {type(u).__name__}")
            if u.group not in (None, self.id):
                raise ValueError(f"group {self.id!r}: unit {u.id!r} already belongs to group {u.group!r}")
            units.append(dataclasses.replace(
                u, group=self.id,
                tag=self.tag if u.tag is None else u.tag,
                replace=self.replace if u.replace is None else u.replace))
        if not units:
            raise ValueError(f"group {self.id!r} has no units")
        object.__setattr__(self, "units", tuple(units))


Piece = Union[str, Unit, Group]


def _pieces(content, what: str) -> tuple[Piece, ...]:
    """A message's content or a text prompt: a string (all fixed), or a list of strings, units and groups."""
    if isinstance(content, str):
        return (content,)
    _need_sequence(content, what)
    out = []
    for p in content:
        if isinstance(p, Unit) and p.group is not None:
            raise ValueError(f"{what}: unit {p.id!r} names group {p.group!r} but is not inside it")
        if not isinstance(p, (str, Unit, Group)):
            raise TypeError(f"{what}: a piece is a string, a Unit or a Group, not {type(p).__name__}")
        out.append(p)
    return tuple(out)


@dataclass(frozen=True)
class Message:
    """One chat message: its role and its content, a sequence of fixed strings, units and groups."""

    role: str
    content: tuple[Piece, ...]

    def __post_init__(self) -> None:
        _need_str(self.role, "message role", empty=False)
        object.__setattr__(self, "content", _pieces(self.content, f"{self.role} message"))


@dataclass(frozen=True)
class Response:
    """A reply of the model: `label` is positive, negative or None; `span` [start, end) marks the characters that
    matter."""

    id: str
    text: str
    label: str | None = None
    span: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        _need_str(self.id, "response id", empty=False)
        _need_str(self.text, f"response {self.id!r}: text", empty=False)
        if self.label is not None and self.label not in LABELS:
            raise ValueError(f"response {self.id!r}: label must be positive, negative or None, not {self.label!r}")
        if self.span is not None:
            _need_sequence(self.span, f"response {self.id!r}: span")
            span = tuple(self.span)
            if len(span) != 2 or not all(isinstance(x, int) and not isinstance(x, bool) for x in span):
                raise TypeError(f"response {self.id!r}: span must be two integers [start, end)")
            start, end = span
            if not 0 <= start < end <= len(self.text):
                raise ValueError(f"response {self.id!r}: span {list(span)} must satisfy 0 <= start < end <= "
                                 f"{len(self.text)}, the length of its text")
            object.__setattr__(self, "span", span)


@dataclass(frozen=True)
class Prompt:
    """A prompt with its removable units: chat `messages` or a plain `text`, never both. Messages may be given as
    dicts {"role", "content"} and responses as dicts; everything not marked as a unit is fixed."""

    id: str
    messages: tuple[Message, ...] | None = None
    text: tuple[Piece, ...] | None = None
    reference: str | None = None
    responses: tuple[Response, ...] = ()

    def __post_init__(self) -> None:
        _need_str(self.id, "prompt id", empty=False)
        where = f"prompt {self.id!r}"
        if (self.messages is None) == (self.text is None):
            raise ValueError(f"{where}: give either messages or text")
        if self.messages is not None:
            _need_sequence(self.messages, f"{where}: messages")
            messages = []
            for m in self.messages:
                if isinstance(m, Mapping):
                    if set(m) != {"role", "content"}:
                        raise ValueError(f"{where}: a message has the fields role and content, not {sorted(m)}")
                    m = Message(m["role"], m["content"])
                elif not isinstance(m, Message):
                    raise TypeError(f"{where}: a message is a Message or a dict, not {type(m).__name__}")
                messages.append(m)
            if not messages:
                raise ValueError(f"{where}: messages is empty")
            object.__setattr__(self, "messages", tuple(messages))
        else:
            object.__setattr__(self, "text", _pieces(self.text, f"{where}: text"))
        _need_opt_str(self.reference, f"{where}: reference")
        _need_sequence(self.responses, f"{where}: responses")
        responses = tuple(r if isinstance(r, Response) else Response(**r) if isinstance(r, Mapping) else
                          _not_a_response(r, where) for r in self.responses)
        ids = [r.id for r in responses]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{where}: response ids must be unique: {ids}")
        object.__setattr__(self, "responses", responses)

        seen: set[str] = set()
        for p in self._top_pieces():
            for name in ([p.id, *(u.id for u in p.units)] if isinstance(p, Group) else
                         [p.id] if isinstance(p, Unit) else []):
                if name in seen:
                    raise ValueError(f"{where}: the id {name!r} is used twice (unit and group ids must be unique)")
                seen.add(name)
        if not self.units:
            raise ValueError(f"{where} has no units: mark at least one piece as a unit")

    def _top_pieces(self):
        if self.messages is not None:
            for m in self.messages:
                yield from m.content
        else:
            yield from self.text

    @cached_property
    def units(self) -> tuple[Unit, ...]:
        """Every unit in order: message by message, piece by piece, a group's units in place."""
        out: list[Unit] = []
        for p in self._top_pieces():
            if isinstance(p, Unit):
                out.append(p)
            elif isinstance(p, Group):
                out.extend(p.units)
        return tuple(out)

    @property
    def n(self) -> int:
        """The number of units: the length of a mask."""
        return len(self.units)

    @property
    def is_chat(self) -> bool:
        return self.messages is not None

    @cached_property
    def unit_ids(self) -> tuple[str, ...]:
        return tuple(u.id for u in self.units)

    @cached_property
    def _index(self) -> Mapping[str, int]:
        return MappingProxyType({u.id: i for i, u in enumerate(self.units)})

    @cached_property
    def groups(self) -> Mapping[str, tuple[int, ...]]:
        """Each group's id → the positions of its units in `units`."""
        out: dict[str, list[int]] = {}
        for i, u in enumerate(self.units):
            if u.group is not None:
                out.setdefault(u.group, []).append(i)
        return MappingProxyType({g: tuple(ix) for g, ix in out.items()})

    def index(self, unit_id: str) -> int:
        """The position of a unit in `units`, and so in a mask."""
        try:
            return self._index[unit_id]
        except KeyError:
            raise KeyError(f"prompt {self.id!r} has no unit {unit_id!r}") from None

    def mask(self, *, keep: Iterable[str] | str | None = None,
             drop: Iterable[str] | str | None = None) -> tuple[bool, ...]:
        """A mask from ids of units or groups: keep only `keep`, or keep every unit except `drop`."""
        if (keep is None) == (drop is None):
            raise ValueError("give either keep or drop")
        names = keep if keep is not None else drop
        chosen: set[int] = set()
        for name in [names] if isinstance(names, str) else names:
            if name in self.groups:
                chosen.update(self.groups[name])
            elif name in self._index:
                chosen.add(self._index[name])
            else:
                raise KeyError(f"prompt {self.id!r} has no unit or group {name!r}")
        return tuple((i in chosen) == (keep is not None) for i in range(self.n))

    def render(self, mask=None, replace: str | None = None):
        """The prompt as the model sees it with only the units `mask` keeps (render.py)."""
        from .render import render

        return render(self, mask, replace)


def _not_a_response(value, where: str):
    raise TypeError(f"{where}: a response is a Response or a dict, not {type(value).__name__}")
