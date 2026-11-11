"""Targets read from `score` (docs/interface.md §5): the log-probability of given responses under each masked prompt,
with nothing generated. A target names the texts to score (its items) and how their numbers combine into the value."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..prompt import Prompt, Response
from ..wrapper.contract import Scored

REDUCES = ("mean", "sum", "logodds")


@dataclass(frozen=True)
class Item:
    """One text scored under every masked prompt: its id, its label, the span that counts, and how its token
    log-probabilities become one number."""

    id: str
    text: str
    label: str | None = None
    span: tuple[int, int] | None = None
    reduce: str = "mean"


class ScoredTarget:
    """A target that needs the wrapper's `score`."""

    needs = "score"

    def items(self, prompt: Prompt) -> list[Item]:
        raise NotImplementedError

    def combine(self, columns: np.ndarray, items: list[Item]) -> np.ndarray:
        """(masks × items) numbers → one value per mask."""
        raise NotImplementedError


@dataclass(frozen=True)
class LogProb(ScoredTarget):
    """The log-probability of one response, or of its span when it has one: per token on average ("mean", the
    negative log-perplexity of SPEX §3), in total ("sum"), or as the logit of its probability ("logodds",
    ContextCite §3). `response` is a response id of the prompt, a Response, or None for the prompt's only one."""

    response: str | Response | None = None
    reduce: str = "mean"

    def __post_init__(self) -> None:
        _check_reduce(self.reduce)

    def items(self, prompt: Prompt) -> list[Item]:
        r = _resolve(prompt, self.response)
        return [Item(r.id, r.text, r.label, r.span, self.reduce)]

    def combine(self, columns: np.ndarray, items: list[Item]) -> np.ndarray:
        return columns[:, 0]


@dataclass(frozen=True)
class Pooled(ScoredTarget):
    """The mean LogProb of the prompt's responses with one label."""

    label: str = "negative"
    reduce: str = "mean"

    def __post_init__(self) -> None:
        _check_reduce(self.reduce)
        if self.label not in ("positive", "negative"):
            raise ValueError(f"Pooled: label must be positive or negative, not {self.label!r}")

    def items(self, prompt: Prompt) -> list[Item]:
        return _labelled(prompt, self.label, self.reduce)

    def combine(self, columns: np.ndarray, items: list[Item]) -> np.ndarray:
        return columns.mean(axis=1)


@dataclass(frozen=True)
class Contrast(ScoredTarget):
    """Pooled("negative") − Pooled("positive"): what moves the model from desired to undesired responses."""

    reduce: str = "mean"

    def __post_init__(self) -> None:
        _check_reduce(self.reduce)

    def items(self, prompt: Prompt) -> list[Item]:
        return _labelled(prompt, "negative", self.reduce) + _labelled(prompt, "positive", self.reduce)

    def combine(self, columns: np.ndarray, items: list[Item]) -> np.ndarray:
        neg = np.array([i.label == "negative" for i in items])
        return columns[:, neg].mean(axis=1) - columns[:, ~neg].mean(axis=1)


@dataclass(frozen=True)
class Choice(ScoredTarget):
    """log p(a) − log p(b), each the total log-probability of that text as the response (SPEX B.2.4: "True" −
    "False")."""

    a: str
    b: str

    def __post_init__(self) -> None:
        if not (isinstance(self.a, str) and isinstance(self.b, str) and self.a and self.b and self.a != self.b):
            raise ValueError("Choice needs two different non-empty texts")

    def items(self, prompt: Prompt) -> list[Item]:
        return [Item("choice:a", self.a, reduce="sum"), Item("choice:b", self.b, reduce="sum")]

    def combine(self, columns: np.ndarray, items: list[Item]) -> np.ndarray:
        return columns[:, 0] - columns[:, 1]


def reduce_scored(scored: Scored, item: Item) -> float:
    """One number from a response's token log-probabilities, as the item says."""
    logprobs = scored.logprobs
    if item.span is not None:
        logprobs = [x for x, inside in zip(logprobs, span_tokens(scored, item)) if inside]
    if not logprobs:
        raise ValueError(f"no tokens to score for {item.id!r}" + (" within its span" if item.span else ""))
    total = math.fsum(logprobs)
    if item.reduce == "mean":
        return total / len(logprobs)
    if item.reduce == "sum":
        return total
    return logodds(total)


def logodds(logprob: float) -> float:
    """log(p / (1 − p)) for p = exp(logprob)."""
    if logprob >= 0:
        return math.inf
    return logprob - math.log(-math.expm1(logprob))


def span_tokens(scored: Scored, item: Item) -> list[bool]:
    """Which tokens overlap the item's character span; the token texts must join back to the response."""
    if scored.tokens is None:
        raise ValueError(f"the span of {item.id!r} needs token texts, which the wrapper's score did not give")
    if "".join(scored.tokens) != item.text:
        raise ValueError(f"the span of {item.id!r} needs token texts that join back to the response; the "
                         f"wrapper's do not")
    start, end = item.span
    out, pos = [], 0
    for t in scored.tokens:
        out.append(pos < end and pos + len(t) > start)
        pos += len(t)
    return out


def _check_reduce(reduce: str) -> None:
    if reduce not in REDUCES:
        raise ValueError(f"reduce must be one of {', '.join(REDUCES)}, not {reduce!r}")


def _resolve(prompt: Prompt, response) -> Response:
    if isinstance(response, Response):
        return response
    if response is None:
        if len(prompt.responses) != 1:
            raise ValueError(f"prompt {prompt.id!r} has {len(prompt.responses)} responses: name the one to explain")
        return prompt.responses[0]
    for r in prompt.responses:
        if r.id == response:
            return r
    raise ValueError(f"prompt {prompt.id!r} has no response {response!r} "
                     f"(it has: {', '.join(r.id for r in prompt.responses) or 'none'})")


def _labelled(prompt: Prompt, label: str, reduce: str) -> list[Item]:
    items = [Item(r.id, r.text, r.label, r.span, reduce) for r in prompt.responses if r.label == label]
    if not items:
        raise ValueError(f"prompt {prompt.id!r} has no {label} response")
    return items
