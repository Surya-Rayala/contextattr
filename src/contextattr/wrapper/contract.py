"""What a user's wrapper must do (docs/wrapper.md), and the checks applied to every answer it gives.

A wrapper is any object with a `name` and at least one of two methods:

    score(prompts, responses) -> list[Scored]       the log-probability of each response token, given its prompt
    generate(prompts, sampling) -> list[list[str]]  sampling.n replies to each prompt

Prompts reach it rendered (`Rendered`): a string, or chat messages [{"role": ..., "content": "..."}]. The wrapper
applies its model's own chat template, batches as it likes, and raises on errors; it never truncates a prompt.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, Union, runtime_checkable

Rendered = Union[str, list[dict[str, str]]]

# A log-probability above 0 is impossible; rounding in low precision can give a hair above it.
LOGPROB_TOLERANCE = 1e-3


class WrapperError(RuntimeError):
    """A wrapper broke the contract: its answer cannot be used."""


@dataclass(frozen=True)
class Sampling:
    """How `generate` samples: `n` replies per prompt; temperature 0 is greedy."""

    n: int = 1
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 256
    seed: int | None = 0
    stop: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _is_int(self.n) or self.n < 1:
            raise ValueError(f"Sampling.n must be an integer of at least 1, not {self.n!r}")
        if not _is_real(self.temperature) or not 0 <= self.temperature < math.inf:
            raise ValueError(f"Sampling.temperature must be a finite number of at least 0, not {self.temperature!r}")
        if not _is_real(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError(f"Sampling.top_p must lie in (0, 1], not {self.top_p!r}")
        if not _is_int(self.max_tokens) or self.max_tokens < 1:
            raise ValueError(f"Sampling.max_tokens must be an integer of at least 1, not {self.max_tokens!r}")
        if self.seed is not None and not _is_int(self.seed):
            raise ValueError(f"Sampling.seed must be an integer or None, not {self.seed!r}")
        stop = (self.stop,) if isinstance(self.stop, str) else tuple(self.stop)
        if not all(isinstance(s, str) and s for s in stop):
            raise ValueError(f"Sampling.stop must hold non-empty strings, not {self.stop!r}")
        object.__setattr__(self, "stop", stop)

    def as_dict(self) -> dict:
        return {"n": self.n, "temperature": float(self.temperature), "top_p": float(self.top_p),
                "max_tokens": self.max_tokens, "seed": self.seed, "stop": list(self.stop)}


@dataclass(frozen=True)
class Scored:
    """What `score` gives for one response: the natural-log probability of each response token, in order, and
    optionally each token's text; the texts must join back to the response exactly, and are needed only for spans."""

    logprobs: tuple[float, ...]
    tokens: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.logprobs, (str, bytes)):
            raise TypeError("Scored.logprobs must be a sequence of numbers")
        try:
            logprobs = tuple(float(x) for x in self.logprobs)
        except (TypeError, ValueError):
            raise TypeError("Scored.logprobs must be a sequence of numbers") from None
        for k, x in enumerate(logprobs):
            if not math.isfinite(x) or x > LOGPROB_TOLERANCE:
                raise ValueError(f"Scored.logprobs[{k}] is {x}: a log-probability must be finite and at most 0")
        object.__setattr__(self, "logprobs", logprobs)
        if self.tokens is not None:
            if isinstance(self.tokens, (str, bytes)):
                raise TypeError("Scored.tokens must be a sequence of strings, one per token")
            tokens = tuple(self.tokens)
            if not all(isinstance(t, str) for t in tokens):
                raise TypeError("Scored.tokens must be a sequence of strings, one per token")
            if len(tokens) != len(logprobs):
                raise ValueError(f"Scored has {len(logprobs)} log-probabilities but {len(tokens)} token texts")
            object.__setattr__(self, "tokens", tokens)


@runtime_checkable
class Scorer(Protocol):
    """A wrapper that can score given responses."""

    name: str

    def score(self, prompts: Sequence[Rendered], responses: Sequence[str]) -> list[Scored]: ...


@runtime_checkable
class Generator(Protocol):
    """A wrapper that can generate replies."""

    name: str

    def generate(self, prompts: Sequence[Rendered], sampling: Sampling) -> list[list[str]]: ...


def model_name(model) -> str:
    """The wrapper's name, which every cache key holds; it must change whenever the answers could change."""
    name = getattr(model, "name", None)
    if not isinstance(name, str) or not name:
        raise WrapperError(f"a wrapper needs a non-empty string `name`; {type(model).__name__} has {name!r}")
    return name


def can_score(model) -> bool:
    return callable(getattr(model, "score", None))


def can_generate(model) -> bool:
    return callable(getattr(model, "generate", None))


def call_score(model, prompts: list[Rendered], responses: list[str]) -> list[Scored]:
    """Ask the wrapper to score, and check its answer's shape."""
    name = model_name(model)
    out = model.score(prompts, responses)
    if isinstance(out, (str, bytes, dict)) or not hasattr(out, "__len__"):
        raise WrapperError(f"{name}.score returned {type(out).__name__}, not a list of Scored")
    out = list(out)
    if len(out) != len(prompts):
        raise WrapperError(f"{name}.score returned {len(out)} answers for {len(prompts)} requests")
    for k, s in enumerate(out):
        if not isinstance(s, Scored):
            raise WrapperError(f"{name}.score answer {k} is {type(s).__name__}, not Scored")
    return out


def call_generate(model, prompts: list[Rendered], sampling: Sampling) -> list[list[str]]:
    """Ask the wrapper to generate, and check that every prompt got exactly sampling.n strings."""
    name = model_name(model)
    out = model.generate(prompts, sampling)
    if isinstance(out, (str, bytes, dict)) or not hasattr(out, "__len__"):
        raise WrapperError(f"{name}.generate returned {type(out).__name__}, not a list of lists of strings")
    out = list(out)
    if len(out) != len(prompts):
        raise WrapperError(f"{name}.generate returned replies for {len(out)} prompts, not {len(prompts)}")
    replies = []
    for k, r in enumerate(out):
        if isinstance(r, (str, bytes)) or not hasattr(r, "__len__"):
            raise WrapperError(f"{name}.generate answer {k} is {type(r).__name__}, not a list of strings")
        r = list(r)
        if len(r) != sampling.n or not all(isinstance(x, str) for x in r):
            raise WrapperError(f"{name}.generate answer {k} must hold {sampling.n} strings (sampling.n)")
        replies.append(r)
    return replies


def _is_int(x) -> bool:
    return isinstance(x, numbers.Integral) and not isinstance(x, bool)


def _is_real(x) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, bool)
