"""The value function (docs/interface.md §1, §5): masks → one number per masked prompt.

Through a wrapper and a target, each distinct request goes to the wrapper once, in batches, and every answer is
cached first: scored targets ask `score` for each (masked prompt, text), the sampled target asks `generate` for the
replies to each masked prompt (the check is applied after the cache, so changing it asks the model nothing). Or
through the user's own function of masked prompts, cached when it has a name.
"""

from __future__ import annotations

import copy
import math

import numpy as np

from .cache import Cache
from .prompt import Prompt
from .prompt.render import render
from .targets.scored import reduce_scored
from .wrapper.contract import (Sampling, Scored, WrapperError, call_generate, call_score, can_generate, can_score,
                               model_name)


def as_masks(masks, n: int) -> np.ndarray:
    """Check masks: an (m × n) array of 0/1 or booleans, one row per mask, 1 keeping a unit."""
    a = np.asarray(masks)
    if a.ndim != 2 or a.shape[1] != n:
        raise ValueError(f"masks must be an (m × {n}) array of 0/1, one row per mask, not shape {a.shape}")
    if a.dtype == bool:
        return a
    if not np.issubdtype(a.dtype, np.number) or not np.isin(a, (0, 1)).all():
        raise ValueError("masks hold only 0/1 or True/False")
    return a.astype(bool)


class Value:
    """f(m) for masks m. Called with an (m × n) array of 0/1 it gives m values; with per_response=True (scored
    targets), an (m × items) array, one column per scored text (see `items`)."""

    def __init__(self, prompt: Prompt, model=None, target=None, *, function=None, name: str | None = None,
                 replace: str | None = None, cache: Cache | str | None = None, batch_size: int = 32,
                 per_response: bool = False):
        if not isinstance(prompt, Prompt):
            raise TypeError(f"Value needs a Prompt, not {type(prompt).__name__}")
        if (model is None) == (function is None):
            raise ValueError("Value needs a model and a target, or a function")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer, not {batch_size!r}")
        if replace is not None and not isinstance(replace, str):
            raise TypeError("replace must be a string or None")
        self.prompt, self.model, self.target, self.function = prompt, model, target, function
        self.replace, self.batch_size, self.per_response = replace, batch_size, per_response
        self.cache = cache if isinstance(cache, Cache) else Cache(cache)
        self.stats = {"requests": 0, "reused": 0, "calls": 0}
        if model is not None:
            needs = getattr(target, "needs", None)
            if needs not in ("score", "generate"):
                raise TypeError(f"target must be a target such as LogProb, Contrast or Sampled, not "
                                f"{type(target).__name__}")
            self.name = model_name(model)
            if needs == "score" and not can_score(model):
                raise WrapperError(f"{type(target).__name__} needs the wrapper's score, which {self.name} lacks "
                                   f"(a wrapper that can only generate serves Sampled)")
            if needs == "generate" and not can_generate(model):
                raise WrapperError(f"{type(target).__name__} needs the wrapper's generate, which {self.name} lacks")
            if needs == "generate" and per_response:
                raise ValueError("per_response applies to scored targets")
            self.items = target.items(prompt) if needs == "score" else []
            if needs == "generate":
                self.stats["generations"] = 0
        else:
            if not callable(function):
                raise TypeError("function must be callable: masked prompts → numbers")
            if target is not None or per_response:
                raise ValueError("a function gives the value itself: no target, no per_response")
            if name is not None and (not isinstance(name, str) or not name):
                raise ValueError("name must be a non-empty string; it keys the cache")
            self.name, self.items = name, []

    @property
    def n(self) -> int:
        return self.prompt.n

    def __call__(self, masks) -> np.ndarray:
        rows = as_masks(masks, self.n)
        return self._values(self._render(rows), per_response=self.per_response)

    def replies(self, masks, repeat: int = 0) -> list[list[str]]:
        """The sampled target's replies to each masked prompt (cached like everything else)."""
        if getattr(self.target, "needs", None) != "generate":
            raise ValueError("replies come from a sampled target")
        rendered = self._render(as_masks(masks, self.n))
        sampling = self.target.sampling_for(repeat)
        requests = [self._request("generate", p, None, repeat, sampling) for p in rendered]
        answers = self._ask(requests, lambda batch: self._generate(batch, sampling))
        return [answers[key] for key, _, _ in requests]

    def noise(self, repeats: int = 3) -> dict:
        """The spread of the full prompt's value: from the main requests and from `repeats` fresh ones, each
        cached under its own key, so a rerun gives the same figures. A sampled target samples a repeat k with seed
        + k, so its spread is the sampling's."""
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
            raise ValueError("repeats must be a positive integer")
        full = self._render(np.ones((1, self.n), dtype=bool))
        values = [float(self._values(full, repeat=k)[0]) for k in range(repeats + 1)]
        return {"values": values, "std": float(np.std(values, ddof=1))}

    # -- inside ---------------------------------------------------------------------------------------------------

    def _render(self, rows: np.ndarray) -> list:
        done: dict[bytes, object] = {}
        out = []
        for row in rows:
            key = row.tobytes()
            if key not in done:
                done[key] = render(self.prompt, row.tolist(), self.replace)
            out.append(done[key])
        return out

    def _values(self, rendered: list, repeat: int = 0, per_response: bool = False) -> np.ndarray:
        needs = getattr(self.target, "needs", None)
        if self.model is not None and needs == "score":
            requests = [[self._request("score", p, item.text, repeat) for item in self.items] for p in rendered]
            answers = self._ask(sum(requests, []), self._score)
            columns = np.array([[reduce_scored(answers[key], item) for (key, _, _), item in zip(row, self.items)]
                                for row in requests], dtype=float).reshape(len(rendered), len(self.items))
            out = columns if per_response else self.target.combine(columns, self.items)
        elif self.model is not None:
            sampling = self.target.sampling_for(repeat)
            requests = [self._request("generate", p, None, repeat, sampling) for p in rendered]
            answers = self._ask(requests, lambda batch: self._generate(batch, sampling))
            out = np.array([self.target.value(self.prompt, answers[key]) for key, _, _ in requests], dtype=float)
        elif self.name is None:                                  # an unnamed function: nothing is cached
            out = np.array(self._apply([(None, p, None) for p in rendered]), dtype=float)
        else:
            requests = [self._request("function", p, None, repeat) for p in rendered]
            answers = self._ask(requests, self._apply)
            out = np.array([answers[key] for key, _, _ in requests], dtype=float)
        if not np.isfinite(out).all():
            raise ValueError("some values are not finite: a response with probability 1 under logodds, or a "
                             "function giving inf or nan")
        return out

    def _request(self, kind: str, prompt, text, repeat: int, sampling: Sampling | None = None):
        req = {"kind": kind, "model" if kind != "function" else "name": self.name, "prompt": prompt}
        if text is not None:
            req["response"] = text
        if sampling is not None:
            req["sampling"] = sampling.as_dict()
        if repeat:
            req["repeat"] = repeat
        return (Cache.key(req), prompt, text)

    def _ask(self, requests: list, run) -> dict:
        """Each distinct request answered once: from the cache, else by `run` in batches, each stored at once."""
        distinct = {key: (key, p, t) for key, p, t in requests}
        found = self.cache.get_many(distinct)
        self.stats["reused"] += len(found)
        missing = [r for key, r in distinct.items() if key not in found]
        kind = "function" if self.model is None else self.target.needs
        for i in range(0, len(missing), self.batch_size):
            batch = missing[i:i + self.batch_size]
            fresh = dict(zip((key for key, _, _ in batch), run(batch)))
            self.cache.put_many(kind, fresh)
            found.update(fresh)
        if self.model is not None and self.target.needs == "score":
            return {key: Scored(tuple(a["logprobs"]), None if a["tokens"] is None else tuple(a["tokens"]))
                    for key, a in found.items()}
        return found

    def _score(self, batch: list) -> list:
        answers = call_score(self.model, [copy.deepcopy(p) for _, p, _ in batch], [t for _, _, t in batch])
        self.stats["requests"] += len(batch)
        self.stats["calls"] += 1
        return [{"logprobs": list(a.logprobs), "tokens": None if a.tokens is None else list(a.tokens)}
                for a in answers]

    def _generate(self, batch: list, sampling: Sampling) -> list:
        replies = call_generate(self.model, [copy.deepcopy(p) for _, p, _ in batch], sampling)
        self.stats["requests"] += len(batch)
        self.stats["calls"] += 1
        self.stats["generations"] = self.stats.get("generations", 0) + len(batch) * sampling.n
        return replies

    def _apply(self, batch: list) -> list:
        values = []
        for i in range(0, len(batch), self.batch_size):
            chunk = batch[i:i + self.batch_size]
            out = self.function([copy.deepcopy(p) for _, p, _ in chunk])
            self.stats["requests"] += len(chunk)
            self.stats["calls"] += 1
            try:
                out = [float(x) for x in out]
            except (TypeError, ValueError):
                raise TypeError("the function must give one number per masked prompt") from None
            if len(out) != len(chunk):
                raise ValueError(f"the function gave {len(out)} numbers for {len(chunk)} masked prompts")
            if not all(math.isfinite(x) for x in out):
                raise ValueError("the function gave a value that is not finite")
            values.extend(out)
        return values
