"""The sampled target (docs/interface.md §5): it needs only the wrapper's `generate`, so it works with any model,
an API's included. For each masked prompt, k fresh replies; the value is the share a check flags as negative, or the
mean of a numeric check (a judge's score, a classifier's probability).

Every masked prompt is sampled with the same seed (common random numbers), so differences between masks are not
swamped by the luck of the draw where the backend honours seeds; the noise check samples with other seeds."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

from ..prompt import Prompt
from ..responses.checks import describe, verdict
from ..wrapper.contract import Sampling

DEFAULT = Sampling(n=4, temperature=0.7, top_p=0.95, max_tokens=256, seed=0)


@dataclass(frozen=True)
class Sampled:
    """`samples` replies to each masked prompt, sampled as `sampling` says (default: temperature 0.7, top_p 0.95,
    256 tokens, seed 0), and the mean of the check's values over them (negative 1, positive 0, or its number)."""

    check: object
    samples: int = 4
    sampling: Sampling | None = None

    needs = "generate"

    def __post_init__(self) -> None:
        if not callable(self.check):
            raise TypeError("Sampled needs a check: a function (prompt, response) → label, bool or number")
        if isinstance(self.samples, bool) or not isinstance(self.samples, int) or self.samples < 1:
            raise ValueError("Sampled: samples must be a positive integer")

    def sampling_for(self, repeat: int = 0) -> Sampling:
        """The sampling of every masked prompt, with `samples` replies; a noise repeat k samples with seed + k."""
        base = self.sampling or DEFAULT
        seed = None if base.seed is None else base.seed + repeat
        return dataclasses.replace(base, n=self.samples, seed=seed)

    def value(self, prompt: Prompt, replies: list[str]) -> float:
        values = [verdict(self.check(prompt, r)).value for r in replies]
        values = [v for v in values if v is not None]
        if not values:
            raise ValueError(f"the check labelled none of the {len(replies)} replies to a masked prompt of "
                             f"{prompt.id!r}")
        return math.fsum(values) / len(values)

    def __repr__(self) -> str:
        """The check (responses.describe) and the sampling every masked prompt gets, so that a run's settings and an
        explanation's target say both."""
        s = self.sampling_for(0)
        stop = f", stop={list(s.stop)!r}" if s.stop else ""
        return (f"Sampled({describe(self.check)}, samples={self.samples}, temperature={float(s.temperature)!r}, "
                f"top_p={float(s.top_p)!r}, max_tokens={s.max_tokens}, seed={s.seed}{stop})")
