"""Explanation: what explain and attribute give (docs/interface.md §8)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from .methods import indices
from .methods.base import Surrogate
from .prompt import Prompt, prompt_from_dict, prompt_to_dict


def sign_meaning(target: str | None) -> str:
    """One line on how to read the sign of a score, from the target's name as an Explanation keeps it. The rule is the
    same for every target: a score above 0 means the unit pushes the model towards what is explained."""
    target = target or ""
    choice = re.fullmatch(r"Choice\(a=(.+), b=(.+)\)", target)
    pooled = re.match(r"Pooled\(label='(\w+)'", target)
    if target.startswith("LogProb("):
        return "above 0: the unit makes the answer more likely · below 0: less likely"
    if pooled:
        return f"above 0: the unit makes the {pooled.group(1)} answers more likely · below 0: less likely"
    if target.startswith("Contrast("):
        return ("above 0: the unit pushes towards the positive (wanted) answers · below 0: towards the negative "
                "(unwanted) ones")
    if choice:
        return f"above 0: the unit pushes towards {choice.group(1)} · below 0: towards {choice.group(2)}"
    if target.startswith("Sampled("):
        return "above 0: the unit makes more replies get flagged by the check · below 0: fewer"
    return "above 0: the unit raises the value · below 0: lowers it"


@dataclass(frozen=True)
class Explanation:
    """The score of each unit (positive: it pushes the value up), how faithful the surrogate is on held-out masks
    (R², SPEX eq. 7), the noise of the value, and what it cost in masks and requests."""

    method: str
    unit_ids: tuple[str, ...]
    surrogate: Surrogate
    r2: float | None
    masks: int
    test_masks: int
    requests: dict
    noise: dict | None = None
    prompt_id: str | None = None
    unit_texts: tuple[str, ...] | None = None
    tags: tuple[str | None, ...] | None = None
    target: str | None = None
    removal: list[dict] | None = None
    prompt: Prompt | None = None

    @property
    def scores(self) -> dict[str, float]:
        """Each unit's id → its score."""
        return dict(zip(self.unit_ids, self.surrogate.scores.tolist()))

    @property
    def interactions(self) -> dict[tuple[str, ...], float]:
        """Each set of two or more units (by id) → what they do together (interaction methods only)."""
        return {tuple(self.unit_ids[i] for i in s): w for s, w in self.surrogate.interactions.items()}

    def shapley(self) -> dict[str, float]:
        """Each unit's Shapley value, read from the whole surrogate (App. C): its share of every term it is in."""
        return dict(zip(self.unit_ids, indices.shapley(self.surrogate).tolist()))

    def banzhaf(self) -> dict[str, float]:
        """Each unit's Banzhaf value, read from the whole surrogate (App. C)."""
        return dict(zip(self.unit_ids, indices.banzhaf(self.surrogate).tolist()))

    def shapley_taylor(self, order: int = 2) -> dict[tuple[str, ...], float]:
        """The Shapley-Taylor index of an order, by unit ids (App. C, eq. 58)."""
        return {tuple(self.unit_ids[i] for i in s): w for s, w in indices.shapley_taylor(self.surrogate, order).items()}

    def top_interactions(self, k: int = 5) -> list[tuple[tuple[str, ...], float]]:
        """The k interactions of largest absolute value."""
        return sorted(self.interactions.items(), key=lambda t: -abs(t[1]))[:k]

    def top(self, k: int = 5) -> list[tuple[str, float]]:
        """The k units with the highest scores: those that push the value up most."""
        order = np.argsort(-self.surrogate.scores, kind="stable")[:k]
        return [(self.unit_ids[i], float(self.surrogate.scores[i])) for i in order]

    def table(self, width: int = 60) -> str:
        """A plain table: the first line, what a score above and below 0 means, then every unit, highest first."""
        noise = self.noise["std"] if self.noise else None
        head = (f"{self.prompt_id or 'attribute'} · {self.method} · {self.masks} masks · "
                f"R² {_fmt(self.r2)} on {self.test_masks} test masks · noise {_fmt(noise)} · "
                f"{self.requests.get('requests', 0)} requests")
        rows = ["  " + sign_meaning(self.target)]
        shown = np.round(self.surrogate.scores, 4) + 0.0     # as printed: ties keep the prompt's order, and no -0.0000
        for i in np.argsort(-shown, kind="stable"):
            text = "" if self.unit_texts is None else self.unit_texts[i].replace("\n", " ")
            text = text if len(text) <= width else text[: width - 1] + "…"
            rows.append(f"  {self.unit_ids[i]:<12} {shown[i]:+10.4f}  {text}")
        for s, w in self.top_interactions(5):
            rows.append(f"  {' & '.join(s):<12} {w:+10.4f}  (together)")
        for r in self.removal or []:
            removed = ", ".join(self.unit_ids[i] for i in r["removed"])
            rows.append(f"  removing {removed}: the value moves by {r['change']:+.4f} (Rem {_fmt(r['rem'])})")
        return "\n".join([head, *rows])

    def as_dict(self) -> dict:
        return {"prompt": self.prompt_id, "method": self.method, "target": self.target, "units": list(self.unit_ids),
                "tags": None if self.tags is None else list(self.tags), "scores": self.surrogate.scores.tolist(),
                "intercept": self.surrogate.intercept,
                "interactions": [{"units": list(s), "score": w} for s, w in self.interactions.items()],
                "r2": self.r2, "masks": self.masks, "test_masks": self.test_masks, "requests": dict(self.requests),
                "noise": self.noise, "removal": self.removal,
                "unit_texts": None if self.unit_texts is None else list(self.unit_texts)}

    @classmethod
    def from_dict(cls, d: dict, prompt: dict | Prompt | None = None) -> Explanation:
        """The Explanation as_dict gave, with its prompt when given (as a Prompt or its JSON object)."""
        units = list(d["units"])
        surrogate = Surrogate(intercept=d["intercept"], scores=d["scores"], interactions={
            tuple(units.index(u) for u in t["units"]): t["score"] for t in d["interactions"]})
        if isinstance(prompt, dict):
            prompt = prompt_from_dict(prompt)
        return cls(method=d["method"], unit_ids=tuple(units), surrogate=surrogate, r2=d["r2"], masks=d["masks"],
                   test_masks=d["test_masks"], requests=dict(d["requests"]), noise=d["noise"], prompt_id=d["prompt"],
                   unit_texts=None if d.get("unit_texts") is None else tuple(d["unit_texts"]),
                   tags=None if d["tags"] is None else tuple(d["tags"]), target=d["target"], removal=d["removal"],
                   prompt=prompt)

    def to_html(self, path=None) -> str:
        """This explanation as an HTML page (report.py), written to `path` when given."""
        from .report import page, write  # noqa: PLC0415

        if path is not None:
            write([self], path)
        return page([self])

    def __str__(self) -> str:
        return self.table()


def _fmt(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3g}"
