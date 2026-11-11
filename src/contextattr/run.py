"""The ways in (docs/interface.md §1, §8): explain a prompt through a model and a target, or through the user's own
function of masked prompts; attribute any function of masks; and Run, a prompt file explained with one setup."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .cache import Cache
from .evaluate import faithfulness, held_out_masks, removal as removal_curve
from .methods import ShapiqMethod, get_method
from .prompt import Prompt, prompt_to_dict, read_prompts
from .responses.checks import describe
from .result import Explanation
from .targets import Contrast, LogProb, Pooled
from .value import Value
from .wrapper.contract import Sampling, model_name


def explain(prompt: Prompt, model=None, target=None, *, value=None, name: str | None = None,
            method="contextcite", budget: int = 64, test_masks: int = 128, seed: int = 0,
            replace: str | None = None, cache: Cache | str | None = None, batch_size: int = 32,
            noise_repeats: int = 3, removal: tuple[int, ...] = ()) -> Explanation:
    """Explain one prompt. The method's masks go through the model and the target (default: LogProb of the prompt's
    only response) or through `value`, a function of masked prompts; the surrogate is fitted and then checked on
    `test_masks` held-out masks, the same for every method run with this seed, so they are asked once and cached.
    `noise_repeats` fresh values of the full prompt give the noise (0 skips it); `removal`, e.g. (1, 2, 3), the top-r
    removal of SPEX eq. 8, each r one more request."""
    if value is not None:
        if model is not None or target is not None:
            raise ValueError("give a model and a target, or a value function, not both")
        v = Value(prompt, function=value, name=name, replace=replace, cache=cache, batch_size=batch_size)
        target_name = name or "a function"
    else:
        if model is None:
            raise ValueError("explain needs a model (with a target) or a value function")
        target = target if target is not None else LogProb()
        v = Value(prompt, model, target, replace=replace, cache=cache, batch_size=batch_size)
        target_name = repr(target)
    return _run(v, prompt.n, method, budget, test_masks, seed, noise_repeats, removal, prompt=prompt,
                target=target_name)


def attribute(f, n: int, *, method="contextcite", budget: int = 64, test_masks: int = 128, seed: int = 0,
              unit_ids=None, removal: tuple[int, ...] = ()) -> Explanation:
    """Explain any function of masks: f takes an (m × n) array of 0/1, one row per mask, and gives m numbers (inputs
    that are not text, such as image patches or table columns)."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive number of units")

    def v(masks):
        out = np.asarray(f(np.array(masks, dtype=bool)), dtype=float).reshape(-1)   # a copy: f may write into it
        if len(out) != len(masks) or not np.isfinite(out).all():
            raise ValueError("f must give one finite number per mask")
        return out

    v.stats = {}
    ids = tuple(str(i) for i in range(n)) if unit_ids is None else tuple(unit_ids)
    if len(ids) != n:
        raise ValueError(f"unit_ids has {len(ids)} names for {n} units")
    return _run(v, n, method, budget, test_masks, seed, 0, removal, unit_ids=ids)


def _run(v, n, method, budget, test_masks, seed, noise_repeats, removal, *, prompt=None, target=None, unit_ids=None):
    m = get_method(method)
    surrogate, used = m.run(v, n, budget, np.random.default_rng(seed))
    r2 = None
    if test_masks:
        held_out = held_out_masks(n, test_masks, seed)
        r2 = faithfulness(surrogate.predict(held_out), v(held_out))
    noise = v.noise(noise_repeats) if noise_repeats else None
    curve = removal_curve(surrogate, v, tuple(removal)) if removal else None
    units = prompt.units if prompt is not None else None
    return Explanation(method=repr(m), unit_ids=unit_ids or tuple(u.id for u in units), surrogate=surrogate, r2=r2,
                       masks=used, test_masks=test_masks or 0, requests=dict(getattr(v, "stats", {})), removal=curve,
                       noise=noise, prompt_id=None if prompt is None else prompt.id,
                       unit_texts=None if units is None else tuple(u.text for u in units),
                       tags=None if units is None else tuple(u.tag for u in units), target=target,
                       prompt=prompt)


class Run:
    """A prompt file explained with one setup (docs/interface.md §8). `plan()` counts masks, requests and tokens before
    any call, and `plan(measure=k)` times k real requests; `execute(out)` explains every prompt into the run directory
    `out` (run.json, cache.sqlite, explanations.jsonl, summary.json, report.html) and resumes a run that stopped.

    With `sample`, each prompt first gets sample.n replies of the model, labelled by `check`. Without `target`, each
    prompt's responses choose it: Contrast with both labels, Pooled with one, LogProb with a single response."""

    def __init__(self, model, prompts, *, target=None, sample: Sampling | None = None, check=None,
                 method="contextcite", budget: int = 64, test_masks: int = 128, seed: int = 0,
                 replace: str | None = None, batch_size: int = 32, noise_repeats: int = 3,
                 removal: tuple[int, ...] = ()):
        self.prompts = read_prompts(prompts) if isinstance(prompts, (str, os.PathLike)) else list(prompts)
        ids = [p.id for p in self.prompts]
        if not ids or len(set(ids)) != len(ids) or not all(isinstance(p, Prompt) for p in self.prompts):
            raise ValueError("Run needs one or more prompts with unique ids")
        if sample is not None and not isinstance(sample, Sampling):
            raise TypeError("sample must be a Sampling")
        self.model, self.target, self.sample, self.check = model, target, sample, check
        self.method, self.budget, self.test_masks, self.seed = method, budget, test_masks, seed
        self.replace, self.batch_size, self.noise_repeats, self.removal = replace, batch_size, noise_repeats, tuple(removal)
        get_method(method)                                       # an unknown method fails now, not at the first prompt

    def target_for(self, prompt: Prompt):
        if self.target is not None:
            return self.target
        labels = {r.label for r in prompt.responses}
        if {"negative", "positive"} <= labels:
            return Contrast()
        for label in ("negative", "positive"):
            if label in labels:
                return Pooled(label)
        if len(prompt.responses) == 1:
            return LogProb()
        raise ValueError(f"prompt {prompt.id!r}: no target given, and its responses choose none (give target=, or "
                         f"responses with labels, or sample= with check=)")

    def settings(self) -> dict:
        """What decides the results; a run directory keeps one set of them."""
        return {"model": model_name(self.model), "method": repr(get_method(self.method)), "budget": self.budget,
                "test_masks": self.test_masks, "seed": self.seed, "replace": self.replace,
                "noise_repeats": self.noise_repeats, "removal": list(self.removal),
                "target": None if self.target is None else repr(self.target),
                "sample": None if self.sample is None else self.sample.as_dict(),
                "check": None if self.check is None else describe(self.check)}

    def plan(self, measure: int = 0) -> dict:
        """Masks, requests and tokens per prompt and in all, counted before any call: an upper bound, since repeated
        masks are asked once (and a prompt of n units has only 2^n masks). Tokens are estimated as characters / 4 of
        the full prompt and response; masked prompts are shorter. measure=k sends k real requests of the first prompt
        (with a cache of its own) and projects the run's time from their seconds per request."""
        rows = []
        for p in self.prompts:
            m = get_method(self.method)
            masks = self.budget if isinstance(m, ShapiqMethod) else len(m.masks(p.n, self.budget,
                                                                                  np.random.default_rng(self.seed)))
            distinct = min(masks + (self.test_masks or 0), 2 ** min(p.n, 60)) + (self.noise_repeats or 0)
            sampled = getattr(self.target, "needs", None) == "generate"
            texts = 1 if sampled else (len(p.responses) + self.sample.n if self.sample else
                                       len(self.target_for(p).items(p)))
            chars = len(json.dumps(p.render(), ensure_ascii=False))
            reply = 4 * 256 if sampled else max([len(r.text) for r in p.responses] + [400])
            requests = distinct * texts + (1 if self.sample else 0)
            generations = (distinct * self.target.samples if sampled else 0) + (self.sample.n if self.sample else 0)
            rows.append({"prompt": p.id, "units": p.n, "masks": masks, "test_masks": self.test_masks or 0,
                         "requests": requests, "generations": generations,
                         "tokens": int(requests * (chars + reply) / 4)})
        total = {k: sum(r[k] for r in rows) for k in ("masks", "requests", "generations", "tokens")}
        out = {"prompts": rows, "total": total, "measured": None}
        if measure:
            p = self.prompts[0]
            q = p if not self.sample else Prompt(p.id, messages=p.messages, text=p.text, reference=p.reference,
                                                 responses=p.responses or ())
            target = self.target if self.target is not None else (self.target_for(q) if q.responses else None)
            if target is None:
                raise ValueError("measure needs a target or responses on the first prompt")
            v = Value(q, self.model, target, replace=self.replace, batch_size=self.batch_size)
            masks = np.random.default_rng(self.seed).random((measure, q.n)) < 0.5
            start = time.perf_counter()
            v(masks)
            seconds = time.perf_counter() - start
            per = seconds / max(v.stats["requests"], 1)
            out["measured"] = {"requests": v.stats["requests"], "seconds": round(seconds, 3),
                               "seconds_per_request": per, "projected_seconds": round(per * total["requests"], 1)}
        return out

    def execute(self, out) -> Results:
        """Explain every prompt into the run directory `out` and give their explanations, in the order of the
        prompts. A directory that holds a run resumes it: a prompt explained there already is read back, unless it
        has changed since (refused: give a new directory or the prompt a new id), and a last line cut short by a kill
        is removed and its prompt explained again. Each line of explanations.jsonl holds the prompt as explained (with
        sampled responses), a digest of the prompt as given, the seconds it took and the explanation."""
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        settings = self.settings()
        meta = out / "run.json"
        if meta.exists():
            stored = json.loads(meta.read_text(encoding="utf-8"))["settings"]
            if stored != settings:
                raise ValueError(f"{out} holds a run with other settings; give a new directory\n"
                                 f"  stored: {stored}\n  now:    {settings}")
        else:
            from . import __version__  # noqa: PLC0415
            meta.write_text(json.dumps({"settings": settings, "contextattr": __version__,
                                        "started_utc": _utc()}, indent=2) + "\n", encoding="utf-8")
        lines = out / "explanations.jsonl"
        done = _rows(lines, repair=True)
        given = {p.id: _digest(p) for p in self.prompts}
        changed = [pid for pid, d in given.items() if pid in done and done[pid].get("input") != d]
        if changed:
            raise ValueError(f"{out}: prompt {', '.join(map(repr, changed))} changed since it was explained there; "
                             f"give a new directory, or the prompt a new id")
        cache = Cache(out / "cache.sqlite")
        from .responses import sample as sample_responses  # noqa: PLC0415
        for p in self.prompts:
            if p.id in done:
                continue
            start = time.perf_counter()
            q = sample_responses(p, self.model, self.sample, self.check, cache=cache, replace=self.replace) \
                if self.sample else p
            e = explain(q, self.model, self.target_for(q), method=self.method, budget=self.budget,
                        test_masks=self.test_masks, seed=self.seed, replace=self.replace, cache=cache,
                        batch_size=self.batch_size, noise_repeats=self.noise_repeats, removal=self.removal)
            row = {"prompt": prompt_to_dict(q), "input": given[p.id], "seconds": round(time.perf_counter() - start, 3),
                   "explanation": e.as_dict()}
            with open(lines, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            done[p.id] = row
        cache.close()
        results = Results([Explanation.from_dict(done[p.id]["explanation"], done[p.id]["prompt"])
                           for p in self.prompts])
        (out / "summary.json").write_text(json.dumps(results.summary(), indent=2) + "\n", encoding="utf-8")
        results.to_html(out / "report.html")
        return results


class Results:
    """The explanations of a run, with a summary over prompts."""

    def __init__(self, explanations: list[Explanation]):
        self.explanations = list(explanations)

    @classmethod
    def load(cls, out) -> Results:
        """Every explanation a run directory holds, the prompts of earlier runs into it included (a last line cut short
        is left out)."""
        path = Path(out) / "explanations.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")
        return cls([Explanation.from_dict(r["explanation"], r["prompt"]) for r in _rows(path).values()])

    def summary(self, by: str = "tag", top: int = 3) -> dict:
        """For each tag (by="tag") or unit id (by="unit"): in how many prompts it occurs, its units' mean score, and the
        share of those prompts in which one of its units is among the `top` highest scores."""
        if by not in ("tag", "unit"):
            raise ValueError("by is tag or unit")
        acc: dict = {}
        for e in self.explanations:
            keys = e.tags if by == "tag" else e.unit_ids
            if keys is None:
                continue
            best = {u for u, _ in e.top(top)}
            seen = set()
            for uid, key, score in zip(e.unit_ids, keys, e.surrogate.scores):
                if key is None:
                    continue
                a = acc.setdefault(key, {"prompts": 0, "units": 0, "score_sum": 0.0, "in_top": 0})
                a["units"] += 1
                a["score_sum"] += float(score)
                if key not in seen:
                    seen.add(key)
                    a["prompts"] += 1
                    a["in_top"] += any(u in best for u, k in zip(e.unit_ids, keys) if k == key)
        return {key: {"prompts": a["prompts"], "units": a["units"], "mean_score": a["score_sum"] / a["units"],
                      f"share_in_top_{top}": a["in_top"] / a["prompts"]} for key, a in sorted(acc.items())}

    def to_html(self, path=None, title: str = "ContextAttr report") -> str:
        from .report import page  # noqa: PLC0415

        text = page(self.explanations, title)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def __len__(self) -> int:
        return len(self.explanations)


def _digest(prompt: Prompt) -> str:
    """The prompt as given, units, texts and responses: what a resumed run compares."""
    text = json.dumps(prompt_to_dict(prompt), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _rows(path: Path, repair: bool = False) -> dict:
    """The rows of explanations.jsonl by prompt id, the first of each. A last line without its newline was cut short
    by a kill while it was written: it is left out, and with `repair` removed from the file. Any other line that is not
    a row means the file is damaged."""
    if not path.exists():
        return {}
    data = path.read_bytes()
    whole, newline, cut = data.rpartition(b"\n")
    if cut and repair:
        with open(path, "r+b") as f:
            f.truncate(len(whole) + len(newline))
    rows: dict = {}
    for k, line in enumerate((whole + newline).decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            rows.setdefault(row["prompt"]["id"], row)
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"{path}: line {k} is not a row of a run ({error}); the file is damaged") from error
    return rows


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
