"""check_model (docs/wrapper.md): about a dozen calls that tell a user whether their wrapper meets the contract.

Every problem is reported, none raised: a check FAILs when the wrapper breaks the contract or its answers cannot
serve attribution, and gives a note when something is legal but worth knowing (answers that vary, seeds ignored).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from ..prompt import Prompt, Unit
from .contract import (Sampling, Scored, WrapperError, call_generate, call_score, can_generate, can_score,
                       model_name)

OK, FAIL, NOTE, INFO, SKIP = "ok", "FAIL", "note", "info", "skip"
DEFAULT_RESPONSE = "It is hot and humid, because Carnival falls in the Brazilian summer."


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


@dataclass
class ModelReport:
    """What check_model found: each check, and the targets the wrapper supports."""

    model: str
    checks: list[Check] = field(default_factory=list)
    scores: bool = False
    generates: bool = False
    spans: bool = False

    @property
    def ok(self) -> bool:
        """Ready for attribution: a name, a working method, and no failed check."""
        return (self.scores or self.generates) and not any(c.status == FAIL for c in self.checks)

    @property
    def targets(self) -> list[str]:
        """The targets this wrapper can serve: the scored ones need `score`, Sampled needs only `generate`."""
        return (["LogProb", "Pooled", "Contrast", "Choice"] if self.scores else []) + \
               (["Sampled"] if self.generates else [])

    def add(self, name: str, status: str, detail: str) -> None:
        self.checks.append(Check(name, status, detail))

    def __str__(self) -> str:
        width = max([len(c.name) for c in self.checks] + [0])
        lines = [f"check_model: {self.model}"]
        lines += [f"  {c.status:<4}  {c.name:<{width}}  {c.detail}" for c in self.checks]
        spans = (" (spans: yes)" if self.spans else " (spans: no)") if self.scores else ""
        lines.append(f"  targets: {', '.join(self.targets) or 'none yet'}{spans}")
        lines.append("  ready: every check passed" if self.ok else "  NOT ready for attribution: see the FAIL lines")
        return "\n".join(lines)


def default_prompt() -> Prompt:
    """The prompt check_model uses unless given one: the interface's Rio example."""
    return Prompt("check", messages=[
        {"role": "system", "content": "Answer in one sentence."},
        {"role": "user", "content": [
            "Documents:\n",
            Unit("d1", "Rio Carnival takes place before Lent, in February or March.", sep="\n"),
            Unit("d2", "Summer in Brazil runs from December to March.", sep="\n"),
            Unit("d3", "Winter in Brazil runs from June to September.", sep="\n"),
            "Question: What is the weather like during Rio Carnival?"]}])


def check_model(model, prompt: Prompt | None = None, response: str | None = None) -> ModelReport:
    """Test a wrapper against the contract on `prompt` (default: the Rio example; give a text prompt for a model that
    takes plain text) and `response` (default: the model's own greedy reply, else a fixed sentence)."""
    prompt = prompt if prompt is not None else default_prompt()
    full, empty = prompt.render(), prompt.render([0] * prompt.n)
    try:
        name = model_name(model)
    except WrapperError as e:
        report = ModelReport(type(model).__name__)
        report.add("name", FAIL, str(e))
        return report
    report = ModelReport(name)
    report.add("name", OK, repr(name))
    has = [m for m, ok in (("score", can_score(model)), ("generate", can_generate(model))) if ok]
    report.add("methods", OK if has else FAIL, ", ".join(has) if has else "neither score nor generate")
    timings: list[tuple[float, int]] = []

    def timed(fn, *args):
        start = time.perf_counter()
        out = fn(*args)
        timings.append((time.perf_counter() - start, len(args[1])))
        return out

    reply = None
    if "generate" in has:
        try:
            greedy = timed(call_generate, model, [full, full], Sampling(n=2, temperature=0.0))
            report.generates = True
            report.add("generate", OK, f"2 greedy replies to each of 2 prompts, e.g. {greedy[0][0][:60]!r}")
            again = timed(call_generate, model, [full], Sampling(n=1, temperature=0.0))
            same = len({greedy[0][0], greedy[0][1], greedy[1][0], greedy[1][1], again[0][0]}) == 1
            report.add("greedy repeats", OK if same else NOTE,
                       "identical" if same else "greedy replies vary: sampled targets carry that noise")
            s1 = timed(call_generate, model, [full], Sampling(n=1, temperature=1.0, seed=7))
            s2 = timed(call_generate, model, [full], Sampling(n=1, temperature=1.0, seed=7))
            report.add("seed", OK if s1 == s2 else NOTE, "honoured" if s1 == s2 else
                       "two calls with seed 7 differ: a rerun samples anew where the cache has no answer")
            reply = greedy[0][0] if greedy[0][0].strip() else None
        except Exception as e:  # noqa: BLE001 - every failure is reported, not raised
            report.add("generate", FAIL, f"{type(e).__name__}: {e}")
    if "score" in has:
        r = response if response is not None else (reply or DEFAULT_RESPONSE)
        try:
            a = timed(call_score, model, [full, full], [r, r])
            b = timed(call_score, model, [full], [r])
            if not a[0].logprobs:
                raise WrapperError("score gave no tokens for a non-empty response")
            report.scores = True
            report.add("score", OK, f"{len(a[0].logprobs)} tokens, finite log-probabilities at most 0")
            diff = max(_diff(a[0], a[1]), _diff(a[0], b[0]))
            report.add("score repeats", OK if diff == 0 else NOTE, "identical" if diff == 0 else
                       f"differ by up to {diff:.2g}: the noise check measures this")
            if a[0].tokens is None:
                report.add("token texts", NOTE, "none given: targets cannot use spans")
            elif "".join(a[0].tokens) == r:
                report.spans = True
                report.add("token texts", OK, "join back to the response: targets can use spans")
            else:
                report.add("token texts", FAIL, "given, but they do not join back to the response exactly")
            c = timed(call_score, model, [empty], [r])
            m_full, m_empty = _mean(a[0]), _mean(c[0])
            report.add("prompt matters", OK if m_full != m_empty else FAIL,
                       f"{m_full:.3f} per token with every unit, {m_empty:.3f} with none" +
                       ("" if m_full != m_empty else ": the score ignores the prompt; does it reach the model?"))
            words = r.split()
            backwards = " ".join(reversed(words))
            if len(set(words)) < 2 or backwards == r:
                report.add("reply above reversed", SKIP, "the response has too few distinct words")
            else:
                d = timed(call_score, model, [full, full], [r, backwards])
                m1, m2 = _mean(d[0]), _mean(d[1])
                report.add("reply above reversed", OK if m1 > m2 else NOTE,
                           f"{m1:.3f} per token, {m2:.3f} with its words reversed" + ("" if m1 > m2 else
                           ": a model blind to word order, or the response not placed as the model's reply"))
        except Exception as e:  # noqa: BLE001
            report.add("score", FAIL, f"{type(e).__name__}: {e}")
    if timings:
        seconds, requests = sum(t for t, _ in timings), sum(k for _, k in timings)
        report.add("time", INFO, f"{seconds / requests:.3g} s per request over {len(timings)} calls")
    return report


def _mean(s: Scored) -> float:
    return math.fsum(s.logprobs) / len(s.logprobs) if s.logprobs else math.nan


def _diff(x: Scored, y: Scored) -> float:
    if len(x.logprobs) != len(y.logprobs):
        return math.inf
    return max((abs(p - q) for p, q in zip(x.logprobs, y.logprobs)), default=0.0)
