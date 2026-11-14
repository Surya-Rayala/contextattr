"""A demonstration on planted facts (docs/wrapper.md §5, docs/results.md): three prompts whose facts are invented, so that the right
result is known before any model is asked. It answers the question that `check_model` leaves open: the wrapper keeps
the contract, but are the results right with this model?

    ca.demo()                      the built-in stand-in: seconds, no model behind it
    ca.demo(model)                 your wrapper
    contextattr demo [my_wrapper.py:MyModel]

  1. one document holds the answer            -> it comes first, and without it the answer changes
  2. two documents hold it together            -> both score, and the pair has a term of its own
  3. a later document overrules an earlier one -> it pushes towards the new answer, the earlier one against it

Each case prints the scores of every method side by side, and then one line per check:

    ok     as it should be
    FAIL   the package or the wrapper is wrong: these checks hold for any model (the scores against brute force over
           every combination of the documents, a pair's term against its direct measurement, the cache)
    note   this model read the documents in another way than planted: a fact about the model, not an error

A wrapper that scores is asked about every combination of each case's 8 documents: 1,024 prompts to score, and a few
short replies if it can also generate. One that can only generate gets the first case alone, by counting replies."""

from __future__ import annotations

import itertools
import math
from importlib.util import find_spec

import numpy as np

from .cache import Cache
from .evaluate import intervene
from .prompt import Prompt, Response, Unit
from .responses import regex
from .run import explain
from .targets import Choice, LogProb, Sampled
from .value import Value
from .wrapper.contract import Sampling, call_generate, can_generate, can_score, model_name
from .wrapper.toy import ToyModel

QUESTION = {1: "In which year was the Velmar Bridge opened? Answer with the year only.",
            2: "In which month is the Lantern Festival of Qarim held? Answer with the month only.",
            3: "How tall is the Brask Tower in metres? Answer with the number only."}
DOCUMENTS = {1: ["The town of Velmar lies on the river Osk, forty kilometres from the coast.",
                 "The Velmar Bridge was opened in 1932.",
                 "Most visitors reach Velmar by the regional train from the capital.",
                 "The town library of Velmar was rebuilt after a fire.",
                 "The Osk valley is known for its orchards and its cider.",
                 "The market of Velmar is held on the square every Saturday.",
                 "The river Osk freezes over in most winters.",
                 "The football club of Velmar plays in green and white."],
             2: ["The town of Qarim lies on the river Senn.",
                 "The Lantern Festival of Qarim is held in the same month as the Apple Fair of Senn.",
                 "The Apple Fair of Senn takes place every October.",
                 "The town library of Qarim was rebuilt after a fire.",
                 "Most visitors reach Qarim by the regional train from the capital.",
                 "The river Senn floods the lower town in some springs.",
                 "The market hall of Qarim dates from the last century.",
                 "The football club of Qarim plays in blue."],
             3: ["The Brask Tower stands in the old harbour of Brask.",
                 "When it was built, the Brask Tower was 310 metres tall.",
                 "The harbour of Brask is known for its fish market.",
                 "Since its new spire was added last year, the Brask Tower is now 450 metres tall.",
                 "Most visitors reach Brask by ferry.",
                 "The old harbour of Brask is closed to cars.",
                 "The lighthouse of Brask is painted red and white.",
                 "The museum of Brask opens every day except Monday."]}
TITLE = {1: "One document holds the answer (d2: the bridge was opened in 1932)",
         2: "Two documents hold the answer together (d2 ties the festival to the fair, d3 gives the fair's month)",
         3: "A later document overrules an earlier one (d2: 310 metres when built; d4: now 450 metres)"}
# Each method with its budget of masks; "all" is every combination of the documents, which makes loo, banzhaf and
# kernelshap exact, so that brute force can be compared with them.
ALONE = [("loo", "all"), ("contextcite", 24), ("lime", 24), ("banzhaf", "all"), ("kernelshap", "all")]
TOGETHER = [("faithbanzhaf", "all"), ("faithshap", "all"), ("spex", 4032), ("proxyspex", 512)]


def planted_toy() -> ToyModel:
    """The stand-in `demo()` uses without a model: fixed rules that answer from the planted documents as a careful
    reader would. It is not a language model and has no weights; its results are right by construction."""
    bridge, tie, fair = "velmar bridge was opened in 1932", "same month as the apple fair", "apple fair of senn takes"
    old, new = "brask tower was 310 metres", "brask tower is now 450 metres"
    return ToyModel(rules={bridge: {"1932": 6.0}, fair: {"october": 3.0}, old: {"310": 5.0},
                           new: {"450": 7.0, "310": -4.0}},
                    pairs={(tie, fair): {"october": 3.0}}, bias=-3.0,
                    replies={bridge: "1932", fair: "October", new: "450", old: "310"}, name="toy-planted")


def case(k: int, responses=()) -> Prompt:
    """The prompt of case k: its 8 documents as units d1 to d8, then the question."""
    units = [Unit(f"d{i + 1}", text, sep="\n") for i, text in enumerate(DOCUMENTS[k])]
    return Prompt(f"planted-{k}", messages=[{"role": "user", "content": ["Documents:\n", *units,
                                                                         "Question: " + QUESTION[k]]}],
                  responses=list(responses))


class _Report:
    def __init__(self, show):
        self.show, self.checks = show, []           # checks: (about the model's reading only, passed)

    def line(self, name: str, passed: bool, detail: str = "", reading: bool = False) -> bool:
        self.checks.append((reading, bool(passed)))
        self.show(f"  {'ok' if passed else 'note' if reading else 'FAIL':<5} {name:<46} {detail}")
        return bool(passed)


def _brute_force(prompt: Prompt, model, target, cache) -> tuple[dict, dict]:
    """From the value of every combination of documents, worked out here: each document's effect when it alone is
    removed, its average effect over all combinations (Banzhaf), and its Shapley value."""
    n = prompt.n
    masks = np.array(list(itertools.product([True, False], repeat=n)))
    f = {tuple(m.tolist()): float(v) for m, v in zip(masks, Value(prompt, model, target, cache=cache)(masks))}
    full = (True,) * n
    removed = [f[full] - f[full[:i] + (False,) + full[i + 1:]] for i in range(n)]
    average = [float(np.mean([v for m, v in f.items() if m[i]]) - np.mean([v for m, v in f.items() if not m[i]]))
               for i in range(n)]
    weight = [math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n) for k in range(n)]
    shapley = [math.fsum(weight[sum(m)] * (f[m[:i] + (True,) + m[i + 1:]] - f[m]) for m in f if not m[i])
               for i in range(n)]
    return {"loo": removed, "banzhaf": average, "kernelshap": shapley}, f


def _table(prompt: Prompt, model, target, cache, groups, report: _Report) -> tuple[dict, dict]:
    """Every method's scores side by side, then the scores against brute force."""
    show = report.show
    made = {name: explain(prompt, model, target, method=name, noise_repeats=0, cache=cache,
                          budget=2 ** prompt.n if budget == "all" else budget)
            for group in groups for name, budget in group}
    titles = ("One score for each document:",
              "Methods that also score documents together (shown: each document's effect alone):")
    for title, group in zip(titles, groups):
        show(f"  {title}")
        show(f"    {'':<4}" + "".join(f"{name:>13}" for name, _ in group) + "   the document")
        for unit in prompt.units:
            show(f"    {unit.id:<4}" + "".join(f"{made[name].scores[unit.id]:>+13.3f}" for name, _ in group) +
                 f"   {unit.text[:50]}")
        show(f"    {'R²':<4}" + "".join(f"{made[name].r2:>13.3f}" for name, _ in group) +
             "   (1 = predicts the model on other combinations)")
    exact, f = _brute_force(prompt, model, target, cache)
    worst = {name: max(abs(a - b) for a, b in zip(values, made[name].scores.values()))
             for name, values in exact.items()}
    report.line("loo, banzhaf, kernelshap = brute force", max(worst.values()) < 1e-6,
                f"over all {len(f)} combinations; largest differences " + ", ".join(f"{x:.1e}" for x in worst.values()))
    return made, f


def _answer(model, prompt: Prompt, without: str | None = None) -> str:
    """The model's most likely reply to the prompt, with every document or without one of them."""
    mask = [int(u.id != without) for u in prompt.units]
    return call_generate(model, [prompt.render(mask)], Sampling(n=1, max_tokens=12))[0][0].strip()


def _first(report: _Report, made: dict, planted: str) -> None:
    tops = [e.top(1)[0][0] for e in made.values()]
    report.line(f"{planted} comes first for every method", set(tops) == {planted}, "first: " + ", ".join(tops),
                reading=True)


def _case_1(model, cache, groups, report: _Report, replies: bool) -> None:
    prompt = case(1, [Response("answer", "1932")])
    made, f = _table(prompt, model, LogProb(), cache, groups, report)
    _first(report, made, "d2")
    row = explain(prompt, model, LogProb(), method="kernelshap", budget=2 ** prompt.n, noise_repeats=0,
                  removal=(1,), cache=cache).removal[0]
    kept = tuple(i not in row["removed"] for i in range(prompt.n))
    report.line("removal's row = the value's real move", math.isclose(row["change"], f[kept] - f[(True,) * prompt.n],
                abs_tol=1e-9), f"removing {prompt.units[row['removed'][0]].id}: {row['change']:+.3f}")
    if replies:
        said, without = _answer(model, prompt), _answer(model, prompt, "d2")
        report.line('the model answers "1932"', "1932" in said, f"its answer: {said!r}", reading=True)
        report.line("without d2 the answer changes", "1932" not in without, f"without d2: {without!r}", reading=True)


def _case_2(model, cache, groups, report: _Report, replies: bool) -> None:
    prompt = case(2, [Response("answer", "October")])
    made, f = _table(prompt, model, LogProb(), cache, groups, report)
    both, d3, d2, neither = (f[tuple(u.id in keep for u in prompt.units)] for keep in
                             ({"d2", "d3"}, {"d3"}, {"d2"}, set()))
    measured = both - d3 - d2 + neither
    report.show(f"  Measured directly, the value with only: d2 and d3 {both:+.3f} | d3 {d3:+.3f} | d2 {d2:+.3f} | "
                f"neither {neither:+.3f}. Together they add {measured:+.3f} beyond their separate effects.")
    if len(groups) > 1:
        from .methods import FaithBanzhaf  # noqa: PLC0415 - the interaction methods are an extra

        full = explain(prompt, model, LogProb(), method=FaithBanzhaf(max_order=prompt.n), budget=2 ** prompt.n,
                       noise_repeats=0, cache=cache)                 # every set of documents gets its own term
        pair = full.interactions.get(("d2", "d3"), 0.0)
        report.line("the pair's own term = its direct measurement", math.isclose(pair, measured, abs_tol=1e-6),
                    f"faithbanzhaf with every set {pair:+.3f}; spex "
                    f"{made['spex'].interactions.get(('d2', 'd3'), 0.0):+.3f}")
    top2 = sorted(u for u, _ in made["kernelshap"].top(2))
    report.line("d2 and d3 are the two top documents", top2 == ["d2", "d3"], "top two: " + " and ".join(top2),
                reading=True)
    report.line("the model needs both of them", measured > 0, "together they add more than apart" if measured > 0 else
                "it gets the answer from one of them alone: the two overlap", reading=True)
    if replies:
        said = _answer(model, prompt)
        report.line('the model answers "October"', "october" in said.lower(), f"its answer: {said!r}", reading=True)


def _case_3(model, cache, groups, report: _Report, replies: bool) -> None:
    prompt = case(3)
    report.show('  The value is log p("450") − log p("310") (ca.Choice): above zero pushes towards 450, below towards '
                "310.")
    made, _ = _table(prompt, model, Choice("450", "310"), cache, groups, report)
    ends = [(max(e.scores, key=e.scores.get), min(e.scores, key=e.scores.get)) for e in made.values()]
    report.line("d4 highest and d2 lowest for every method", set(ends) == {("d4", "d2")},
                "highest/lowest: " + ", ".join(f"{a}/{b}" for a, b in ends), reading=True)
    if replies:
        said, without = _answer(model, prompt), _answer(model, prompt, "d4")
        report.line("450 with d4, 310 without it", "450" in said and "310" in without and "450" not in without,
                    f"with: {said!r}; without: {without!r}", reading=True)


def _replies_only(model, cache, report: _Report) -> None:
    """The first case for a wrapper that can only generate: the share of replies that give the planted year."""
    show = report.show
    prompt, check = case(1), regex(r"\b1932\b")
    sampling = Sampling(temperature=0.7, top_p=0.95, max_tokens=12)
    target = Sampled(check, samples=4, sampling=sampling)
    show("  The value is the share of 4 replies that say 1932 (ca.Sampled), with leave-one-out:")
    e = explain(prompt, model, target, method="loo", test_masks=0, noise_repeats=2, cache=cache)
    for line in str(e).splitlines():
        show("  " + line)
    n = prompt.n
    masks = np.vstack([np.ones((1, n), dtype=bool), ~np.eye(n, dtype=bool)])
    values = Value(prompt, model, target, cache=cache)(masks)
    report.line("loo = the measured change of the share", np.allclose(list(e.scores.values()), values[0] - values[1:],
                atol=1e-9), f"with every document {values[0]:.2f} of the replies say 1932")
    report.line("d2 comes first", e.top(1)[0][0] == "d2", f"first: {e.top(1)[0][0]}; the noise is "
                f"{e.noise['std']:.2f}", reading=True)
    found = intervene(prompt, model, check, ["d2"], samples=16, sampling=sampling, cache=cache)
    report.line("without d2 fewer replies say 1932", found["after"] < found["before"],
                f"{found['before']:.0%} of 16 replies before, {found['after']:.0%} after", reading=True)


def demo(model=None, *, cache: Cache | str | None = None, interactions: bool | None = None, show=print) -> bool:
    """Run the demonstration on `model` (a wrapper; none: the built-in stand-in) and print it with `show`. `cache`
    is a file that keeps every answer of the model, so that a second run asks nothing. `interactions` False leaves
    out the four methods of the extra [interactions]; None uses them when they are installed. True is returned when
    every check of the package passed, whatever the model made of the documents."""
    toy = model is None
    model = planted_toy() if toy else model
    name = model_name(model)
    scores, replies = can_score(model), can_generate(model)
    if interactions is None:
        interactions = all(find_spec(p) is not None for p in ("shapiq", "lightgbm"))
    groups = [ALONE, TOGETHER] if interactions else [ALONE]
    cache = cache if isinstance(cache, Cache) else Cache(cache)
    report = _Report(show)
    show(f"contextattr demo: {name}")
    if toy:
        show("  This is the built-in stand-in: fixed rules, no model and no weights. Its results are right by "
             "construction, so every line below should say ok. Give your own wrapper to learn about a real model.")
    if not interactions:
        show('  The four methods that score documents together are left out: pip install "contextattr[interactions]"')
    if not scores and not replies:
        report.line("the wrapper", False, "it has neither score nor generate")
    elif not scores:
        show("  This wrapper can only generate, so the first case alone is run, by counting replies (about 80 "
             "short replies).")
        show(f"\n== 1. {TITLE[1]} ==\n  Question: {QUESTION[1]}")
        _replies_only(model, cache, report)
    else:
        show("  Three cases of 8 documents each, every combination asked: 1,024 prompts to score" +
             (", and 5 short replies." if replies else "; this wrapper cannot generate, so no replies are read."))
        for k, run in ((1, _case_1), (2, _case_2), (3, _case_3)):
            show(f"\n== {k}. {TITLE[k]} ==\n  Question: {QUESTION[k]}")
            run(model, cache, groups, report, replies)
        again = explain(case(1, [Response("answer", "1932")]), model, LogProb(), method="kernelshap",
                        budget=2 ** 8, noise_repeats=0, cache=cache)
        show("")
        report.line("the cache: asked again, nothing is sent", again.requests["requests"] == 0,
                    f"{again.requests['reused']} answers reused")
    package = [passed for reading, passed in report.checks if not reading]
    reading = [passed for reading, passed in report.checks if reading]
    show(f"\n{sum(package)} of {len(package)} checks of the package passed" +
         ("." if all(package) else ": the lines marked FAIL. They hold whatever the model says. Run check_model on "
          "the wrapper first; if that passes, please report this output."))
    if reading:
        show(f"{sum(reading)} of {len(reading)} expectations about the model held" +
             ("." if all(reading) else ": the lines marked note. They say how this model read the documents, "
              "not that anything is wrong."))
    return all(package)
