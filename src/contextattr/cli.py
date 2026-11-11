"""The command line (docs/interface.md §8):

    contextattr plan   run.toml [--measure K]   masks, requests and tokens before any call (K real requests timed)
    contextattr run    run.toml                 explain every prompt into the run directory; resumes a stopped run
    contextattr report <run directory>          the HTML report again, and the summary by tag
    contextattr template [my_wrapper.py]        a wrapper to fill in (wrapper/template.py), written or printed

run.toml names the wrapper by import path, so one run file works wherever its wrapper does:

    prompts = "prompts.jsonl"               # paths are relative to this file
    out = "runs/rio"
    [model]
    use = "my_wrappers.py:MyModel"          # or "package.module:Class", or "contextattr:ToyModel.demo"
    args = { model_id = "…" }               # given to it as keyword arguments
    [run]                                   # optional: method, budget, test_masks, seed, replace, batch_size,
    method = "contextcite"                  #           noise_repeats, removal
    [target]                                # optional: kind = logprob | pooled | contrast | choice | sampled, with
    kind = "contrast"                       #           response, reduce, label, a, b, samples, temperature, top_p,
                                            #           max_tokens, seed
    [check]                                 # optional: regex = "…" (negative = true) or reference = "contains"
    [sample]                                # optional: n, temperature, top_p, max_tokens, seed
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import tomllib
from pathlib import Path

from .responses import reference, regex
from .run import Results, Run
from .targets import Choice, Contrast, LogProb, Pooled, Sampled
from .wrapper.contract import Sampling

TOP = {"prompts", "out", "model", "run", "target", "check", "sample"}
RUN = {"method", "budget", "test_masks", "seed", "replace", "batch_size", "noise_repeats", "removal"}
SAMPLING = {"n", "temperature", "top_p", "max_tokens", "seed", "stop"}


def load(path) -> tuple[Run, Path]:
    """The Run a run.toml describes, and its run directory."""
    path = Path(path)
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    _keys(cfg, TOP, {"prompts", "out", "model"}, path.name)
    base = path.parent
    model = build(cfg["model"], base)
    check = None
    if "check" in cfg:
        c = dict(cfg["check"])
        if "regex" in c:
            _keys(c, {"regex", "negative"}, {"regex"}, "[check]")
            negative = c.get("negative", True)
            if not isinstance(c["regex"], str) or not isinstance(negative, bool):
                raise ValueError("[check]: regex must be a string, and negative true or false")
            check = regex(c["regex"], negative=negative)
        elif "reference" in c:
            _keys(c, {"reference"}, {"reference"}, "[check]")
            if c["reference"] not in ("contains", "exact") or not isinstance(c["reference"], str):
                raise ValueError(f"[check]: reference must be \"contains\" or \"exact\", not {c['reference']!r}")
            check = reference(contains=c["reference"] == "contains")
        else:
            raise ValueError("[check] needs regex = \"…\" or reference = \"contains\" | \"exact\"")
    sample = None
    if "sample" in cfg:
        _keys(cfg["sample"], SAMPLING, set(), "[sample]")
        sample = Sampling(**{**{"n": 8, "temperature": 0.7, "top_p": 0.95}, **cfg["sample"]})
    target = _target(cfg.get("target"), check)
    run_args = dict(cfg.get("run", {}))
    _keys(run_args, RUN, set(), "[run]")
    if "removal" in run_args:
        run_args["removal"] = tuple(run_args["removal"])
    run = Run(model, base / cfg["prompts"], target=target, sample=sample, check=check, **run_args)
    return run, base / cfg["out"]


def build(spec: dict, base: Path = Path(".")):
    """The wrapper `use` names ("module:attr" or "file.py:attr", attr may be dotted), called with `args`."""
    _keys(spec, {"use", "args"}, {"use"}, "[model]")
    where, _, attr = spec["use"].partition(":")
    if not attr:
        raise ValueError(f"[model] use must be \"module:attr\" or \"file.py:attr\", not {spec['use']!r}")
    if where.endswith(".py"):
        file = (base / where).resolve()
        loader = importlib.util.spec_from_file_location(file.stem, file)
        if loader is None:
            raise ValueError(f"cannot load {file}")
        module = importlib.util.module_from_spec(loader)
        sys.modules.setdefault(file.stem, module)
        loader.loader.exec_module(module)
    else:
        module = importlib.import_module(where)
    obj = module
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj(**spec.get("args", {}))


def _target(spec, check):
    if spec is None:
        return None
    spec = dict(spec)
    kind = spec.pop("kind", None)
    if kind == "logprob":
        _keys(spec, {"response", "reduce"}, set(), "[target]")
        return LogProb(spec.get("response"), reduce=spec.get("reduce", "mean"))
    if kind == "pooled":
        _keys(spec, {"label", "reduce"}, set(), "[target]")
        return Pooled(spec.get("label", "negative"), reduce=spec.get("reduce", "mean"))
    if kind == "contrast":
        _keys(spec, {"reduce"}, set(), "[target]")
        return Contrast(reduce=spec.get("reduce", "mean"))
    if kind == "choice":
        _keys(spec, {"a", "b"}, {"a", "b"}, "[target]")
        return Choice(spec["a"], spec["b"])
    if kind == "sampled":
        if check is None:
            raise ValueError("a sampled target needs a [check]")
        _keys(spec, {"samples"} | SAMPLING - {"n"}, set(), "[target]")
        samples = spec.pop("samples", 4)
        sampling = Sampling(**{**{"temperature": 0.7, "top_p": 0.95}, **spec, "n": samples}) if spec else None
        return Sampled(check, samples=samples, sampling=sampling)
    raise ValueError(f"[target] kind must be logprob, pooled, contrast, choice or sampled, not {kind!r}")


def _keys(d: dict, allowed: set, required: set, where: str) -> None:
    unknown, missing = sorted(set(d) - allowed), sorted(required - set(d))
    if unknown:
        raise ValueError(f"{where}: unknown key {', '.join(unknown)} (allowed: {', '.join(sorted(allowed))})")
    if missing:
        raise ValueError(f"{where}: missing {', '.join(missing)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="contextattr", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="masks, requests and tokens before any call")
    plan.add_argument("config")
    plan.add_argument("--measure", type=int, default=0, help="time this many real requests of the first prompt")
    run = sub.add_parser("run", help="explain every prompt into the run directory")
    run.add_argument("config")
    report = sub.add_parser("report", help="the HTML report and the summary of a run directory")
    report.add_argument("directory")
    template = sub.add_parser("template", help="a wrapper to fill in around your model: written to FILE, or printed")
    template.add_argument("file", nargs="?")
    args = ap.parse_args(argv)
    if args.command == "template":
        from importlib.resources import files  # noqa: PLC0415

        text = files("contextattr.wrapper").joinpath("template.py").read_text(encoding="utf-8")
        if args.file is None:
            print(text, end="")
            return 0
        try:
            with open(args.file, "x", encoding="utf-8") as f:    # "x": an existing file is never overwritten
                f.write(text)
        except FileExistsError:
            print(f"contextattr template: {args.file} exists; give another name", file=sys.stderr)
            return 1
        print(f"wrote {args.file}: fill in score and/or generate, then run it (python {args.file}) to check it")
        return 0
    if args.command == "plan":
        r, _ = load(args.config)
        print(json.dumps(r.plan(measure=args.measure), indent=2))
    elif args.command == "run":
        r, out = load(args.config)
        results = r.execute(out)
        print(f"{len(results)} prompts explained in {out}; report: {out / 'report.html'}")
    else:
        results = Results.load(args.directory)
        results.to_html(Path(args.directory) / "report.html")
        print(json.dumps(results.summary(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
