"""The run of a prompt file (interface §8): Run's plan and execute, resuming, refusing other settings, the
summary by tag, the HTML report, the command line on the example, and the examples themselves."""

import json
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from prompt_helpers import rio

import contextattr as ca
from contextattr import cli, report

EXAMPLES = Path(__file__).parent.parent / "examples"


def prompts():
    p = rio()
    good = ca.Response("good-1", "It is hot and humid: it is summer in Brazil.", label="positive")
    two = ca.Prompt("rio-02", messages=p.messages, responses=(*p.responses, good))
    return [p, two]


def test_plan_counts_before_any_call():
    class Silent(ca.ToyModel):
        def score(self, prompts, responses):
            raise AssertionError("plan must not call the model")

    plan = ca.Run(Silent.demo(), prompts(), method="kernelshap", budget=8).plan()
    rows = {r["prompt"]: r for r in plan["prompts"]}
    assert rows["rio-01"]["masks"] == 8 and rows["rio-01"]["requests"] == (8 + 3) * 1   # 2^3 masks, 3 noise repeats
    assert rows["rio-02"]["requests"] == (8 + 3) * 2                                    # Contrast: two texts a mask
    assert plan["total"]["requests"] == 33 and plan["measured"] is None
    measured = ca.Run(ca.ToyModel.demo(), prompts(), method="loo").plan(measure=5)["measured"]
    assert measured["requests"] >= 1 and measured["projected_seconds"] >= 0


def test_execute_resume_and_settings(tmp_path):
    out = tmp_path / "run"
    results = ca.Run(ca.ToyModel.demo(), prompts(), method="kernelshap", budget=8, removal=(1,)).execute(out)
    assert len(results) == 2 and {p.name for p in out.iterdir()} == {
        "run.json", "cache.sqlite", "explanations.jsonl", "summary.json", "report.html"}
    first, second = results.explanations
    assert first.target.startswith("Pooled") and second.target.startswith("Contrast")   # chosen by the labels
    assert first.scores["d3"] > 0 and second.scores["d2"] < 0
    lines = (out / "explanations.jsonl").read_text().splitlines()
    again = ca.Run(ca.ToyModel.demo(), prompts(), method="kernelshap", budget=8, removal=(1,)).execute(out)
    assert (out / "explanations.jsonl").read_text().splitlines() == lines and len(again) == 2   # nothing redone
    with pytest.raises(ValueError, match="other settings"):
        ca.Run(ca.ToyModel.demo(), prompts(), method="loo").execute(out)
    loaded = ca.Results.load(out).explanations[1]
    assert loaded.scores == pytest.approx(second.scores) and loaded.prompt == second.prompt
    assert loaded.removal == second.removal


def test_sampled_responses_and_the_summary(tmp_path):
    p = rio()
    bare = ca.Prompt("bare", messages=p.messages)
    run = ca.Run(ca.ToyModel.demo(), [bare], sample=ca.Sampling(n=8, temperature=0.7, seed=0),
                 check=ca.checks.regex(r"\bcold\b"), method="kernelshap", budget=8)
    results = run.execute(tmp_path / "s")
    (e,) = results.explanations
    assert e.target.startswith("Contrast") and len(e.prompt.responses) == 8
    summary = ca.Results([e]).summary(by="unit", top=1)
    assert summary["d3"]["share_in_top_1"] == 1.0 and summary["d1"]["prompts"] == 1
    with pytest.raises(ValueError):
        results.summary(by="nothing")


def test_report_html(tmp_path):
    e = ca.explain(rio(), ca.ToyModel.demo(), method="kernelshap", budget=8, removal=(1,))
    page = e.to_html(tmp_path / "one.html")
    assert (tmp_path / "one.html").read_text() == page
    assert page.startswith("<!doctype html>") and "<script" not in page and "http" not in page.split("<body>")[0]
    assert "title='d3: +0.2500'" in page and "Winter in Brazil" in page and "removing the top r" in page
    escaped = report.page([ca.explain(ca.Prompt("x", text=[ca.Unit("u", "<b>")]), value=lambda ps: [1.0] * len(ps),
                                      method="loo", noise_repeats=0)])
    assert "&lt;b&gt;" in escaped and "<b>" not in escaped.split("<body>")[1]


def test_cli_on_the_example(tmp_path, capsys):
    for name in ("rio.toml", "rio.jsonl"):
        shutil.copy(EXAMPLES / name, tmp_path / name)
    assert cli.main(["plan", str(tmp_path / "rio.toml")]) == 0
    assert json.loads(capsys.readouterr().out)["total"]["masks"] == 16
    assert cli.main(["run", str(tmp_path / "rio.toml")]) == 0
    assert "2 prompts explained" in capsys.readouterr().out
    assert cli.main(["report", str(tmp_path / "runs" / "rio")]) == 0
    assert set(json.loads(capsys.readouterr().out)) == {"doc", "instruction"}
    out = subprocess.run([sys.executable, "-m", "contextattr", "--help"], capture_output=True, text=True)
    assert out.returncode == 0 and "plan" in out.stdout


def test_cli_checks_its_config(tmp_path):
    def run_toml(text):
        (tmp_path / "c.toml").write_text(text)
        return cli.load(tmp_path / "c.toml")

    shutil.copy(EXAMPLES / "rio.jsonl", tmp_path / "rio.jsonl")
    base = 'prompts = "rio.jsonl"\nout = "o"\n[model]\nuse = "contextattr:ToyModel.demo"\n'
    run, out = run_toml(base + '[target]\nkind = "sampled"\nsamples = 3\n[check]\nregex = "cold"\n')
    assert out == tmp_path / "o" and run.target.samples == 3 and run.target.sampling_for(0).temperature == 0.7
    run, _ = run_toml(base + '[sample]\nn = 4\n[check]\nreference = "contains"\n[run]\nmethod = "loo"\n')
    assert run.sample.n == 4 and run.sample.temperature == 0.7 and run.method == "loo"
    (tmp_path / "w.py").write_text("class W:\n    def __init__(self, k):\n        self.name = f'w{k}'\n")
    assert cli.build({"use": "w.py:W", "args": {"k": 2}}, tmp_path).name == "w2"
    for bad in (base + "extra = 1\n", base + '[run]\nmethods = "loo"\n', base + '[target]\nkind = "nope"\n',
                base + '[target]\nkind = "sampled"\n', base.replace("ToyModel.demo", "Nothing"),
                'prompts = "rio.jsonl"\nout = "o"\n[model]\nuse = "no-colon"\n', base + "[check]\nother = 1\n"):
        with pytest.raises((ValueError, AttributeError)):
            run_toml(bad)


def test_the_examples_run(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["quickstart.py", str(tmp_path / "q.html")])
    runpy.run_path(str(EXAMPLES / "quickstart.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert "ready: every check passed" in out and "removing d3" in out and (tmp_path / "q.html").exists()


def test_the_template_command(tmp_path, capsys):
    """`contextattr template FILE` writes the wrapper to fill in and never over an existing file; without FILE it
    prints it. Unfilled, the template fails check_model on score and generate."""
    path = tmp_path / "my_wrapper.py"
    assert cli.main(["template", str(path)]) == 0
    template = runpy.run_path(str(path))
    report = ca.check_model(template["MyModel"]())
    assert not report.ok and {c.status for c in report.checks if c.name in ("score", "generate")} == {"FAIL"}
    path.write_text("mine")
    assert cli.main(["template", str(path)]) == 1 and path.read_text() == "mine"
    capsys.readouterr()
    assert cli.main(["template"]) == 0
    assert capsys.readouterr().out == (Path(cli.__file__).parent / "wrapper" / "template.py").read_text()
