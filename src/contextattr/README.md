# src/contextattr: the package

One folder or file per role. The package reaches a model only through a wrapper its user writes ([docs/wrapper.md](../../docs/wrapper.md)) and imports no model library, which `tests/test_package.py` checks.

| Role | Holds |
|---|---|
| [`prompt/`](prompt/README.md) | the input: prompts, units, splitting, masking, `prompts.jsonl` |
| [`wrapper/`](wrapper/README.md) | what a user's wrapper must do, `check_model`, and `ToyModel` |
| [`targets/`](targets/README.md) | the number read for each masked prompt: `LogProb`, `Pooled`, `Contrast`, `Choice`; `Sampled`, which needs only generate |
| [`responses/`](responses/README.md) | checks that label a response (`regex`, `reference`, any function) and replies sampled from the model |
| `value.py` | `Value`: masks → numbers, each distinct request asked once, in batches, cached; or a user's own function; the noise check |
| `cache.py` | `Cache`: every answer stored once in one SQLite file (or memory), so a run resumes |
| [`methods/`](methods/README.md) | which masks and which fit: `LeaveOneOut`, `Banzhaf`, `LIME`, `ContextCite`, `KernelSHAP`; `FaithBanzhaf`, `FaithShap`, `SPEX`, `ProxySPEX` through shapiq, and the indices of App. C |
| `evaluate.py` | the faithfulness R² on held-out masks (SPEX eq. 7) and those masks; top-r removal (eq. 8) and Recovery@r (eq. 9); `intervene`, removing units and sampling again |
| `result.py` | `Explanation`: the scores, R², noise and costs; `top`, `table`, `as_dict`, `from_dict`, `to_html`; Shapley, Banzhaf and Shapley-Taylor read from its surrogate |
| `run.py` | `explain` (a prompt through a model and a target, or a function of masked prompts), `attribute` (any function of masks), and `Run` (a prompt file: `plan`, `execute` into a run directory, resumable) with `Results` (`summary` by tag) |
| `report.py` | one self-contained HTML page: each prompt with its units shaded by score, the scores, interactions and removal |
| `cli.py`, `__main__.py` | `contextattr plan | run | report` (also `python -m contextattr`), from a TOML file naming the wrapper by import path |
| `__init__.py` | the public names |
