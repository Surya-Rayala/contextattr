# tests/: CPU tests with exact answers

Run from the repository or the unpacked source archive, after `pip install -e ".[test]"`: `python -m pytest`. Hypothesis properties run derandomized, the same 300 examples each time; `CA_HYPOTHESIS_PROFILE=deep python -m pytest` draws 20,000 fresh ones per property. No test here needs a model; how the package was checked on real models is in `docs/methods.md`.

| File | Checks |
|---|---|
| `test_package.py` | the package imports no model library, names nothing of Anvil, has a README in every folder, its version matches `pyproject.toml`, and the README's examples run |
| `test_prompt_units.py` | building prompts, units, groups and responses, and every check on them |
| `test_prompt_split.py` | each splitter's rules on worked examples; on any text, the units give the text back exactly |
| `test_prompt_render.py` | masks rendered exactly: deletion with separators, replacement, groups; random prompts and masks against a reference renderer |
| `test_prompt_io.py` | `prompts.jsonl`: the interface's example, write-read round trips, the line and place of every error |
| `test_wrapper_contract.py` | `Sampling` and `Scored`, and the checks on every answer a wrapper gives |
| `test_wrapper_toy.py` | `ToyModel`'s answers against its formula worked by hand; its token texts and replies |
| `test_wrapper_check.py` | `check_model` on the toy model and on wrappers that each break one rule |
| `test_targets_scored.py` | which texts each target scores and how they combine; mean, sum, log-odds; spans |
| `test_cache.py` | stable keys, exact answers, the first answer kept, persistence, many keys |
| `test_value.py` | values equal the toy model's formula; each distinct request asked once, in batches; resuming after a failure; a user's function; noise; every check; random prompts and masks |
| `test_methods.py` | every method against exact answers: closed forms from Möbius coefficients and brute force; exact paths, axioms, sampled convergence, KernelSHAP's design and its exactness on pairwise games |
| `test_explain.py` | `explain` and `attribute` end to end on the toy model: known scores, shared held-out masks, requests, noise, the result's views |
| `test_interactions.py` | the indices of App. C from a surrogate, top-r removal against brute force, Recovery@r, interaction surrogates, the shapiq adapter (with a fake shapiq) |
| `test_shapiq_methods.py` | marked `shapiq`, run with `CA_TEST_SHAPIQ=1` (5–7 minutes): Faith indices exact from every mask, SPEX recovering planted interactions exactly, ProxySPEX close, the indices equal to shapiq's exact ones |
| `test_sampled.py` | checks, sampled responses, `Sampled` against the toy model's known rates, its noise from fresh seeds, `intervene` |
| `test_run.py` | `Run`'s plan and execute, resuming, refusing other settings, the summary, the HTML report, the command line on the example, the examples themselves, `contextattr template` |
| `prompt_helpers.py`, `conftest.py` | the example prompt, Hypothesis strategies and settings |
