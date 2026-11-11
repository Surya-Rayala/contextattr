# examples/

| File | Shows |
|---|---|
| `quickstart.py` | the Rio prompt explained through `ToyModel.demo()`: `check_model`, `explain` with `Contrast`, the removal check and an intervention; runs anywhere in seconds (`python examples/quickstart.py report.html`) |
| `rio.toml`, `rio.jsonl` | the command line on two prompts: `python -m contextattr plan examples/rio.toml`, then `run`, then `report examples/runs/rio` |

`contextattr template my_wrapper.py` writes a wrapper to fill in around your own model or service; running it runs `check_model` on it. The guide, with two complete wrappers to copy from, is [docs/wrapper.md](../docs/wrapper.md).
