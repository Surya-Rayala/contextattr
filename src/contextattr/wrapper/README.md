# wrapper/: what a user's wrapper must do ([docs/wrapper.md](../../../docs/wrapper.md))

The package ships no wrapper for any model or service: users write their own, and this folder says what it must do and checks that it does.

| File | Holds |
|---|---|
| `contract.py` | `Sampling`, `Scored`, the `Scorer` and `Generator` protocols; `call_score` and `call_generate`, which check the shape of every answer a wrapper gives |
| `check.py` | `check_model`: about a dozen calls on a built-in prompt, reporting each rule a wrapper keeps or breaks and the targets it can serve (`Sampled` needs only `generate`) |
| `toy.py` | `ToyModel`: a compatible wrapper with known answers and no model behind it; `ToyModel.demo()` models the interface's Rio example |
| `template.py` | a wrapper to fill in; `contextattr template my_wrapper.py` writes it out (it is a file to copy, not imported by the package) |
