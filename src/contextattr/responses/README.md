# responses/: responses and labels ([docs/interface.md](../../../docs/interface.md) §4)

| File | Holds |
|---|---|
| `checks.py` | `Verdict` (a label or a number, and the span that matters), `verdict` (any check's answer as one), two checks: `regex` (a match is negative, or positive, with its span) and `reference` (against the prompt's reference, normalised), and `describe` (a check's name in a run's settings: its code and the values it holds) |
| `sample.py` | `sample`: k replies of the model to the full prompt, added to its responses as `sample-0`, `sample-1`, … and labelled by a check; cached |

A check is any function `check(prompt, response)` giving a `Verdict`, a label, `True` (negative) / `False` (positive), or a number; a judge model is a check of the same shape. The package exposes this folder as `ca.checks` too.
