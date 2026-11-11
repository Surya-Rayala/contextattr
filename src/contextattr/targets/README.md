# targets/: the number read for each masked prompt ([docs/interface.md](../../../docs/interface.md) §5)

| File | Holds |
|---|---|
| `scored.py` | the targets that need the wrapper's `score`: `LogProb` (one response, mean, sum or log-odds, or its span), `Pooled` (the responses with one label), `Contrast` (negatives − positives), `Choice` (log p(a) − log p(b)) |
| `sampled.py` | `Sampled`, which needs only `generate`: k fresh replies to each masked prompt, and the share a check flags or the mean of its numbers |

A target names the texts to score under every masked prompt (its items) and how their numbers combine into one value; `value.py` asks the wrapper and applies it.
