# prompt/: the input ([docs/interface.md](../../../docs/interface.md) §2)

| File | Holds |
|---|---|
| `units.py` | `Prompt`, `Message`, `Unit`, `Group`, `Response` and their checks: unique ids, at least one unit, labels, spans |
| `split.py` | `split`: cuts a text into units by word, sentence, line, paragraph or a function returning spans, keeping the text exactly |
| `render.py` | a mask → the prompt as the model sees it, chat messages or text; removed units deleted or replaced |
| `io.py` | reading and writing `prompts.jsonl`, naming the line and place of any error |

A mask has one entry per unit, in the order of `Prompt.units` (message by message, a group's units in place): 1 keeps the unit, 0 removes it. `Prompt.mask(keep=…)` or `Prompt.mask(drop=…)` builds one from unit or group ids.
