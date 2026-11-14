# ContextAttr

Find out which parts of a prompt make a language model give an answer.

You split a prompt into parts: sentences, documents, words, or any pieces you choose. ContextAttr asks your model about many copies of the prompt with some parts removed. It then tells you how much each part, and each small group of parts working together, pushes the model towards an answer, and how far to trust that result. It works with any model whose log-probabilities you can read, wherever it runs: you connect yours with a short wrapper.

Version 0.2 is an early release: the interface may still change.

## What you need

| You bring | What it is |
|---|---|
| **A model** | an open-weight model that you run yourself or host on your own server, with a wrapper written for it (see below). ContextAttr reads the log-probabilities the model gives to answers (how likely it finds each piece of an answer), so the model must let you read them |
| **Prompts** | the prompts your model gets, as chat messages or plain text |
| **A way to cut them into parts** | the pieces whose effect you want to know: documents, sentences, instructions, examples or words. You mark them yourself, or let `ca.split` cut a text |
| **Answers** | replies the model could give, each labelled `negative` (an answer you do not want) or `positive` (one you want). You write them, or the model writes them and a check labels them |

Which answers to give depends on what you want to improve ([Reading a result](https://github.com/Surya-Rayala/contextattr/blob/main/docs/results.md), section 4, shows all four on one case):

| You give | You learn | What to explain |
|---|---|---|
| unwanted answers only | which parts make the model likely to give them: what to remove or rewrite | `ca.Pooled("negative")` |
| wanted answers only | which parts support them: what to keep | `ca.Pooled("positive")` |
| both | which parts push the model towards the wanted answers, and which towards the unwanted ones. This is the clearest result: prefer it whenever you have both | `ca.Contrast()` |
| one answer, with no label | which parts support that answer | `ca.LogProb()` |

## Install

```
pip install contextattr
```

Add `[interactions]` (`pip install "contextattr[interactions]"`) for SPEX and the other methods that find parts which matter only together. With uv: `uv add contextattr`. Needs Python 3.12 or later.

## Try it

This runs anywhere in seconds. `ToyModel` stands in for a real model: it follows a few fixed rules written for this example, so its results are known in advance. It is not a language model.

```python
import contextattr as ca

prompt = ca.Prompt("rio", messages=[{"role": "user", "content": [
    ca.Unit("d1", "Rio Carnival takes place in February or March.", sep="\n"),
    ca.Unit("d2", "Summer in Brazil runs from December to March.", sep="\n"),
    ca.Unit("d3", "Winter in Brazil runs from June to September.", sep="\n"),
    "Question: What is the weather like during Rio Carnival?"]}],
    responses=[ca.Response("answer", "It is hot and humid: it is summer in Brazil.")])

model = ca.ToyModel.demo()        # a stand-in: put your own model's wrapper here
e = ca.explain(prompt, model, ca.LogProb(), method="kernelshap", budget=8)
print(e)
```

It prints:

```
rio · KernelSHAP(paired=True) · 8 masks · R² 0.985 on 128 test masks · noise 0 · 11 requests
  above 0: the unit makes the answer more likely · below 0: less likely
  d2              +0.2854  Summer in Brazil runs from December to March.
  d1              +0.0234  Rio Carnival takes place in February or March.
  d3              +0.0000  Winter in Brazil runs from June to September.
```

**What the example says:**

| In the code | Means |
|---|---|
| `ca` | the package, under the short name all the docs use |
| `ca.Prompt("rio", messages=...)` | one prompt, named `rio`, as chat messages |
| `ca.Unit("d1", "...", sep="\n")` | a part that may be removed, named `d1`. `sep` is text that goes with it, here the line break. Everything else, like the question, always stays |
| `ca.Response("answer", "...")` | the answer to explain: here the right one |
| `model` | the wrapper around your model |
| `ca.LogProb()` | what to explain: how likely the model finds that answer |
| `method="kernelshap"`, `budget=8` | how the scores are worked out, and how many copies of the prompt, each with some parts removed, the model is asked about |
| `e` | the result |

**How to read the output:**

- The first line names the prompt and the method. `8 masks` are the 8 copies of the prompt, each with some parts removed, that the model was asked about, and `11 requests` is what that cost. `R²` says how far to trust the result: close to 1, the scores describe the model well; close to 0, do not use them. `noise` is how much the model's number changes when it is asked the same thing again.
- The second line says how to read the scores. The rule is the same for every result:
  - **above 0**: the part pushes the model towards the answer. `d2`, the sentence about summer, is what makes this model say "hot";
  - **below 0**: the part pushes the model away from the answer;
  - **near 0**: the part hardly matters. `d3`, the sentence about winter, does nothing for this answer.
- `e.scores` gives the scores as a dictionary, and `e.to_html("report.html")` writes a page with each part shaded by its score.

**How the scores are computed:**

1. A *mask* is one choice of parts to keep. For each mask, ContextAttr builds the prompt with only those parts and reads one number from your model, the *value*. With `ca.LogProb()` it is how likely the model finds the answer: its mean log-probability per token.
2. The method chooses which masks to ask about, and fits a simple formula to their values: a base number, plus one score for each part that is kept. Those scores are what is printed. Methods such as `spex` add a term for each set of parts that matters together.
3. The formula is then tested on a separate draw of masks. R² says how well it gives the model's values there.

[Reading a result](https://github.com/Surya-Rayala/contextattr/blob/main/docs/results.md) shows how to read a result, on the three cases of `contextattr demo`: one part holds the answer, two parts hold it together, and two parts disagree.

## Use it with your own model

ContextAttr reaches your model through a small class that you write, a *wrapper*. [Writing a wrapper](https://github.com/Surya-Rayala/contextattr/blob/main/docs/wrapper.md) is the guide, with a complete wrapper for a model on your own GPU.

1. **Get the file to fill in.** `contextattr template my_wrapper.py` writes it. You fill in:
   - `score`: says how likely the model finds an answer you give it: a log-probability for each token (the pieces of text a model reads).
   - `generate`, optional: writes replies to a prompt. It is for explaining the model's own answers.
2. **Check it.** `python my_wrapper.py` runs `ca.check_model` on it and says what, if anything, is wrong.
3. **Check the results.** `contextattr demo my_wrapper.py:MyModel` explains three prompts with planted facts, where the right result is known, and says whether your model's results agree. `contextattr demo` alone shows it on the stand-in.
4. **Use it.** Put your wrapper where the example has `model`.

## Choose what to explain, and how

- **Parts:** mark them with `ca.Unit`, as above, or cut a text automatically: `ca.split(text, "sentence", id="s")` (or `"word"`, `"line"`, `"paragraph"`, or at a delimiter of your own: [the reference](https://github.com/Surya-Rayala/contextattr/blob/main/docs/interface.md), section 2).
- **Answer:** `ca.LogProb()` explains one answer. `ca.Contrast()` explains the answers you want against the ones you don't. `ca.Sampled(check)` counts how often the model's own replies do something: with the check `ca.checks.regex(r"\bcold\b")`, how often they say "cold".
- **Method:** `contextcite`, the default, takes a quick and cheap look, even at long prompts. `spex` finds parts that matter only together. `budget` sets how many copies of the prompt the model is asked about: more is steadier and costs more.
- **Many prompts:** `contextattr run run.toml` explains a whole file of prompts and writes a report.

## Notes for good results

- **Where it runs.** ContextAttr itself runs on the CPU and needs no GPU. Your model runs wherever your wrapper puts it: a CPU, a GPU or a server. The calls to the model are nearly all of the cost.
- **Read R² first.** When it is low, [Reading a result](https://github.com/Surya-Rayala/contextattr/blob/main/docs/results.md), section 5, says what that means and what to try.
- **Improve a prompt.** Give several wanted and several unwanted answers, find the part that pushes towards the unwanted ones, change it, and measure again. Section 8 of the same guide works through an example.
- **Pick the method by the size of the prompt.** [Methods and settings](https://github.com/Surya-Rayala/contextattr/blob/main/docs/methods.md), section 1, says what each method's score is and when to use it. Section 2 gives the computation: which copies of the prompt each method asks about, how it weighs them, and the formula it fits.

## Learn more

| Document | Covers |
|---|---|
| [docs/wrapper.md](https://github.com/Surya-Rayala/contextattr/blob/main/docs/wrapper.md) | connecting your model: what a wrapper takes and returns, a complete example for a model on your GPU, and how to check it |
| [docs/results.md](https://github.com/Surya-Rayala/contextattr/blob/main/docs/results.md) | reading a result, on three cases with known answers: what a score and its sign mean, parts that matter together, how far to trust a result, and how to get from scores to a better prompt |
| [docs/methods.md](https://github.com/Surya-Rayala/contextattr/blob/main/docs/methods.md) | which method to use, how each one computes its scores (masks, weights and formulas, with a worked example), how many masks to give it, every setting with its default, and the references |
| [docs/interface.md](https://github.com/Surya-Rayala/contextattr/blob/main/docs/interface.md) | the reference: prompts and the prompt file, answers and labels, what can be explained, costs, runs and the command line |

## Licence

ContextAttr is released under the MIT licence: see [LICENSE](https://github.com/Surya-Rayala/contextattr/blob/main/LICENSE).
