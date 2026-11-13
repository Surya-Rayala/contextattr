# ContextAttr

Find out which parts of a prompt make a language model give an answer.

You split a prompt into parts: sentences, documents, words, or any pieces you choose. ContextAttr asks your model about many copies of the prompt with some parts removed. It then tells you how much each part, and each small group of parts working together, pushes the model towards an answer, and how far to trust that result. It works with any model whose log-probabilities you can read, wherever it runs: you connect yours with a short wrapper.

Version 0.1 is an early release: the interface may still change.

## What you need

| You bring | What it is |
|---|---|
| **A model** | an open-weight model that you run yourself or host on your own server, with a wrapper written for it (see below). ContextAttr reads the log-probabilities the model gives to answers, so the model must let you read them. To try the package first, a built-in stand-in needs no model |
| **Prompts** | the prompts your model gets, as chat messages or plain text |
| **A way to cut them into parts** | the pieces whose effect you want to know: documents, sentences, instructions, examples or words. You mark them yourself, or let `ca.split` cut a text |
| **Answers** | replies the model could give, each labelled `negative` (an answer you do not want) or `positive` (one you want). You write them, or the model writes them and a check labels them |

Which answers to give depends on what you want to improve:

| You give | You learn | What to explain |
|---|---|---|
| unwanted answers only | which parts make the model likely to give them: what to remove or rewrite | `ca.Pooled("negative")` |
| wanted answers only | which parts support them: what to keep | `ca.Pooled("positive")` |
| both | which parts move the model from the wanted answers to the unwanted ones. This is the clearest result: prefer it whenever you have both | `ca.Contrast()` |
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
    responses=[ca.Response("bad", "It is cold, because Carnival is in winter.", label="negative"),
               ca.Response("good", "It is hot and humid: it is summer in Brazil.", label="positive")])

model = ca.ToyModel.demo()        # a stand-in: put your own model's wrapper here
e = ca.explain(prompt, model, ca.Contrast(), method="kernelshap", budget=8)
print(e)
```

It prints:

```
rio · KernelSHAP(paired=True) · 8 masks · R² 0.991 on 128 test masks · noise 0 · 22 requests
  d3              +0.2500  Winter in Brazil runs from June to September.
  d1              -0.0234  Rio Carnival takes place in February or March.
  d2              -0.2854  Summer in Brazil runs from December to March.
```

**What the example says:**

| In the code | Means |
|---|---|
| `ca` | the package, under the short name all the docs use |
| `ca.Prompt("rio", messages=...)` | one prompt, named `rio`, as chat messages |
| `ca.Unit("d1", "...", sep="\n")` | a part that may be removed, named `d1`. `sep` is text that goes with it, here the line break. Everything else, like the question, always stays |
| `ca.Response("bad", "...", label="negative")` | an answer the model could give. `negative` marks the answer you do not want, `positive` the one you do |
| `model` | the wrapper around your model |
| `ca.Contrast()` | what to explain: how much more likely the model finds the unwanted answers than the wanted ones |
| `method="kernelshap"`, `budget=8` | how the scores are worked out, and how many copies of the prompt, each with some parts removed, the model is asked about |
| `e` | the result |

**How to read the output:**

- The first line names the prompt and the method. `8 masks` are the 8 copies of the prompt, each with some parts removed, that the model was asked about, and `22 requests` is what that cost. `R²` says how far to trust the result: close to 1, the scores describe the model well; close to 0, do not use them. `noise` is how much the model's number changes when it is asked the same thing again.
- Then each part with its score, highest first. A positive score pushes the model towards the unwanted answer: `d3`, the sentence about winter, is what makes this model say "cold". A negative score pushes it away: `d2` protects against it. A score near zero means the part hardly matters.
- `e.scores` gives the scores as a dictionary, and `e.to_html("report.html")` writes a page with each part shaded by its score.

**How the scores are computed:**

1. A *mask* is one choice of parts to keep. For each mask, ContextAttr builds the prompt with only those parts and reads one number from your model, the *value*. With `ca.Contrast()` it is the mean log-probability per token that the model gives the unwanted answers, minus the same for the wanted answers.
2. The method chooses which masks to ask about, and fits a simple formula to their values: a base number, plus one score for each part that is kept. Those scores are what is printed. Methods such as `spex` add a term for each set of parts that matters together.
3. The formula is then tested on test masks, drawn separately. On a long prompt these are masks the method did not ask about; on one this short they are the same 8 masks. R² says how well the formula gives the model's values there.

[Reading a result](https://github.com/Surya-Rayala/contextattr/blob/main/docs/results.md) shows every method's scores on this example, what they mean, and how far to trust them. [Methods and settings](https://github.com/Surya-Rayala/contextattr/blob/main/docs/methods.md), section 2, gives each method's computation in full.

## Use it with your own model

ContextAttr reaches your model through a small class that you write, a *wrapper*. [Writing a wrapper](https://github.com/Surya-Rayala/contextattr/blob/main/docs/wrapper.md) is the guide, with a complete wrapper for a model you run yourself and the recipe for a model behind a server.

1. **Get the file to fill in.** `contextattr template my_wrapper.py` writes it. You fill in:
   - `score`: says how likely the model finds an answer you give it: a log-probability for each token (the pieces of text a model reads). Any model whose log-probabilities you can read can do this, in your own code or behind a server.
   - `generate`, optional: writes replies to a prompt. It is for explaining the model's own answers.
2. **Check it.** `python my_wrapper.py` runs `ca.check_model` on it and says what, if anything, is wrong.
3. **Check the results.** `contextattr demo my_wrapper.py:MyModel` explains three prompts with planted facts, where the right result is known, and says whether your model's results agree. `contextattr demo` alone shows it on the stand-in.
4. **Use it.** Put your wrapper where the example has `model`.

## Choose what to explain, and how

- **Parts:** mark them with `ca.Unit`, as above, or cut a text automatically: `ca.split(text, "sentence", id="s")` (or `"word"`, `"line"`, `"paragraph"`).
- **Answer:** `ca.LogProb()` explains one answer. `ca.Contrast()` explains what moves the model from the answers you want to the ones you don't. `ca.Sampled(check)` counts how often the model's own replies do something: with the check `ca.checks.regex(r"\bcold\b")`, how often they say "cold".
- **Method:** `contextcite`, the default, takes a quick and cheap look, even at long prompts. `spex` finds parts that matter only together. `budget` sets how many copies of the prompt the model is asked about: more is steadier and costs more.
- **Many prompts:** `contextattr run run.toml` explains a whole file of prompts and writes a report.

## Notes for good results

- **Where it runs.** ContextAttr itself runs on the CPU and needs no GPU. Your model runs wherever your wrapper puts it: a CPU, a GPU or a server. The calls to the model are nearly all of the cost.
- **Read R² first.** When it is low, [Reading a result](https://github.com/Surya-Rayala/contextattr/blob/main/docs/results.md), section 6, says what that means and what to try.
- **Improve a prompt.** Give several wanted and several unwanted answers, find the part that pushes towards the unwanted ones, change it, and measure again. Section 9 of the same guide works through an example.
- **Pick the method by the size of the prompt.** [Methods and settings](https://github.com/Surya-Rayala/contextattr/blob/main/docs/methods.md), section 1, says what each method's score is and when to use it. Section 2 gives the computation: which copies of the prompt each method asks about, how it weighs them, and the formula it fits.

## Learn more

| Document | Covers |
|---|---|
| [docs/wrapper.md](https://github.com/Surya-Rayala/contextattr/blob/main/docs/wrapper.md) | connecting your model: what the model has to give, a complete wrapper, the recipe for a model behind a server, the rules, and what to do when the check fails |
| [docs/results.md](https://github.com/Surya-Rayala/contextattr/blob/main/docs/results.md) | reading a result, kept short: every method's scores on one example, what they mean, how far to trust them, and how to get from scores to a better prompt |
| [docs/methods.md](https://github.com/Surya-Rayala/contextattr/blob/main/docs/methods.md) | which method to use, how each one computes its scores (masks, weights and formulas, with a worked example), how many masks to give it, every setting with its default, and the references |
| [docs/interface.md](https://github.com/Surya-Rayala/contextattr/blob/main/docs/interface.md) | the reference: prompts and the prompt file, answers and labels, what can be explained, costs, runs and the command line |

## Licence

ContextAttr is released under the MIT licence: see [LICENSE](https://github.com/Surya-Rayala/contextattr/blob/main/LICENSE).
