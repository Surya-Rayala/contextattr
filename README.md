# ContextAttr

Find out which parts of a prompt make a language model give an answer.

You split a prompt into parts: sentences, documents, words, or any pieces you choose. ContextAttr asks your model about many copies of the prompt with some parts removed. It then tells you how much each part, and each small group of parts working together, pushes the model towards an answer, and how far to trust that result. It works with any model, wherever it runs: you connect yours with a short wrapper.

Version 0.1.0 is an early release: the interface may still change.

## Install

```
pip install contextattr
```

Add `[interactions]` (`pip install "contextattr[interactions]"`) for SPEX and the other methods that find parts which matter only together. With uv: `uv add contextattr`. Needs Python 3.12 or later.

## Try it

This runs anywhere in seconds: `ToyModel` stands in for a real model.

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

- The first line names the prompt and the method and says how far to trust the result. R² close to 1 means the scores describe the model well; close to 0, do not use them.
- Then each part with its score, highest first. A positive score pushes the model towards the unwanted answer: `d3`, the sentence about winter, is what makes this model say "cold". A negative score pushes it away: `d2` protects against it. A score near zero means the part hardly matters.
- `e.scores` gives the scores as a dictionary, and `e.to_html("report.html")` writes a page with each part shaded by its score.

[Reading a result](https://github.com/Surya-Rayala/contextattr/blob/main/docs/results.md) shows the output of every method on this example and what its scores mean.

## Use it with your own model

ContextAttr reaches your model through a small class that you write, a *wrapper*. [Writing a wrapper](https://github.com/Surya-Rayala/contextattr/blob/main/docs/wrapper.md) is the guide, with two complete wrappers to copy from: one for an API and one for a model you run yourself.

1. **Get the file to fill in.** `contextattr template my_wrapper.py` writes it. It needs one or both of:
   - `generate`: writes replies to a prompt. Every model can do this, including the OpenAI, Gemini and Anthropic APIs.
   - `score`: gives the log-probability of each token of a given answer. Possible when you run the model yourself, in your own code or on a vLLM or SGLang server.
2. **Check it.** `python my_wrapper.py` runs `ca.check_model` on it and says what, if anything, is wrong.
3. **Use it.** Put your wrapper where the example has `model`.

## Choose what to explain, and how

- **Parts:** mark them with `ca.Unit`, as above, or cut a text automatically: `ca.split(text, "sentence", id="s")` (or `"word"`, `"line"`, `"paragraph"`).
- **Answer:** `ca.LogProb()` explains one answer. `ca.Contrast()` explains what moves the model from the answers you want to the ones you don't. `ca.Sampled(check)` works when your model can only generate text.
- **Method:** `contextcite`, the default, takes a quick and cheap look, even at long prompts. `spex` finds parts that matter only together. `budget` sets how many copies of the prompt the model is asked about: more is steadier and costs more.
- **Many prompts:** `contextattr run run.toml` explains a whole file of prompts and writes a report.

## Learn more

| Document | Covers |
|---|---|
| [docs/wrapper.md](https://github.com/Surya-Rayala/contextattr/blob/main/docs/wrapper.md) | connecting your model: what to write, two complete wrappers, the rules, and what to do when the check fails |
| [docs/results.md](https://github.com/Surya-Rayala/contextattr/blob/main/docs/results.md) | reading a result: the output of every method on one example, what its scores mean, and how far to trust them |
| [docs/methods.md](https://github.com/Surya-Rayala/contextattr/blob/main/docs/methods.md) | which method to use, how many masks to give it, and every setting with its default |
| [docs/interface.md](https://github.com/Surya-Rayala/contextattr/blob/main/docs/interface.md) | the reference: prompts and the prompt file, answers and labels, what can be explained, costs, runs and the command line |

## Licence

ContextAttr is released under the MIT licence: see [LICENSE](https://github.com/Surya-Rayala/contextattr/blob/main/LICENSE).
