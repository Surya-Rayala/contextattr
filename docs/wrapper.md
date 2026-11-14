# Writing a wrapper

ContextAttr never talks to a model itself. It hands prompts to a small class that you write, the *wrapper*, and your class asks the model. Where the model runs, and which software runs it, is yours to choose: a wrapper is right when what it takes and what it returns match this page. `ca` is the package: `import contextattr as ca`.

## 1. The wrapper class

A wrapper is a Python class with three members:

```python
class MyModel:
    name = "my-model"

    def score(self, prompts, responses):
        ...

    def generate(self, prompts, sampling):
        ...
```

| Member | Needed | What it is for |
|---|---|---|
| `name` | yes | a `str` that names the model. Every answer is stored under it, so change it whenever the answers could change: another model, version or precision |
| `score` | yes | says how likely the model finds an answer you give it. This does the main work: ContextAttr calls it for many copies of the prompt, each with some parts removed, and a part's score comes from how that likelihood changes. The model writes nothing |
| `generate` | optional | lets the model write its own replies. `ca.sample` collects them so that you can explain what the model really says, and `ca.intervene` uses them to confirm a result on the model's behaviour |

Two words: a *token* is the piece of text a model reads at a time, about a word. A *log-probability* says how likely the model finds it: 0 is certain, and the more negative, the less likely.

Any class whose functions take and return what is written below is a valid wrapper. ContextAttr never looks inside it, so how you get the numbers, and from where, is up to you. `check_model` (section 3) then tests that the formats are right and that the numbers behave as log-probabilities should.

### A prompt

Both functions get a list of prompts. Each prompt arrives ready to send, with the parts under test already removed, in one of two forms:

| Form | Type | Example |
|---|---|---|
| chat messages | `list[dict]`; each `dict` has `"role"` and `"content"`, both `str` | `[{"role": "system", "content": "Answer in one sentence."}, {"role": "user", "content": "Documents: ... Question: ..."}]` |
| plain text | `str` | `"Documents: ... Question: ..."` |

### `score(prompts, responses)`

| | Type | Example, for two prompts |
|---|---|---|
| `prompts` | `list` of prompts | `[prompt_1, prompt_2]` |
| `responses` | `list[str]`: one answer for each prompt | `["It is hot.", "It is cold."]` |
| returns | `list[ca.Scored]`: one for each prompt, in the same order | `[ca.Scored([-0.21, -1.35, -0.08]), ca.Scored([-2.30, -0.92, -1.10])]` |

`ca.Scored(logprobs)` holds a `list[float]`: one number for each token of the response. Each number is the natural-log probability the model gives that token, after the prompt and the tokens before it, so each is 0 or below. To get them:

1. Write the prompt in the model's own chat format, up to where its reply begins.
2. Put the response there.
3. Run the model once over the whole text. Nothing is generated.
4. For each token of the response, keep the log-probability the model gave it. A model's raw output is a score for every possible token, called a logit; `log_softmax` turns those into log-probabilities.
5. Leave out the marker that ends the reply.

### `generate(prompts, sampling)`

| | Type | Example, for two prompts |
|---|---|---|
| `prompts` | `list` of prompts | `[prompt_1, prompt_2]` |
| `sampling` | `ca.Sampling` | `ca.Sampling(n=2, temperature=0.7)` |
| returns | `list[list[str]]`: for each prompt, in the same order, exactly `sampling.n` replies | `[["It is hot.", "Hot and humid."], ["It is hot.", "Warm."]]` |

A reply is the text the model wrote and nothing else. The fields of `sampling`:

| Field | Type | Meaning |
|---|---|---|
| `n` | `int` | replies wanted for each prompt. You must return exactly this many |
| `temperature` | `float` | how random the replies are; 0 asks for the most likely reply |
| `top_p` | `float` | sample only among the most likely tokens that together have this probability |
| `max_tokens` | `int` | the longest reply |
| `seed` | `int` or `None` | the same seed should give the same replies |
| `stop` | `tuple` of `str` | strings that end a reply |

Pass the fields other than `n` on to your model where it supports them.

## 2. An example: your own model on a GPU

A complete wrapper for a Hugging Face model, run with transformers on your GPU, or on the CPU when there is none. Save it as `my_model.py`, put in your model's name, and run the file.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import contextattr as ca


class LocalModel:
    def __init__(self, path, device=None, **model_options):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")     # the GPU when there is one
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_options).to(device).eval()
        self.name = f"{path}:{self.model.dtype}"     # the precision changes the answers, so it is in the name

    def prompt_ids(self, prompt):
        """The prompt's tokens, up to where the model's reply begins."""
        if isinstance(prompt, str):                  # a plain-text prompt
            return self.tokenizer(prompt).input_ids
        text = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        return self.tokenizer(text, add_special_tokens=False).input_ids

    @torch.no_grad()
    def score(self, prompts, responses):
        out = []
        for prompt, response in zip(prompts, responses):
            before = self.prompt_ids(prompt)
            if not before:
                raise ValueError("an empty prompt: nothing comes before the response")
            reply = self.tokenizer(response, add_special_tokens=False).input_ids
            ids = torch.tensor([before + reply], device=self.model.device)
            logprobs = torch.log_softmax(self.model(ids).logits[0].float(), dim=-1)
            # the model predicts each token at the position before it
            out.append(ca.Scored([logprobs[len(before) + k - 1, token].item() for k, token in enumerate(reply)]))
        return out

    @torch.no_grad()
    def generate(self, prompts, sampling):
        out = []
        for prompt in prompts:
            ids = torch.tensor([self.prompt_ids(prompt)], device=self.model.device)
            if sampling.seed is not None:
                torch.manual_seed(sampling.seed)
            if sampling.temperature > 0:
                options = dict(do_sample=True, temperature=sampling.temperature, top_p=sampling.top_p, top_k=0,
                               num_return_sequences=sampling.n)
            else:
                options = dict(do_sample=False)      # greedy: one reply, repeated below
            new = self.model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=sampling.max_tokens,
                                      pad_token_id=self.tokenizer.eos_token_id, repetition_penalty=1.0, **options)[:, ids.shape[1]:]
            replies = self.tokenizer.batch_decode(new, skip_special_tokens=True)
            for stop in sampling.stop:               # a reply ends before a stop string
                replies = [reply.split(stop)[0] for reply in replies]
            out.append(replies if sampling.temperature > 0 else replies * sampling.n)
        return out


if __name__ == "__main__":
    print(ca.check_model(LocalModel("path/or/hub-id/of/your/model", dtype=torch.bfloat16)))
```

- **Precision.** `dtype=torch.bfloat16` halves the memory on a GPU. The wrapper's `name` carries the precision, because it changes the numbers a little.
- **Speed.** This version scores one prompt at a time, which is the simplest way to be correct. To go faster, score a batch of prompts in one pass.
- **A model behind a server.** The class stays the same; `score` and `generate` send the same text to your server and read the same numbers back from its answer.

## 3. Check the wrapper

`contextattr template my_wrapper.py` writes an empty wrapper to fill in, if you do not start from the example. Running the file calls `ca.check_model`, which asks your wrapper about a dozen small questions and prints one line for each. For Llama-3.2-3B-Instruct on one GPU:

```
check_model: my-model
  ok    name                  'my-model'
  ok    methods               score, generate
  ok    generate              2 greedy replies to each of 2 prompts, e.g. 'During Rio Carnival, the weather is typically warm and sunny'
  ok    greedy repeats        identical
  ok    seed                  honoured
  ok    score                 31 tokens, finite log-probabilities at most 0
  ok    score repeats         identical
  ok    token texts           join back to the response: targets can use spans
  ok    prompt matters        -0.272 per token with every unit, -0.978 with none
  ok    reply above reversed  -0.272 per token, -5.308 with its words reversed
  info  time                  0.935 s per request over 8 calls
  targets: LogProb, Pooled, Contrast, Choice, Sampled (spans: yes)
  ready: every check passed
```

| A line starts with | It means |
|---|---|
| `ok` | the wrapper does this correctly |
| `note` | allowed, but worth knowing: for example, the model ignores seeds, so two runs sample different replies |
| `FAIL` | the wrapper is not ready: the line says what went wrong |
| `info` | a measurement, such as the seconds per request |

The example of section 2 gets a `note` on `token texts`, because it returns none. That is fine: they are needed only to score a part of an answer.

When a line says `FAIL`:

| The line says | What to fix |
|---|---|
| `name`: a wrapper needs a non-empty string `name` | give the class or the object a `name` |
| `methods`: neither score nor generate | write at least one of the two |
| `generate`: `NotImplementedError`, or another error's name and message | the error was raised inside your `generate`: the message is its own |
| `generate`: answer 0 must hold 2 strings (sampling.n) | return exactly `sampling.n` replies for every prompt |
| `generate`: returned replies for 1 prompts, not 2 | return one list of replies for each prompt, in the order given |
| `score`: answer 0 is list, not Scored | wrap each list of log-probabilities: `ca.Scored(logprobs)` |
| `score`: a log-probability must be finite and at most 0 | return natural-log probabilities, not probabilities or raw logits |
| `token texts`: given, but they do not join back to the response exactly | make the texts join to the response character for character, or leave `tokens` out |
| `prompt matters`: the score ignores the prompt | the prompt is not reaching the model: check how you join prompt and response |

A `note` on `reply above reversed` means the model finds its own reply no more likely than the same words in reverse. For a real language model, that usually means the response is not placed where the model's reply begins.

## 4. Rules a wrapper keeps

| Rule | Why |
|---|---|
| Use your model's own chat format, and its way of handling a system prompt | ContextAttr does not know any model's format |
| A call gets up to `batch_size` prompts (32 unless you set it). Run them however is fastest | ContextAttr hands over its prompts in batches of that size |
| Raise an error on a failure and on a prompt that is too long. Never cut a prompt short | a shortened prompt is a different prompt, and its score would be wrong without any sign |
| Keep the prompt the same from day to day. Some chat formats write today's date into it: fix that date | cached answers from yesterday must match today's prompts |

## 5. Check the results with planted facts

`check_model` tells you that the wrapper answers in the right form. It cannot tell you whether the results are right with your model. The demonstration can: it explains three prompts whose facts are invented, so the right result is known before the model is asked.

```bash
contextattr demo                          # the built-in stand-in: no model, about a minute
contextattr demo my_wrapper.py:MyModel    # your wrapper
```

A wrapper that takes arguments is run from Python:

```python
import torch
import contextattr as ca
from my_model import LocalModel               # the wrapper of section 2, saved as my_model.py

ca.demo(LocalModel("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32), cache="demo.sqlite")
```

**What it asks of the model.** Three cases of 8 documents each. The wrapper is asked about every combination of the documents: 1,024 prompts to score (seconds on a GPU, minutes on a CPU), and five short replies if it can also generate. With `cache=`, a second run asks nothing.

**What you should see, and why:** [results.md](results.md) goes through the three cases and how to read each.

**The lines under each table:**

| A line starts with | It means | What to do |
|---|---|---|
| `ok` | as it should be | |
| `FAIL` | a check of the package itself failed. These checks compare the scores with brute force over every combination of the documents, a pair's term with its direct measurement, and a second run with the cache. They hold whatever the model says | report it, with the output |
| `note` | your model read the documents in another way than planted | nothing is wrong: the line says what the model did |

**The stand-in is not a model.** Without a wrapper, the demonstration uses fixed rules written for these three prompts. There is no neural network behind it and no weights, so every line says `ok` by construction. It shows that the package is installed and works, and what the output looks like. It says nothing about any real model.

**On a real model** (Qwen2.5-0.5B-Instruct on a CPU, measured): every check of the package passes. Cases 1 and 3 come out as planted for all nine methods, with R² from 0.72 (`lime`) to 1.00 in case 1. Case 2 gets one `note`: this small model says "October" from `d3` alone, so the two documents overlap instead of needing each other, and the methods give the pair the term that was measured directly (−0.506). A model that cannot read at all, such as a test model with random weights, gets a `note` in every case and still passes every check of the package.

## 6. Use it

```python
import contextattr as ca
from my_wrapper import MyModel

model = MyModel()
e = ca.explain(prompt, model, ca.Contrast())
```

A run file names the wrapper by its file and class ([interface.md](interface.md) §8).

Here `prompt` is a `ca.Prompt` and `e` is the result. [results.md](results.md) shows how to read it, [methods.md](methods.md) covers the methods and every setting, and [interface.md](interface.md) covers prompts, targets and runs.
