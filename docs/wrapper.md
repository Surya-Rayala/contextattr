# Writing a wrapper

ContextAttr never talks to a model itself. It hands prompts to a small class that you write, the *wrapper*, and your class calls the model: in your own Python process, on your server, or through an API. A wrapper is usually 20 to 50 lines, and `check_model` tells you whether yours is right before you spend anything on a run. In all the docs, `ca` is the package: `import contextattr as ca`.

**The short way.** If your model answers through an OpenAI-style API, copy the wrapper of section 5. If it is a Hugging Face model that you run yourself, copy the one of section 6. Put in your model's name, run the file, and it tells you whether it works. The other sections explain what those two wrappers do, for when yours has to differ.

## 1. What a wrapper is

```python
class MyModel:
    name = "my-model"                          # every answer is cached under this name

    def generate(self, prompts, sampling):     # writes replies to each prompt
        ...

    def score(self, prompts, responses):       # says how likely a given reply is
        ...
```

It needs `name` and at least one of the two methods. Which ones you write decides what you can explain (the *target*: [interface.md](interface.md) §5):

| Method | What it does | What you can explain with it | Which models can |
|---|---|---|---|
| `generate` | writes replies to a prompt | how often the model behaves in some way: `ca.Sampled(check)` | every model and API |
| `score` | says how likely the model finds a reply that you supply: a log-probability for each of its tokens | why the model gives one answer, or one kind of answer and not another: `ca.LogProb`, `ca.Pooled`, `ca.Contrast`, `ca.Choice` | only backends (the software or service that runs the model) that return log-probabilities of text you give them |

A *log-probability* says how likely the model finds a text: 0 is certain, and the more negative, the less likely. A *token* is the piece of text a model reads at a time, about a word.

Write `score` if your backend can do it: it costs one pass over the prompt with nothing generated, and it gives the same number every time. Write both where you can: `generate` also gives you the model's own answers to explain (`ca.sample`).

## 2. What your backend can do

| Where the model runs | `generate` | `score` |
|---|---|---|
| in your own process: transformers or any framework, on a GPU, a CPU or Apple MPS | yes | yes |
| a vLLM server | yes | yes: `prompt_logprobs` on its chat and completions endpoints. Each response to a prompt costs a full pass over it, even with prefix caching |
| an SGLang server | yes | yes: `return_logprob` with `logprob_start_len` inside the prompt (the default gives generated tokens only) |
| AWS Bedrock, a model brought in with Custom Model Import | yes | yes: `InvokeModel` with `return_logprobs` returns the prompt's log-probabilities |
| AWS Bedrock's own models through `Converse` | yes | no: the API has no log-probability field |
| AWS SageMaker or another managed endpoint | yes | whatever its serving container exposes |
| the OpenAI and Gemini APIs | yes | no: log-probabilities of generated tokens only |
| the Anthropic API | yes | no: it gives no log-probabilities |

As found on 2026-11-05, from each service's documentation (for Gemini and Anthropic, from their client libraries and users' reports). Services change, so check yours.

## 3. The steps

1. **Get the file to fill in.** `contextattr template my_wrapper.py` writes it.
2. **Fill it in.** Sections 4 to 6 show what goes in each method, with two complete wrappers to copy from.
3. **Check the wrapper** (below).
4. **Check the results.** `contextattr demo my_wrapper.py:MyModel` explains three prompts with planted facts, where the right result is known (section 10).
5. **Use it** (section 9).

**Checking the wrapper.** `python my_wrapper.py` runs `ca.check_model` on your wrapper: about a dozen small calls on a built-in prompt. It prints one line per check. For example, for Llama-3.2-3B-Instruct on one GPU:

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
| `note` | allowed, but worth knowing: for example, the model does not honour seeds, so two runs sample different replies |
| `FAIL` | the wrapper is not ready: the line says what went wrong (section 8) |
| `info` | a measurement, such as the seconds per request |

The last two lines say which targets your wrapper can serve and whether it is ready.

## 4. What the two methods get and return

Both get `prompts`, a list. Each prompt is one of:

- chat messages with text content, such as `[{"role": "system", "content": "Answer in one sentence."}, {"role": "user", "content": "Documents:\n..."}]`
- a plain string, when the prompt was built from `text=` and not from `messages=`

The prompts are already masked: ContextAttr has removed the parts it is testing. Your wrapper only passes each prompt on in the form your model expects.

**`generate(prompts, sampling)`** returns one list per prompt, holding exactly `sampling.n` replies. A reply is the text the model wrote and nothing else. For two prompts and `sampling.n == 1`:

```python
[["It is hot and humid."], ["I do not know."]]
```

`sampling` says how to sample:

| Field | Default | Meaning |
|---|---|---|
| `n` | 1 | replies per prompt |
| `temperature` | 0.0 | how random the replies are. 0 always picks the most likely next token (called greedy); higher gives more varied replies |
| `top_p` | 1.0 | sample only among the most likely tokens that together have this probability. 1.0 means all of them |
| `max_tokens` | 256 | the longest reply |
| `seed` | 0 | the same seed should give the same replies; `None` means no seed |
| `stop` | `()` | strings that end a reply; the reply does not include them |

**`score(prompts, responses)`** gets one response (a string) per prompt and returns one `ca.Scored` per pair. For a response the model reads as three tokens:

```python
[ca.Scored(logprobs=[-0.21, -1.35, -0.08])]
```

Each number is the natural-log probability the model gives that token of the response, given the prompt and the response's earlier tokens. To get them:

1. Format the prompt as your model expects. A chat model wants each message wrapped in its own marker tokens: its *chat template*.
2. Put the response where the model's reply would begin.
3. Run the model once and read off the probability of each response token. Nothing is generated.
4. Do not count the marker that closes the reply (the end-of-turn token).

Optional: `ca.Scored(logprobs, tokens)` also carries the text of each token. The texts must join back to the response exactly. With them, a target can cover only part of a response (a `span`).

## 5. A complete wrapper for an API: `generate`

This one works with every service that speaks the OpenAI chat API: OpenAI itself, and vLLM, Ollama, llama.cpp and many hosted services through `base_url`. For another API, change only the request in `reply`.

```python
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

import contextattr as ca


class ChatAPI:
    def __init__(self, model, **client_options):
        self.name = model                            # the model's id: use a dated one where the service has them
        self.client = OpenAI(**client_options)       # base_url="http://localhost:8000/v1" for your own server

    def reply(self, prompt, sampling):
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        answer = self.client.chat.completions.create(
            model=self.name, messages=messages, n=sampling.n, temperature=sampling.temperature,
            top_p=sampling.top_p, max_tokens=sampling.max_tokens, seed=sampling.seed,
            stop=list(sampling.stop) or None)
        return [choice.message.content or "" for choice in answer.choices]

    def generate(self, prompts, sampling):
        with ThreadPoolExecutor(max_workers=8) as pool:           # the prompts of one call, eight at a time
            return list(pool.map(lambda prompt: self.reply(prompt, sampling), prompts))


if __name__ == "__main__":
    print(ca.check_model(ChatAPI("your-model-id")))
```

- Services differ in the options they accept. OpenAI's newer models take `max_completion_tokens` in place of `max_tokens`, and some servers return one reply whatever `n` says: `check_model` then fails on `generate`, and the fix is to call the service `sampling.n` times.
- The API key is read by the client from its usual environment variable. Keep it out of the wrapper file.

## 6. A complete wrapper for a model in your own process: `score` and `generate`

This one runs a Hugging Face model with transformers, one prompt at a time: the simplest version that is correct.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import contextattr as ca


class LocalModel:
    def __init__(self, path, device="cpu", **model_options):
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
                                      pad_token_id=self.tokenizer.eos_token_id, **options)[:, ids.shape[1]:]
            replies = self.tokenizer.batch_decode(new, skip_special_tokens=True)
            for stop in sampling.stop:               # a reply ends before a stop string
                replies = [reply.split(stop)[0] for reply in replies]
            out.append(replies if sampling.temperature > 0 else replies * sampling.n)
        return out


if __name__ == "__main__":
    print(ca.check_model(LocalModel("path/or/hub-id/of/your/model")))
```

- **Speed.** `score` is where a run spends its time. To go faster, score a batch of prompts in one pass (pad on the left, so that every response ends at the last position) and load the model in a lower precision, for example `LocalModel(path, device="cuda", dtype=torch.bfloat16)` (`torch_dtype` in older versions of transformers).
- **Precision and batches.** In bfloat16, the value of a masked prompt shifts a little with the batch it is padded into: for Llama-3.2-3B-Instruct by 0.022 in the mean log-probability per token, 1.6 % of the spread over masks (measured). Where that matters, score in float32.
- **The model's own generation settings.** `model.generate` also applies whatever the model's `generation_config` sets, such as a repetition penalty. Replace that config if you want only `sampling` to decide.
- **Prompts that change by the day.** Llama 3's chat template writes today's date into every prompt. Pass a fixed `date_string="..."` to `apply_chat_template`, or cached answers from yesterday no longer match today's prompts.

## 7. The rules

| Rule | Why |
|---|---|
| Change `name` whenever the answers could change: another model, revision, precision or prompt format | every cached answer is stored under the name |
| Apply your model's own chat template and its way of handling a system prompt | ContextAttr does not know any model's format |
| `score`: put the response right where the model's reply begins, and return one log-probability per response token, without the end-of-turn token | the number explained is the likelihood of that response |
| `generate`: return exactly `sampling.n` replies per prompt, the reply's text only, and honour `temperature`, `top_p`, `max_tokens`, `stop` and `seed` where your backend can | `check_model` reports what is not honoured |
| A call gets up to `batch_size` prompts (32 unless you set it): run them however is fastest, as one GPU batch or as concurrent requests | ContextAttr sends all of a method's prompts at once |
| Raise an error on a failure and on a prompt that is too long. Never cut a prompt short | a shortened prompt is a different prompt, and its score would be wrong without any sign |
| Retries and rate limits are the wrapper's job (most API clients already retry) | every answer is cached, so an interrupted run continues where it stopped |

## 8. When `check_model` says FAIL

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

## 9. Use it

```python
import contextattr as ca
from my_wrapper import MyModel

model = MyModel()
e = ca.explain(prompt, model, ca.Contrast())                                # needs score
e = ca.explain(prompt, model, ca.Sampled(ca.checks.regex(r"\bcold\b")))     # needs only generate
```

On the command line, the run file names the wrapper by its file and class, and `args` are passed to it:

```toml
[model]
use = "my_wrapper.py:MyModel"
args = { path = "..." }
```

Here `prompt` is a `ca.Prompt` and `e` is the result. [results.md](results.md) shows how to read it, [methods.md](methods.md) covers the methods and every setting, and [interface.md](interface.md) prompts, targets and runs. To try everything before writing a wrapper, `ca.ToyModel.demo()` is a ready-made stand-in: fixed rules written for the docs' example, with no model behind it.

## 10. Check the results with planted facts

`check_model` tells you that the wrapper answers in the right form. It cannot tell you whether the results are right with your model. The demonstration can: it explains three prompts whose facts are invented, so the right result is known before the model is asked.

```bash
contextattr demo                          # the built-in stand-in: no model, about a minute
contextattr demo my_wrapper.py:MyModel    # your wrapper
```

A wrapper that takes arguments is run from Python:

```python
import torch
import contextattr as ca
from my_model import LocalModel               # the wrapper of section 6, saved as my_model.py

ca.demo(LocalModel("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32), cache="demo.sqlite")
```

**What it asks of the model.** Three cases of 8 documents each. A wrapper that scores is asked about every combination of the documents: 1,024 prompts to score (seconds on a GPU, minutes on a CPU), and five short replies if it can also generate. A wrapper that can only generate gets the first case alone, by counting replies: about 80 short ones. With `cache=`, a second run asks nothing.

**What you should see, and why:**

| Case | What is planted | What you should see | Why |
|---|---|---|---|
| 1. One document holds the answer | only `d2` says that the bridge was opened in 1932 | `d2` comes first for every method, far above the others, and without `d2` the model's answer changes | the year is invented, so the model can have it only from `d2` |
| 2. Two documents hold the answer together | `d2` ties the festival to the fair, and `d3` gives the fair's month | `d2` and `d3` are the two top documents, and the pair has a term of its own | neither sentence answers the question alone |
| 3. A later document overrules an earlier one | `d2`: 310 metres when built. `d4`: 450 metres now | `d4` has the highest score and `d2` the lowest, for the value log p("450") − log p("310"). The model says 450 with `d4` and 310 without it | each number stands in one document only |

**The lines under each table:**

| A line starts with | It means | What to do |
|---|---|---|
| `ok` | as it should be | |
| `FAIL` | a check of the package itself failed. These checks compare the scores with brute force over every combination of the documents, a pair's term with its direct measurement, and a second run with the cache. They hold whatever the model says | report it, with the output |
| `note` | your model read the documents in another way than planted | nothing is wrong: the line says what the model did |

**The stand-in is not a model.** Without a wrapper, the demonstration uses fixed rules written for these three prompts. There is no neural network behind it, no weights, and nothing random, so every line says `ok` by construction. It shows that the package is installed and works, and what the output looks like. It says nothing about any real model.

**On a real model** (Qwen2.5-0.5B-Instruct on a CPU, measured): every check of the package passes. Cases 1 and 3 come out as planted for all nine methods, with R² from 0.72 (`lime`) to 1.00 in case 1. Case 2 gets one `note`: this small model says "October" from `d3` alone, so the two documents overlap instead of needing each other, and the methods give the pair the term that was measured directly (−0.506). A model that cannot read at all, such as a test model with random weights, gets a `note` in every case and still passes every check of the package.
