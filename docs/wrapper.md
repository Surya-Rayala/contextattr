# Writing a wrapper

ContextAttr never talks to a model itself. It hands prompts to a small class that you write, the *wrapper*, and your class calls the model: in your own Python process, on your server, or through an API. A wrapper is usually 20 to 50 lines, and `check_model` tells you whether yours is right before you spend anything on a run.

## 1. What a wrapper is

```python
class MyModel:
    name = "my-model"                          # every answer is cached under this name

    def generate(self, prompts, sampling):     # writes replies to each prompt
        ...

    def score(self, prompts, responses):       # says how likely a given reply is
        ...
```

It needs `name` and at least one of the two methods. Which ones you write decides what you can explain:

| Method | What it does | What you can explain with it | Which models can |
|---|---|---|---|
| `generate` | writes replies to a prompt | how often the model behaves in some way: `ca.Sampled(check)` | every model and API |
| `score` | gives the log-probability of each token of a reply that you supply | why the model gives one answer, or one kind of answer and not another: `ca.LogProb`, `ca.Pooled`, `ca.Contrast`, `ca.Choice` | only backends that return log-probabilities of text you give them |

Write `score` if your backend can do it: it costs one pass over the prompt with nothing generated, and it gives the same number every time. Write both where you can: `generate` also gives you the model's own answers to explain (`ca.sample`).

## 2. What your backend can do

| Where the model runs | `generate` | `score` |
|---|---|---|
| in your own process: transformers or any framework, on a GPU, a CPU or Apple MPS | yes | yes |
| a vLLM server | yes | yes: `prompt_logprobs` on its chat and completions endpoints |
| an SGLang server | yes | yes: `return_logprob` with `logprob_start_len` inside the prompt (the default gives generated tokens only) |
| AWS Bedrock, a model brought in with Custom Model Import | yes | yes: `InvokeModel` with `return_logprobs` returns the prompt's log-probabilities |
| AWS Bedrock's own models through `Converse` | yes | no: the API has no log-probability field |
| AWS SageMaker or another managed endpoint | yes | whatever its serving container exposes |
| the OpenAI and Gemini APIs | yes | no: log-probabilities of generated tokens only |
| the Anthropic API | yes | no: it gives no log-probabilities |

As found on 2026-11-05, from each service's documentation (for Gemini and Anthropic, from their client libraries and users' reports). Services change, so check yours.

## 3. Three steps

1. **Get the file to fill in.** `contextattr template my_wrapper.py` writes it.
2. **Fill it in.** Sections 4 to 6 show what goes in each method, with two complete wrappers to copy from.
3. **Check it.** `python my_wrapper.py` runs `ca.check_model` on your wrapper: about a dozen small calls on a built-in prompt. It prints one line per check. For example, for Llama-3.2-3B-Instruct on one GPU:

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
| `temperature` | 0.0 | 0 means greedy: always the most likely token |
| `top_p` | 1.0 | nucleus sampling |
| `max_tokens` | 256 | the longest reply |
| `seed` | 0 | the same seed should give the same replies; `None` means no seed |
| `stop` | `()` | strings that end a reply; the reply does not include them |

**`score(prompts, responses)`** gets one response (a string) per prompt and returns one `ca.Scored` per pair. For a response the model reads as three tokens:

```python
[ca.Scored(logprobs=[-0.21, -1.35, -0.08])]
```

Each number is the natural-log probability the model gives that token of the response, given the prompt and the response's earlier tokens. To get them: format the prompt as your model expects, put the response where the model's reply would begin, run the model once, and read off the probability of each response token. Nothing is generated. Leave out the end-of-turn token that a chat template adds after a reply.

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

[interface.md](interface.md) describes prompts, targets and runs; [methods.md](methods.md) the methods and how to read a result. To try everything before writing a wrapper, `ca.ToyModel.demo()` is a ready-made stand-in with no model behind it.
