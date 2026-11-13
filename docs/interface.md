# Reference: prompts, answers, targets, costs and runs

Everything you give ContextAttr and everything you get back. The other docs: [wrapper.md](wrapper.md) connects your model, [methods.md](methods.md) covers the methods and every setting, and [results.md](results.md) shows how to read a result.

## Words and names

| Word | Meaning |
|---|---|
| prompt | what the model is asked: chat messages, or plain text |
| unit | a part of the prompt that may be removed: a word, a sentence, a document, or any piece you mark. `n` is the number of units in a prompt |
| fixed text | everything you do not mark as a unit, such as the question. It is never removed |
| mask (`m`) | one choice of units to keep: one entry per unit, 1 to keep it and 0 to remove it |
| masked prompt | the prompt with only the units a mask keeps |
| response | an answer of the model: one you give, or one sampled from the model |
| token | the piece of text a model reads at a time: about a word, or part of one |
| log-probability | how likely the model finds a text, on a log scale: 0 is certain, and the more negative, the less likely |
| label | `positive` for an answer you want, `negative` for one you do not |
| check | a function that looks at a response and labels it |
| target | says which number is read from the model for a masked prompt (§5) |
| value (`f(m)`) | that number, for mask `m` |
| method | decides which masks to ask about and works out the scores from their values |
| budget | how many masks a method may use for one prompt |
| score | how much a unit pushes the value up (positive) or down (negative) |
| interaction | what a set of units does together, beyond their own scores |
| R² | how well the scores predict the value on test masks, drawn separately from the method's own: 1 is perfect |
| request | one unit of work for the model: one masked prompt with one response to score, or one masked prompt to reply to |
| wrapper | your class around the model ([wrapper.md](wrapper.md)) |

In the code examples: `ca` is the package (`import contextattr as ca`), `prompt` is a `ca.Prompt`, `model` is your wrapper, `e` is one result (an `Explanation`), `run` is a `ca.Run` and `results` is what it returns.

## 1. The flow

1. You give prompts with their units marked (§2), and a wrapper around your model (§3).
2. Each prompt gets responses: the ones you give, or ones sampled from the model and labelled by a check (§4).
3. The method picks masks. For each mask, the wrapper scores the responses or generates replies, and the target turns that into one value (§5).
4. The method works out a score per unit, and per set of units, from the values. The result is checked on test masks (R²) and reported.

Three ways in, from the usual to the most general:

| Call | You give | For |
|---|---|---|
| `ca.explain(prompt, model, target)` | a wrapper and a target | language models: the usual case |
| `ca.explain(prompt, value=f)` | any function `f` from masked prompts to numbers | classifiers, reward models, judges |
| `ca.attribute(f, n)` | any function `f` from masks to numbers | inputs that are not text, such as image patches or table columns |

`ca.Run` (§8) does the first for a whole file of prompts.

## 2. Prompts

### In Python

```python
prompt = ca.Prompt("rio-01",
    messages=[
        {"role": "system", "content": "Answer in one sentence."},
        {"role": "user", "content": [
            "Documents:\n",
            ca.Unit("d1", "Rio Carnival takes place before Lent, in February or March.", sep="\n", tag="doc"),
            ca.Unit("d2", "Summer in Brazil runs from December to March.", sep="\n", tag="doc"),
            ca.Unit("d3", "Winter in Brazil runs from June to September.", sep="\n", tag="doc"),
            "Question: What is the weather like during Rio Carnival?"]}],
    reference="Hot and humid: it is summer in Brazil.",
    responses=[ca.Response("bad-1", "It is cold, because Carnival is in winter.", label="negative")])
```

| Piece | Setting | Default | Meaning |
|---|---|---|---|
| `ca.Prompt(id, ...)` | `id` | | the prompt's name, used in results |
| | `messages` | | chat messages. `role` is `system`, `user` or `assistant`. `content` is a string (fixed text), or a list of strings (fixed) and units |
| | `text` | | used in place of `messages` for a model that takes plain text: a string, or a list of strings and units |
| | `reference` | `None` | the answer you want, for a check to compare with (§4) |
| | `responses` | `()` | answers to explain (§4) |
| `ca.Unit(id, text, ...)` | `id` | | the unit's name, unique in the prompt. Results show it |
| | `text` | | the unit's text, kept exactly as written |
| | `sep` | `""` | text that follows the unit and is removed with it, such as a line break |
| | `tag` | `None` | a label for units that play the same part in different prompts (`"doc"`, `"instruction"`), used to summarise over prompts (§8) |
| | `replace` | `None` | what this unit is replaced by when removed, in place of the run's `replace` setting (§6) |
| `ca.Response(id, text, ...)` | `id`, `text` | | the answer's name and its text |
| | `label` | `None` | `"positive"`, `"negative"` or none |
| | `span` | `None` | `(start, end)` character positions: the part of the answer that matters. Needs a wrapper that returns token texts |
| `ca.split(text, by, id=...)` | `by` | `"sentence"` | cuts a text into units: `"word"`, `"sentence"`, `"line"`, `"paragraph"`, or your own function that returns `(start, end)` positions |
| | `id` | | the units are named `<id>.0`, `<id>.1`, and so on |
| | `sep`, `tag`, `replace` | | as for a unit, for the whole piece |

`ca.split` goes where a unit would go, for example `"content": [ca.split(long_text, "sentence", id="s"), "Question: ..."]`. Whatever the text, the units and what lies between them give it back exactly, and with every unit kept the prompt is exactly the original.

How `split` cuts:

| `by` | A unit is |
|---|---|
| `"word"` | a run of characters without a space |
| `"sentence"` | text within one line up to `.`, `!`, `?` or `…` (with any closing quotes or brackets) followed by a space and a character that is not lower case, or up to `。`, `！` or `？`. A period does not end a sentence after an initial (`J.`), a dotted abbreviation (`U.S.`, `e.g.`), a title (`Dr.`, `vs.`, `al.`), `Fig.`, `No.`, `Jan.` and the like before a number, or a list marker at a line's start (`1.`, `a.`) |
| `"line"` | a line with text, its indentation included |
| `"paragraph"` | lines with text between blank lines |

### In a file: `prompts.jsonl`, one prompt per line

The same prompt (shown on several lines here; in the file it is one line). `ca.read_prompts(path)` reads the file and `ca.write_prompts(prompts, path)` writes it.

```json
{"id": "rio-01",
 "messages": [
   {"role": "system", "content": "Answer in one sentence."},
   {"role": "user", "content": [
     "Documents:\n",
     {"unit": "d1", "text": "Rio Carnival takes place before Lent, in February or March.", "sep": "\n", "tag": "doc"},
     {"unit": "d2", "text": "Summer in Brazil runs from December to March.", "sep": "\n", "tag": "doc"},
     {"unit": "d3", "text": "Winter in Brazil runs from June to September.", "sep": "\n", "tag": "doc"},
     "Question: What is the weather like during Rio Carnival?"
   ]}
 ],
 "reference": "Hot and humid: it is summer in Brazil.",
 "responses": [{"id": "bad-1", "text": "It is cold, because Carnival is in winter.", "label": "negative"}]}
```

| Field | Meaning |
|---|---|
| `id`, `messages` or `text`, `reference`, `responses` | as in Python above |
| `{"unit": ..., "text": ...}` | a unit; `unit` is its id. `sep`, `tag` and `replace` as above |
| `"split": "sentence"` on a unit | cuts its text into units `<unit>.0`, `<unit>.1`, … as `ca.split` does |
| `{"group": ..., "units": [...]}` | an explicit group of units, with an optional `lead` and `sep` that stay while any of its units shows. This is how `write_prompts` writes a split piece |

## 3. The model wrapper, written by the user

ContextAttr ships no wrapper for any model or service: writing one is the user's job, and everything after it is the package's. [wrapper.md](wrapper.md) is the guide: what to write, two complete wrappers to copy from, the rules a wrapper keeps, and which backends can score. In short, a wrapper is any object with a `name` and at least one of two methods:

```python
Rendered = str | list[dict]    # a masked prompt as it reaches the wrapper: plain text, or chat messages
                               # [{"role": "user", "content": "..."}] with text content

@dataclass(frozen=True)
class Sampling:
    n: int = 1                 # responses per prompt
    temperature: float = 0.0   # 0: greedy
    top_p: float = 1.0
    max_tokens: int = 256
    seed: int | None = 0
    stop: tuple[str, ...] = ()

@dataclass(frozen=True)
class Scored:
    logprobs: Sequence[float]             # natural-log probability of each response token, in order
    tokens: Sequence[str] | None = None   # the text of each response token, joining back to the response exactly; needed only for spans

class Model(Protocol):
    name: str   # changes whenever answers could change (model, revision, dtype, quantisation); part of every cache key
    def score(self, prompts: Sequence[Rendered], responses: Sequence[str]) -> list[Scored]: ...
    def generate(self, prompts: Sequence[Rendered], sampling: Sampling) -> list[list[str]]: ...
```

- `score` gets one response per prompt and returns, for each of the response's tokens, the natural logarithm of the probability the model gives that token after the prompt and the response's earlier tokens: `log p(token t | prompt, tokens 1 to t − 1)`. Nothing is generated.
- `generate` returns `sampling.n` replies for each prompt.
- These are the two capabilities of lm-evaluation-harness's model interface (`loglikelihood` and `generate_until`), so anyone who has wrapped a model for it knows the shape.
- `ca.ToyModel` is a compatible wrapper that follows fixed rules: known answers, no model and no weights behind it. It is for tests and a first try.

## 4. Responses and labels

A response is an answer whose likelihood is explained. There are three ways to get them:

| Where the responses come from | How | Model calls |
|---|---|---|
| you give them | `responses=[ca.Response(...)]` in the prompt | none |
| the model's own answer | `ca.sample(prompt, model, ca.Sampling(n=1))`, or `sample=ca.Sampling(n=1)` in a run (§8). Temperature 0 gives the model's most likely answer, which is what SPEX and ContextCite explain | one generation per prompt |
| several sampled answers | `ca.sample(prompt, model, ca.Sampling(n=8, temperature=0.7), check)`, or `sample=` and `check=` in a run | 8 generations per prompt, once |

`ca.sample` returns the prompt with the replies added to its responses as `sample-0`, `sample-1`, …, each labelled by the check. Without a `sampling` it draws 8 replies at temperature 0.7 and top_p 0.95, up to 256 tokens, with seed 0.

A *check* labels a response. Three kinds:

| Check | Labels a response |
|---|---|
| `ca.checks.regex(pattern, negative=True)` | `negative` when the pattern is found in it, `positive` otherwise. With `negative=False` the other way round. Case is ignored |
| `ca.checks.reference(contains=True)` | `positive` when it contains the prompt's `reference` (lower case, punctuation and extra spaces ignored), `negative` otherwise. With `contains=False` it must equal the reference |
| your own function `check(prompt, response)` | returns `"positive"`, `"negative"` or `None`; or `True` (negative) or `False` (positive); or a number, such as a judge model's score; or a `ca.Verdict(label, score, span)` to also say which part of the response matters |

A run remembers which check it used, by its code and the values it holds, and refuses to continue with a different one. If your check depends on anything else, such as the version of a judge model, put that into the function's `__name__` and set `check.named = True`.

## 5. Targets: which number is explained

The target turns the model's answers for one masked prompt into one number, the value. A score is measured in the same thing as the value.

| Target | The value for one masked prompt | It answers | Needs |
|---|---|---|---|
| `ca.LogProb(response=None, reduce="mean")` | the log-probability of one response under that prompt | what in the prompt supports this answer | `score` |
| `ca.Pooled(label="negative", reduce="mean")` | the mean `LogProb` of all the responses with that label | what produces the negative (or positive) answers as a group | `score` |
| `ca.Contrast(reduce="mean")` | `Pooled("negative")` − `Pooled("positive")` | what moves the model from the answers you want to the ones you do not | `score` |
| `ca.Choice(a, b)` | log p(`a`) − log p(`b`), each taken as the model's whole answer, for example `ca.Choice("True", "False")` | yes/no and multiple-choice questions | `score` |
| `ca.Sampled(check, samples=4, sampling=None)` | the share of `samples` fresh replies to that prompt that the check labels negative; when the check returns numbers, their mean | the behaviour itself. The only target for a wrapper that can only generate | `generate` |

As formulas, for a response of T tokens, where `lp_t` is what the wrapper's `score` returns for token t (§3):

```
LogProb, reduce="mean"      (lp_1 + … + lp_T) / T
         reduce="sum"       lp_1 + … + lp_T  =  log p(the whole response | the masked prompt)
         reduce="logodds"   log( p / (1 − p) ),  with p = exp(lp_1 + … + lp_T)
Pooled(label)               the mean of LogProb over the responses with that label
Contrast()                  Pooled("negative") − Pooled("positive")
Choice(a, b)                the sum for a − the sum for b
Sampled(check)              (replies the check labels negative) / (replies it labels at all),
                            or the mean of the numbers the check returns
```

A response with a `span` (§2) counts only the tokens that overlap the span.

Their settings:

| Setting | Default | Meaning |
|---|---|---|
| `response` (`LogProb`) | `None` | the id of the response to explain. `None` works when the prompt has exactly one |
| `reduce` | `"mean"` | `"mean"`: log-probability per token, which does not favour short answers (the negative log-perplexity the SPEX paper uses, its §3). `"sum"`: of the whole answer. `"logodds"`: log(p / (1 − p)) of the whole answer's probability p, as the ContextCite paper uses |
| `label` (`Pooled`) | `"negative"` | which responses to pool |
| `samples` (`Sampled`) | 4 | fresh replies per masked prompt. More give less noise and cost more |
| `sampling` (`Sampled`) | `None` | how the replies are sampled, a `ca.Sampling`. `None` means temperature 0.7, top_p 0.95, up to 256 tokens, seed 0 |

Which to use:

- Scored targets (the first four) cost one pass over prompt and response per mask, generate nothing, and give the same number every time. Use them wherever your wrapper can `score`.
- Use `Contrast` whenever you have both labels. Removing context tends to make every answer less likely, so the `LogProb` of an unwanted answer alone mixes "needed for any answer" with "pushes towards the unwanted one". The difference cancels what the two share.
- `Sampled` costs `samples` generations per mask and is noisy (§7). Keep its budget small and confirm the result with `ca.intervene` ([results.md](results.md) §8).

## 6. Settings

[methods.md](methods.md) §4 lists every setting of `ca.explain` with its default and meaning: `method`, `budget`, `test_masks`, `seed`, `replace`, `noise_repeats`, `removal`, `cache` and `batch_size`. `ca.Run` takes the same ones except `cache` (a run keeps every answer in its own folder), and adds `sample` and `check` (§4, §8). Device, precision, chat template and server options are not settings of ContextAttr: they belong to your wrapper.

How a unit is removed:

| `replace` | A removed unit is | Use it when |
|---|---|---|
| `None` (the default) | deleted, with its `sep` | always possible, with every model and API, and the text stays natural (ContextCite does this) |
| a text, such as `"[UNK]"` | replaced by that text; its `sep` stays | you want to reproduce SPEX, which masks with `[UNK]` |

A unit's own `replace` (§2) takes the place of the run's setting for that unit. Replacing can make a prompt longer than the original, since a placeholder may be several tokens: the longest prompt is then the one with every unit replaced, so check that it still fits your model.

## 7. Cost and what is practical

Requests per prompt = (the method's masks + the test masks) × the responses scored, plus the noise repeats and, once, the samples of §4. A mask that comes up twice is asked once. For `Sampled`, a request is one masked prompt, and it generates `samples` replies. A request carries about the masked prompt and one response: on your own hardware the time is tokens ÷ throughput, and on an API the cost is tokens × price. `run.plan()` counts all of this before any call, and `run.plan(measure=10)` also sends ten real requests, times them and projects the whole run (§8).

For example (measured with a transformers wrapper on one H100 GPU): Llama-3.2-3B-Instruct in bfloat16 scores 45,000 tokens per second at a batch of 16, which is 58 requests per second of 777 tokens each.

Masks per prompt, by the number of units n ([methods.md](methods.md) §3):

| Method | Masks per prompt | n = 20 | n = 100 | n = 1,000 |
|---|---|---|---|---|
| `loo` | n + 1 | 21 | 101 | 1,001 |
| `contextcite`, `lime`, `banzhaf`, `kernelshap` | the budget | 64–256 | 256–1,024 | 1,024 or more |
| `faithbanzhaf`, `faithshap` with pairs | a few times 1 + n + n(n − 1)/2 | about 600 | about 15,000 | not practical |
| `spex` | SPEX's Table 2 | 5,376 | 7,872 | 10,752 |
| `proxyspex` | about a tenth of `spex`'s (Butler 2025) | about 500 | about 800 | about 1,100 |

For example (worked out, not measured): a prompt cut into 100 sentences, with 4 responses scored per mask and 128 test masks, needs (64 + 128) × 4 = 768 requests with `contextcite` at 64 masks, and (7,872 + 128) × 4 = 32,000 with `spex`.

Suggestions:

- Cost grows with masks × responses × prompt length. Use sentences or documents as units for long contexts, and words only for short texts.
- Screen many prompts with `contextcite` at 64 masks (`loo` when n is 20 or less). Run `spex` or `proxyspex` only on the prompts worth a closer look, with a scored target.
- `Sampled` targets are noisy: a share p counted from k replies has a standard error of √(p(1 − p)/k), which is 0.18 for 8 replies and a share of one half. Keep their budgets small, prefer a check that returns a number to a yes/no check, and confirm the result with `ca.intervene` on 64 or more replies.
- Every answer of the model is cached. A run keeps them in its folder, so an interrupted run continues and nothing is paid for twice. A run folder holds one method. With `ca.explain(..., cache=path)`, another method on the same prompt reuses the answers for the masks the two share, such as the test masks.

## 8. Runs: many prompts, and the command line

```python
import contextattr as ca

model = MyWrapper(...)                    # your own wrapper (wrapper.md)
print(ca.check_model(model))              # what it supports, before any run

run = ca.Run(model, ca.read_prompts("prompts.jsonl"),
             sample=ca.Sampling(n=8, temperature=0.7),             # 8 replies of the model to each full prompt
             check=ca.checks.regex(r"\bwinter\b", negative=True),  # labels them: negative if they say "winter"
             target=ca.Contrast(),                                 # ca.Sampled(check, samples=4) if the model can only generate
             method="contextcite", budget=64)
print(run.plan())                         # masks, requests and tokens, counted before any call
results = run.execute("runs/rio")         # explains every prompt into the folder runs/rio
print(results.summary(by="tag"))          # over prompts: how the units with each tag score
e = results.explanations[0]               # the first prompt's result, read as in results.md
print(ca.intervene(e.prompt, model, ca.checks.regex(r"\bwinter\b"), [u for u, _ in e.top(3)], samples=64))
                                          # remove the top 3 units and sample again: does the share of "winter" fall?
```

| `ca.Run(model, prompts, ...)` | Default | Meaning |
|---|---|---|
| `prompts` | | a list of `ca.Prompt`, or the path of a `prompts.jsonl` |
| `target` | `None` | a target of §5. `None` chooses for each prompt by its responses: `Contrast` when both labels are there, `Pooled` with one label, `LogProb` for a single response |
| `sample` | `None` | a `ca.Sampling`: that many replies of the model are added to each prompt's responses first (§4) |
| `check` | `None` | labels the sampled replies (§4) |
| `method`, `budget`, `test_masks`, `seed`, `replace`, `batch_size`, `noise_repeats`, `removal` | as for `ca.explain` | [methods.md](methods.md) §4 |

| Call | Gives |
|---|---|
| `run.plan(measure=0)` | before any spending: for each prompt its units, masks, requests, generations and tokens, and the totals. With `measure=10`, ten real requests of the first prompt are also sent and timed, and the time of the whole run is projected. Nothing has been sampled at that point, so with `sample=` the first prompt must already have the responses the target needs (for `Contrast`, one of each label) |
| `run.execute(folder)` | explains every prompt into the folder and returns the results. If the folder holds an interrupted run with the same settings, it continues where that stopped. Other settings, or a prompt changed since, are refused |
| `results.explanations` | one `Explanation` per prompt ([results.md](results.md) §7) |
| `results.summary(by="tag", top=3)` | for each tag: how many prompts and units have it, the mean score of its units, and the share of those prompts in which one of them is among the top 3. `by="unit"` does the same for each unit id |
| `results.to_html(path)` | one page with every prompt, its units shaded by score |
| `ca.Results.load(folder)` | the results of a finished run, read back |

The run folder holds:

| File | Holds |
|---|---|
| `run.json` | every setting, the package version, the wrapper's name |
| `cache.sqlite` | every answer of the model |
| `explanations.jsonl` | one result per prompt: units, scores, interactions, R², noise, removal, requests |
| `summary.json` | the summary by tag |
| `report.html` | the page of `results.to_html` |

### The command line

`contextattr plan run.toml`, `contextattr run run.toml` and `contextattr report runs/rio` do the same from a file. `run.toml`:

```toml
prompts = "prompts.jsonl"            # paths are relative to this file
out = "runs/rio"                     # the run folder

[model]
use = "my_wrapper.py:MyModel"        # the file and the class of your wrapper; or "package.module:Class"
args = { path = "..." }              # given to the class as keyword arguments

[run]                                # optional: any of method, budget, test_masks, seed, replace, batch_size,
method = "contextcite"               #           noise_repeats, removal
budget = 64

[sample]                             # optional: n, temperature, top_p, max_tokens, seed, stop
n = 8
temperature = 0.7

[check]                              # optional: regex = "..." with negative = true or false,
regex = "\\bwinter\\b"               #           or reference = "contains" or "exact"

[target]                             # optional: kind = "logprob", "pooled", "contrast", "choice" or "sampled",
kind = "contrast"                    #           with that target's settings (§5)
```

| Command | Does |
|---|---|
| `contextattr plan run.toml` | prints the plan; `--measure 10` also times ten real requests |
| `contextattr run run.toml` | explains every prompt into the run folder, and continues an interrupted run |
| `contextattr report runs/rio` | writes `report.html` again and prints the summary |
| `contextattr template my_wrapper.py` | writes a wrapper to fill in ([wrapper.md](wrapper.md)) |

The run file names a method by its name only, so a method with its own settings (`ca.methods.SPEX(degree=3)`) needs Python.
