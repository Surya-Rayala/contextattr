# ContextAttr: what a user provides and gets

ContextAttr is for anyone with a model, hosted anywhere: in the caller's process, on a self-hosted server, on AWS or another cloud, or behind an API. Its user writes a small wrapper ([wrapper.md](wrapper.md) is the guide); ContextAttr checks the wrapper and does everything after it. [methods.md](methods.md) holds the methods, their settings and how to read a result.

Words used here: a **prompt** is what the model is asked; a **unit** is a removable piece of a prompt; a **mask** keeps some units and removes the rest; a **response** is a reply of the model; the **target** says which number is read from the model for each masked prompt, the **value** f(m); the **budget** is the number of masks a method evaluates per prompt.

## 1. The flow

```
prompts (a file or Python) → units
  → responses: given, or k sampled from the full prompt → labels from a check (positive, negative)
the method picks masks → masked prompts → the wrapper scores or generates → the target → one value per mask
values → scores per unit and per set of units → checks (R², removal, resampling) → report; summary over prompts
```

Three ways in, from the usual to the most general:

| Call | The user gives | For |
|---|---|---|
| `ca.explain(prompt, model, target=…)` | a wrapper ([wrapper.md](wrapper.md)) and a target (§5) | language models: the usual case |
| `ca.explain(prompt, value=f)` | any function from masked prompts to numbers | classifiers, reward models, judges, an API's first-token probabilities |
| `ca.attribute(f, n)` | any function from masks (an m × n array of 0 and 1) to numbers | inputs that are not text, such as image patches or table columns |

## 2. The prompt file: `prompts.jsonl`, one prompt per line

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

| Field | Meaning | Rules |
|---|---|---|
| `id` | the prompt's name | unique in the file |
| `messages` | chat messages; `role` is system, user or assistant | `content` is a string (fixed text) or a list of strings (fixed) and units. Base models and classifiers take `text` (a string or such a list) instead |
| `unit` | the unit's id | unique in the prompt; makes the piece removable |
| `text` | the unit's text | kept exactly as written |
| `sep` | text after the unit, removed with it | default empty; kept when the unit is replaced |
| `split` | `word`, `sentence`, `line` or `paragraph`: cuts the piece into units `<unit>.0`, `<unit>.1`, … grouped under `<unit>` | the pieces join back to the exact original; the piece's `sep` follows the group and goes only with its last unit |
| `group` | an explicit group: `units` (unit objects), optional `lead` and `sep` | how `write_prompts` writes a split piece; `lead` and `sep` stay while any of its units shows |
| `tag` | a label for units that play the same part in different prompts (an instruction, a retrieved document, a few-shot example) | used only to summarise over prompts |
| `replace` | this unit's replacement text, overriding the run's removal setting | optional |
| `reference` | the desired answer, for checks | optional |
| `responses` | given responses: `id`, `text`, `label` (`positive`, `negative` or absent), optional `span` [start, end) of the characters that matter | optional; sampled responses are added to them |

Everything not marked as a unit is fixed and never removed. With every unit kept, the rendered prompt equals the original exactly (a test). In Python the same prompt is built without JSON, with `ca.Prompt`, `ca.Unit` and `ca.split`.

How `split` cuts. Whatever the text, the units and what lies between them (their `sep`) give it back exactly:

| `split` | A unit is |
|---|---|
| `word` | a run of non-space characters |
| `sentence` | text within one line up to `.`, `!`, `?` or `…` (with any closing quotes or brackets) followed by a space and a character that is not lower case, or up to `。`, `！` or `？`. A period ends none after an initial (`J.`), a dotted abbreviation (`U.S.`, `e.g.`), a title (`Dr.`, `vs.`, `al.`), `Fig.`, `No.`, `Jan.` and the like before a number, or a list marker at a line's start (`1.`, `a.`) |
| `line` | a line with text, its indentation included |
| `paragraph` | lines with text between blank lines |
| a function | each (start, end) character span it returns |

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

These are the two capabilities of lm-evaluation-harness's model interface (`loglikelihood` and `generate_until`), so anyone who has wrapped a model for it knows the shape.

- `contextattr template my_wrapper.py` writes a wrapper to fill in.
- `ca.check_model(model)` tests a wrapper before any run: about a dozen calls on a built-in prompt, one line per check, and the targets the wrapper can serve.
- `ca.ToyModel` is a compatible wrapper with known answers and no real model behind it, for tests and a first try; `ca.ToyModel.demo()` models the Rio example.
- A wrapper without `score` still supports the sampled target of §5.

## 4. Responses and labels

| Source | Setting | Model calls |
|---|---|---|
| given | `responses` in the prompt file | none |
| the model's own answer | `sample: {n: 1, temperature: 0}`: what SPEX (§3) and ContextCite explain | one generation per prompt |
| k samples | `sample: {n: k, temperature: 0.7, top_p: 0.95, max_tokens: 256, seed: 0}` | k generations per prompt, once |

Labels come from the file, or from a check run on each response: `reference` (normalised match), `regex` (a pattern, and whether a match means negative or positive), or any Python function `check(prompt, response)` returning `positive`, `negative` or `None`, and optionally the `span` of the text that matters. A judge model is a check of the same shape. A run names a check by its code and the values it holds (`ca.checks.describe`), so resuming with another check is refused; a check that depends on anything else carries it in its `__name__` and sets `named = True`.

## 5. Targets: which number is explained

| Target | Value f(m) for one masked prompt | Answers | Needs |
|---|---|---|---|
| `LogProb(response, reduce="mean")` | mean log-probability per token of one response or of its `span` (the negative log-perplexity, SPEX §3); `"sum"` for the total; `"logodds"` for the logit of the response's probability (ContextCite §3) | what in the prompt supports this response | `score` |
| `Pooled(label)` | the mean `LogProb` of the responses with that label | what produces the negative (or positive) responses as a group | `score` |
| `Contrast()` | `Pooled("negative")` − `Pooled("positive")` | what moves the model from desired to undesired responses | `score` |
| `Choice(a, b)` | log p(a) − log p(b) as the response, e.g. "True" − "False" (SPEX B.2.4) | yes/no and multiple-choice answers | `score` |
| `Sampled(check, samples=k)` | the share of k fresh responses to the masked prompt that the check flags; with a numeric metric instead (a judge's score, a classifier's probability, the similarity to the original response), their mean | the behaviour itself; the only kind a generate-only model allows | `generate` |

- Scored targets cost one pass over prompt and response per mask, with no decoding, and repeat exactly; sampled targets cost k generations per mask and carry sampling noise (§7). Use scored targets wherever the backend allows them.
- Suggested: `Contrast` whenever both labels exist. Removing context tends to lower the likelihood of every response, so a negative response's `LogProb` alone mixes "needed for any answer" with "pushes towards the bad one"; the difference cancels what the two share.
- `per_response=True` fits one explanation per response instead of one per pool, at no extra model cost.

## 6. Settings that differ between implementations

Only settings independent of the model are here; device, dtype, chat template, quantisation and server options belong to the wrapper.

| Setting | Choices | Default | Why |
|---|---|---|---|
| units | marked spans, or `split` into words, sentences, lines or paragraphs | as marked in the file | sentences: ContextCite §2.1; words and sentences: SPEX §6 |
| fixed parts | everything not marked | system prompt and question fixed | ContextCite removes only context sources |
| removal | `delete`, or `replace: "<text>"` | `delete` | works with every model and API and keeps the text natural (ContextCite §2.1, footnote 4); `replace: "[UNK]"` reproduces SPEX (B.2) |
| responses | given, the greedy answer, or k samples (§4) | those given in the prompt; none are sampled unless `sample` is set (`sample=ca.Sampling(n=1)` adds the greedy answer) | SPEX §3 and ContextCite explain the original output |
| target | §5 | `ca.explain`: `LogProb` (mean) of the prompt's one response. A run chooses by each prompt's responses: `Contrast` when both labels exist, `Pooled` with one label, `LogProb` for a single response. `Sampled` is used only when named | |
| fresh responses per mask, for `Sampled` | k, and how they are sampled | 4, sampled as in §4 | the noise in §7 |
| method and budget | §7 | `contextcite` with 64 masks to screen; `spex` or `proxyspex` for a closer look | the costs in §7 |
| mask seed | an integer | 0 | the same masks for every response and every rerun |
| test masks for R² | a count, shared by every method run on the prompt | 128 (the paper used 10,000 per example, SPEX §6.1) | each costs as much as a method's mask |
| noise repeats | the full prompt's value recomputed, bypassing the cache | 3 | scores smaller than the noise mean nothing |
| batch size | items per wrapper call | 32 | |

## 7. Cost and what is practical

Requests per prompt = (method masks + test masks) × (responses scored, or k fresh responses for `Sampled`), plus the noise repeats and, once, the samples of §4. A request carries about the masked prompt plus one response: on one's own hardware the time is tokens ÷ throughput, on an API the cost is tokens × price. `run.plan(measure=10)` sends ten real requests, times them and projects the whole run before anything more is spent. For example (measured with a transformers wrapper on one H100): Llama-3.2-3B-Instruct in bf16 scores 45,000 tokens/s at batch 16, 58 requests/s of 777 tokens each.

Masks per prompt (SPEX Table 2 for spex, b = 6):

| Method | Masks per prompt | n = 20 | n = 100 | n = 1,000 |
|---|---|---|---|---|
| leave-one-out | n + 1 | 21 | 101 | 1,001 |
| contextcite, lime, banzhaf, kernelshap | the budget | 64–256 | 256–1,024 | 1,024 or more |
| faith, order 2 | a few times the 1 + n + n(n−1)/2 terms | ≈ 600 | ≈ 15,000 | not practical |
| spex | ≈ C·2ᵇ·t·log n | 5,376 | 7,872 | 10,752 |
| proxyspex | about a tenth of spex (Butler 2025) | ≈ 500 | ≈ 800 | ≈ 1,100 |

For example (derived): a RAG prompt cut into 100 sentences, with 4 responses scored per mask and 128 test masks, needs (64 + 128) × 4 = 768 requests with `contextcite` at 64 masks, and (7,872 + 128) × 4 = 32,000 with `spex`.

Suggestions:
- Cost grows with masks × responses × prompt length: sentence or document units for long contexts, words only for short texts.
- Screen many prompts with `contextcite` at 64 masks (leave-one-out when n ≤ 20); run `spex` or `proxyspex` only on the prompts worth a closer look, with a scored target.
- `Sampled` targets are noisy: the share of k responses has a standard error of √(p(1 − p)/k), 0.18 at k = 8 and p = 0.5. Keep their budgets small, prefer a numeric metric to a yes/no check, and confirm the result by removing the top units and sampling 64 or more responses (`ca.intervene`), as SPEX did on the trolley prompt (accuracy 11.8% → 20.8%, B.5.2).
- In bf16, a model in the caller's process gives a masked prompt a value that shifts a little with the batch it is padded into. For Llama-3.2-3B-Instruct the shift was std 0.022 in the mean log-probability per token, 1.6% of the spread over masks (measured). The noise repeats recompute only the full prompt, alone, so they miss it. Where it matters, score the same masks at two batch sizes, or in fp32.
- Every model result is cached in one SQLite file per run: an interrupted run resumes, another method on the same masks reuses the results, and nothing is paid for twice.
- A wrapper for an API should send a batch concurrently, within the provider's rate limits. One around a model in the caller's process should batch on its device, and can compute each masked prompt once for all its responses.
- A vLLM server recomputes the whole prompt for prompt log-probabilities even with prefix caching (vLLM V1 guide), so k responses to one masked prompt cost k passes there.
- `replace` can lengthen a prompt (`[UNK]` is several ordinary tokens for Llama 3's tokenizer), so the length check renders the prompt once with every unit replaced.

## 8. Using it, and what a run leaves

```python
import contextattr as ca

model = MyWrapper(...)                    # the user's own wrapper (wrapper.md)
print(ca.check_model(model))              # what it supports, before any run

run = ca.Run(model, ca.read_prompts("prompts.jsonl"),
             sample=ca.Sampling(n=8, temperature=0.7),            # k responses per prompt from the full prompt
             check=ca.checks.regex(r"\bwinter\b", negative=True),  # labels them
             target=ca.Contrast(),                                 # ca.Sampled(check, samples=4) if the model can only generate
             method="contextcite", budget=64)
print(run.plan(measure=10))               # requests, tokens, measured seconds per request, projected time
results = run.execute("runs/rio")         # resumable: an interrupted run continues; a prompt changed since is refused
results.summary(by="tag")                 # over prompts: how often each tag ranks in the top 3
e = results.explanations[0]               # scores, R², noise, costs; e.table(), e.to_html(path), e.shapley()
ca.intervene(e.prompt, model, ca.checks.regex(r"\bwinter\b"), [u for u, _ in e.top(3)], samples=64)
                                          # remove the top units and sample again: does the negative share fall?
```

The command line reads the same settings from a TOML file (read by Python's standard library) and names the wrapper by import path, so one run file works wherever its wrapper does:

```toml
[model]
use = "my_wrappers.py:BedrockModel"
args = { model_id = "…", region = "us-east-1" }
```

`contextattr plan run.toml`, `contextattr run run.toml`, `contextattr report runs/rio`. A run directory holds `run.json` (every setting, the package version, the wrapper's name, seeds), `cache.sqlite` (every model result, one file), `explanations.jsonl` (per prompt: units, scores, top interactions, R², noise, removal curve, requests, seconds, and a digest of the prompt as given, which a resumed run compares), `summary.json` and `report.html`.
