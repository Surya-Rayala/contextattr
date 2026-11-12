# Reading a result

What ContextAttr prints, what a score means for each method, and how far to trust it. Everything here is real output from one small example that runs anywhere in seconds. In all the docs, `ca` is the package: `import contextattr as ca`.

## 1. The example

```python
import contextattr as ca

prompt = ca.Prompt("rio", messages=[{"role": "user", "content": [
    ca.Unit("d1", "Rio Carnival takes place in February or March.", sep="\n"),
    ca.Unit("d2", "Summer in Brazil runs from December to March.", sep="\n"),
    ca.Unit("d3", "Winter in Brazil runs from June to September.", sep="\n"),
    "Question: What is the weather like during Rio Carnival?"]}],
    responses=[ca.Response("bad", "It is cold, because Carnival is in winter.", label="negative"),
               ca.Response("good", "It is hot and humid: it is summer in Brazil.", label="positive")])

model = ca.ToyModel.demo()
e = ca.explain(prompt, model, ca.Contrast(), method="kernelshap", budget=8)
print(e)
```

The [README](../README.md) explains each piece of this code. Three words are used from here on:

- a *unit* is a part of the prompt that may be removed: `d1`, `d2` and `d3`;
- the *target*, here `ca.Contrast()`, says which number is read from the model (section 2);
- the *budget* is how many masked copies of the prompt the model is asked about.

`model` is `ca.ToyModel.demo()`, a stand-in whose behaviour is known exactly. Yours is the wrapper around your model ([wrapper.md](wrapper.md)).

## 2. The value: the number every score is about

A *mask* is one choice of which units to keep. For each mask, ContextAttr builds the prompt with only those units and reads one number from the model, the *value*. The target decides what that number is. With `ca.Contrast()` it is:

> the mean log-probability per token of the negative answers, minus that of the positive answers

A *log-probability* says how likely the model finds a text: 0 is certain, and the more negative, the less likely. A *token* is the piece of text a model reads at a time, about a word. So a higher value means the model leans more towards the unwanted answer. Three units give 2³ = 8 masks, and these are their values in the example:

| Units kept | Value |
|---|---|
| `d1`, `d2`, `d3` (the full prompt) | −0.059 |
| `d1`, `d2` | −0.309 |
| `d1`, `d3` | +0.250 |
| `d2`, `d3` | −0.012 |
| `d1` | 0.000 |
| `d2` | −0.262 |
| `d3` | +0.250 |
| none | 0.000 |

Read from the table:

- `d3` (the winter sentence) raises the value by 0.25 whenever it is there: it pushes the model towards "cold".
- `d2` (the summer sentence) lowers it by 0.262: it pushes the model towards "hot".
- `d1` (Carnival is in February or March) does nothing alone. With `d2` it lowers the value by a further 0.047 (−0.309 against −0.262): the two sentences matter *together*, because only together do they say that Carnival falls in summer.

A score is a summary of this table. A real prompt has far too many masks to ask about all of them (2ⁿ for n units), so each method asks about some and estimates the rest. The scores also predict the value of any mask: start from a base number and add the scores of the units kept. R² (section 6) says how close that prediction comes to what the model really gives.

**A score is measured in the same thing as the value**, so what a score of +0.25 means depends on the target. No fixed size counts as large: compare a score with the noise and with the other scores (section 6).

| Target | The value, and so each score, is in |
|---|---|
| `ca.LogProb()` | log-probability per token of the answer (natural logarithm). With `reduce="sum"`, of the whole answer |
| `ca.Pooled(label)` | the same, averaged over the answers with that label |
| `ca.Contrast()` | a difference of two such log-probabilities: negative answers minus positive ones |
| `ca.Choice(a, b)` | log p(a) − log p(b): how much more likely answer `a` is than `b` |
| `ca.Sampled(check)` | a share of replies, between 0 and 1: +0.25 means 25 percentage points more replies flagged by the check (section 8) |

## 3. The printed result

<!-- output: rio-kernelshap -->
```
rio · KernelSHAP(paired=True) · 8 masks · R² 0.991 on 128 test masks · noise 0 · 22 requests
  d3              +0.2500  Winter in Brazil runs from June to September.
  d1              -0.0234  Rio Carnival takes place in February or March.
  d2              -0.2854  Summer in Brazil runs from December to March.
```

The first line:

| Part | Meaning |
|---|---|
| `rio` | the prompt's id |
| `KernelSHAP(paired=True)` | the method and its settings |
| `8 masks` | how many masks the method used |
| `R² 0.991 on 128 test masks` | how well the scores predict the model on 128 other masks. 1 is perfect (section 6) |
| `noise 0` | how much the value of the full prompt changes when the model is asked again. Scores smaller than this mean nothing (section 6) |
| `22 requests` | what it cost: each request is one masked prompt with one answer to score. Here 8 different masked prompts × 2 answers, and 6 more to measure the noise. A mask that comes up twice costs nothing more: the answer is remembered |

Then one row per unit, highest score first: the unit's id, its score, and its text.

- **A positive score pushes the value up.** With `Contrast` that is towards the unwanted answers: `d3` is what makes this model say "cold".
- **A negative score pushes it down**: `d2` protects against the unwanted answer.
- **A score near zero** means the unit hardly matters for this value.

Interaction methods add rows marked `(together)` (section 4), and `removal=` adds rows that begin with `removing` (section 6).

## 4. What a score means, method by method

Every method ranks the three units the same way here, but the numbers differ, because each method answers a slightly different question. These are the scores of all the methods on the example, each from the call of section 1 with another `method=` (`spex` with `budget=4032`: it needs budgets of its own, [methods.md](methods.md) §2):

| `method=` | `d1` | `d2` | `d3` | `d1` and `d2` together | R² | A unit's score is |
|---|---|---|---|---|---|---|
| `loo` | −0.047 | −0.309 | +0.250 | | 0.984 | how much the value falls when that unit alone is removed from the full prompt |
| `banzhaf` | −0.023 | −0.285 | +0.250 | | 0.996 | the unit's effect, averaged over every combination of the other units |
| `kernelshap` | −0.023 | −0.285 | +0.250 | | 0.991 | its Shapley value: a fair share of the difference between the full prompt and the empty one. The scores add up to that difference (here −0.059) |
| `contextcite` | −0.007 | −0.268 | +0.238 | | 0.991 | as `banzhaf`, pulled towards zero, so that units that hardly matter get exactly 0 |
| `lime` | −0.021 | −0.252 | +0.230 | | 0.987 | as `banzhaf`, but masks close to the full prompt count more, and the scores are shrunk a little |
| `faithbanzhaf`, `faithshap`, `spex` | 0.000 | −0.262 | +0.250 | −0.047 | 1.000 | the unit's effect on its own. What units do together is a separate row |
| `proxyspex` | needs a longer prompt: section 5 | | | | | as `spex`, estimated from fewer masks |

`d1` shows the difference. Its effect exists only together with `d2`, and the methods account for that in three ways:

- **`loo`** gives the whole joint effect (−0.047) to `d1`, and again to `d2`, since removing either one loses it. Its scores can therefore count a joint effect twice.
- **`banzhaf` and `kernelshap`** split the joint effect evenly: half (−0.023) goes to `d1` and half to `d2`.
- **The interaction methods** keep it apart: `d1` alone scores 0, `d2` alone −0.262, and the row `d1 & d2` holds the −0.047.

### Leave-one-out: `loo`

<!-- output: rio-loo -->
```
rio · LeaveOneOut() · 4 masks · R² 0.984 on 128 test masks · noise 0 · 22 requests
  d3              +0.2500  Winter in Brazil runs from June to September.
  d1              -0.0469  Rio Carnival takes place in February or March.
  d2              -0.3089  Summer in Brazil runs from December to March.
```

It uses n + 1 masks, the full prompt and one mask per unit, so `budget` does not apply. The cheapest look, and easy to explain. Its weakness: when two units say the same thing, removing one changes little, and both get a small score.

### One score per unit: `banzhaf`, `kernelshap`, `contextcite`, `lime`

<!-- output: rio-contextcite -->
```
rio · ContextCite(alpha=0.01, standardize=True) · 64 masks · R² 0.991 on 128 test masks · noise 0 · 22 requests
  d3              +0.2375  Winter in Brazil runs from June to September.
  d1              -0.0074  Rio Carnival takes place in February or March.
  d2              -0.2682  Summer in Brazil runs from December to March.
```

These four remove many units at once, and then find the one number per unit that, added up over the units kept, comes closest to the values the model gave. Read a score as "the value is this much higher, on average, when the unit is there". `contextcite` and `lime` trade a little accuracy in each number for steadier results from few masks, which is why their scores here are slightly smaller than the exact −0.285 and +0.250.

### Scores for units and for sets of units: `faithbanzhaf`, `faithshap`, `spex`, `proxyspex`

<!-- output: rio-spex -->
```
rio · SPEX(degree=5, chase_depth=10) · 4032 masks · R² 1 on 128 test masks · noise 0 · 22 requests
  d3              +0.2500  Winter in Brazil runs from June to September.
  d1              +0.0000  Rio Carnival takes place in February or March.
  d2              -0.2620  Summer in Brazil runs from December to March.
  d1 & d2         -0.0469  (together)
```

A row marked `(together)` is an *interaction*: what the units add when all of them are in the prompt, beyond their own scores. To get the value these scores predict for any choice of units, add the scores of the units kept and the interactions whose units are all kept. For the full prompt: 0.000 − 0.262 + 0.250 − 0.047 = −0.059, the value in section 2.

- A negative interaction means the units push the value down more together than apart, as here. A positive one means they push it up more together.
- When two units say the same thing, each scores high alone and their interaction has the opposite sign: together they add less than their two scores suggest.
- To compare with the one-score-per-unit methods, `e.shapley()` and `e.banzhaf()` fold the interactions back into one number per unit. Here both give `d1` −0.023, `d2` −0.285 and `d3` +0.250, the `kernelshap` scores.

`faithbanzhaf` and `faithshap` score every pair (or every set up to `max_order`), so they print many interactions that are zero; `spex` and `proxyspex` keep only the few that matter.

## 5. A longer prompt

The same three sentences, with nine more that say nothing about the weather:

```python
documents = [
    "Rio Carnival takes place in February or March.", "Summer in Brazil runs from December to March.",
    "Winter in Brazil runs from June to September.", "Brazil's capital is Brasília.",
    "Portuguese is the official language of Brazil.", "The Amazon River flows through the north of Brazil.",
    "Samba schools parade in the Sambadrome.", "Rio de Janeiro lies on the Atlantic coast.",
    "The real is Brazil's currency.", "Sugarloaf Mountain overlooks Guanabara Bay.",
    "Brazil has won the football World Cup five times.", "Coffee is one of Brazil's main exports."]
long_prompt = ca.Prompt("rio-12", messages=[{"role": "user", "content": [
    *[ca.Unit(f"d{i}", text, sep="\n") for i, text in enumerate(documents, start=1)],
    "Question: What is the weather like during Rio Carnival?"]}],
    responses=prompt.responses)
print(ca.explain(long_prompt, model, ca.Contrast(), method="contextcite", budget=64))
```

`contextcite`, the default method, with 64 masks. The nine sentences that do not matter score exactly zero:

<!-- output: rio12-contextcite -->
```
rio-12 · ContextCite(alpha=0.01, standardize=True) · 64 masks · R² 0.995 on 128 test masks · noise 0 · 384 requests
  d3              +0.2404  Winter in Brazil runs from June to September.
  d4              -0.0000  Brazil's capital is Brasília.
  d5              +0.0000  Portuguese is the official language of Brazil.
  d6              -0.0000  The Amazon River flows through the north of Brazil.
  d7              +0.0000  Samba schools parade in the Sambadrome.
  d8              +0.0000  Rio de Janeiro lies on the Atlantic coast.
  d9              -0.0000  The real is Brazil's currency.
  d10             -0.0000  Sugarloaf Mountain overlooks Guanabara Bay.
  d11             +0.0000  Brazil has won the football World Cup five times.
  d12             +0.0000  Coffee is one of Brazil's main exports.
  d1              -0.0163  Rio Carnival takes place in February or March.
  d2              -0.2734  Summer in Brazil runs from December to March.
```

`lime` with the same 64 masks. Its R² is lower, and the sentences that do not matter get small scores that are only scatter: `d12` (coffee) even ranks below `d1`. Scores of that size, next to others just like them, mean nothing:

<!-- output: rio12-lime -->
```
rio-12 · LIME(kernel_width=25.0, alpha=1.0) · 64 masks · R² 0.953 on 128 test masks · noise 0 · 372 requests
  d3              +0.2022  Winter in Brazil runs from June to September.
  d9              +0.0194  The real is Brazil's currency.
  d4              +0.0181  Brazil's capital is Brasília.
  d10             +0.0140  Sugarloaf Mountain overlooks Guanabara Bay.
  d5              +0.0069  Portuguese is the official language of Brazil.
  d11             +0.0056  Brazil has won the football World Cup five times.
  d7              +0.0004  Samba schools parade in the Sambadrome.
  d8              -0.0025  Rio de Janeiro lies on the Atlantic coast.
  d6              -0.0129  The Amazon River flows through the north of Brazil.
  d1              -0.0183  Rio Carnival takes place in February or March.
  d12             -0.0274  Coffee is one of Brazil's main exports.
  d2              -0.2459  Summer in Brazil runs from December to March.
```

`spex` with 5,376 masks finds the two sentences that matter alone and the pair that matters together, and nothing else (the rows of the nine other sentences, all zero, are left out here):

<!-- output: rio12-spex -->
```
rio-12 · SPEX(degree=5, chase_depth=10) · 5376 masks · R² 1 on 128 test masks · noise 0 · 5944 requests
  d3              +0.2500  Winter in Brazil runs from June to September.
  d1              +0.0000  Rio Carnival takes place in February or March.
  d2              -0.2620  Summer in Brazil runs from December to March.
  d1 & d2         -0.0469  (together)
```

`proxyspex` with 512 masks, a tenth of the cost, finds the two strong sentences. It misses the weak pair and puts its effect into `d2` (the ten rows of zeros are left out here):

<!-- output: rio12-proxyspex -->
```
rio-12 · ProxySPEX(max_order=5, hpo=True) · 512 masks · R² 0.992 on 128 test masks · noise 0 · 1240 requests
  d3              +0.2496  Winter in Brazil runs from June to September.
  d2              -0.2860  Summer in Brazil runs from December to March.
```

`proxyspex` needs a prompt of 8 units or more ([methods.md](methods.md) §2 says why): on the three-unit example it has only 8 masks to learn from, finds nothing, and reports all scores as zero with R² −0.008. That is what a failed run looks like, and why R² is the first thing to read. While it runs, LightGBM prints progress lines of its own; they can be ignored.

## 6. How far to trust a result

| Look at | It tells you | Rule of thumb |
|---|---|---|
| `e.r2` | how well the scores predict the model on masks the method did not use. 1 is perfect; 0 is no better than always guessing the average; below 0 is worse than that | near 1: use the scores. Near 0 or below: do not. In between: trust the order of the top units more than the size of small scores |
| `e.noise["std"]` | how much the value of the full prompt moves when the model is asked again. 0 for a model that always answers the same | a score smaller than the noise means nothing |
| scores of the same size on units that cannot matter | the scatter of the method itself, as `lime` shows in section 5 | only scores that stand clear of the rest mean something |
| `removal=(1, 2, 3)` | the value is measured again without the 1, 2 and 3 units whose removal should change it most, up or down | if the value hardly moves, those units are not what drives it |
| `ca.intervene(...)` | the model generates fresh replies with and without the top units | the final check, on the behaviour itself (section 8) |

**When R² is low.** A real model gives less than this toy example: with Llama-3.2-3B-Instruct on a prompt of 20 documents, R² was between 0.81 and 0.99, depending on the method. Well below that, try these in order:

1. **More masks.** Double `budget` and see whether R² rises. On the 12-sentence prompt of section 5, `lime` has R² 0.49 with 16 masks, 0.95 with 64 and 0.99 with 128.
2. **A steadier value.** If the noise in the first line is above 0, or the target is `ca.Sampled`, the values themselves are shaky. Raise `samples`, or use a scored target if your wrapper can score.
3. **A method that scores units together.** With one score per unit, R² stays low when units matter only together. On a real model (Qwen2.5-0.5B-Instruct, the three sentences of section 1), `kernelshap` had R² 0.73 and `spex` 1.00.
4. **The right method for the size.** `proxyspex` gives all zeros on a prompt of fewer than 8 units ([methods.md](methods.md) §2). `loo` and `lime` often have a lower R² than the others, as in sections 4 and 5, even when their order of the units is right.

If R² stays low, use only the order of the top units, and confirm it with `removal=` or `ca.intervene` before you act on it. `R² n/a` with test masks means that every one of them gave the same value: nothing you marked changes this answer.

With `removal=(1, 2)`, the result ends with:

<!-- output: rio-removal -->
```
  removing d2: the value moves by +0.3089 (Rem 5.25)
  removing d1, d2: the value moves by +0.3089 (Rem 5.25)
```

Each row names the units whose removal changes the result most, and how far the value really moved when the model was asked without them. `Rem` is that change divided by the size of the full prompt's value (the measure the SPEX paper reports). When the full prompt's value is near zero, as with `Contrast` here, read the change and not `Rem`.

## 7. Everything a result holds

| Use | Gives |
|---|---|
| `print(e)`, `e.table()` | the printed result of section 3 |
| `e.scores` | `{unit id: score}`, for example `{"d1": -0.023, "d2": -0.285, "d3": 0.25}` |
| `e.top(k)` | the `k` units with the highest scores, as `[(unit id, score), ...]` |
| `e.interactions` | `{(unit ids): score}` for every set of two or more units (interaction methods only) |
| `e.top_interactions(k)` | the `k` interactions that are largest in size |
| `e.r2` | R² (section 6) |
| `e.noise` | `{"values": [...], "std": ...}`: the full prompt's value asked again, and the spread |
| `e.masks`, `e.test_masks` | how many masks the method used, and how many more R² was measured on |
| `e.requests` | `{"requests": ..., "reused": ..., "calls": ...}`: requests sent to the model, answers taken from the cache instead, and calls of the wrapper |
| `e.removal` | the removal rows of section 6, as a list |
| `e.shapley()`, `e.banzhaf()` | one number per unit from any method's result, interactions folded in |
| `e.shapley_taylor(order)` | for specialists: the Shapley-Taylor interaction index of that order |
| `e.to_html(path)` | a page showing the prompt with each unit shaded by its score |
| `e.as_dict()` | everything above as a dictionary, to save as JSON; `ca.Explanation.from_dict` reads it back |

## 8. When the model can only generate: `ca.Sampled`

With a wrapper that has no `score`, the value is counted from replies. `ca.Sampled(check, samples=8)` asks for 8 fresh replies to each masked prompt and takes the share of them that the check flags:

```python
cold = ca.checks.regex(r"\bcold\b")          # a check: flags a reply that says "cold"
e = ca.explain(prompt, model, ca.Sampled(cold, samples=8), method="loo")
print(e)
```

<!-- output: rio-sampled -->
```
rio · LeaveOneOut() · 4 masks · R² 0.588 on 128 test masks · noise 0.12 · 11 requests
  d3              +0.3750  Winter in Brazil runs from June to September.
  d1              +0.1250  Rio Carnival takes place in February or March.
  d2              -0.6250  Summer in Brazil runs from December to March.
```

- The scores are shares of replies: without `d3`, 37.5 percentage points fewer replies say "cold"; without `d2`, 62.5 points more do.
- The noise is 0.12, because a share of 8 replies changes from one draw to the next. `d1`'s +0.125 is the size of the noise: it means nothing. More `samples` lower the noise; each sample costs one more generation per masked prompt.
- R² is lower for the same reason: the values the scores are compared with are themselves noisy.

To confirm a result on the behaviour itself, remove the top units and let the model answer again, many times:

```python
ca.intervene(prompt, model, cold, ["d3"], samples=64)
```

gives `{"dropped": ["d3"], "before": 0.531, "after": 0.0, "change": -0.531, "se": 0.063, ...}`: 53 % of 64 replies said "cold" with the full prompt and none did without `d3`. `se` is the standard error of the change; a change of several times `se` is real.

## 9. From scores to a better prompt

The scores are for acting on: find the part that pushes the model towards the answers you do not want, change it, and measure again.

**1. Give several answers of each kind.** With one wanted and one unwanted answer, the scores are about those two wordings. With several, `ca.Contrast()` takes the mean over the unwanted answers and the mean over the wanted ones, so the scores say what moves the model between the two kinds of answer. Each answer costs one more request per mask.

```python
answers = [ca.Response("bad-1", "It is cold, because Carnival is in winter.", label="negative"),
           ca.Response("bad-2", "Cold: bring a winter coat.", label="negative"),
           ca.Response("good-1", "It is hot and humid: it is summer in Brazil.", label="positive"),
           ca.Response("good-2", "Hot, since Carnival falls in the summer.", label="positive")]
several = ca.Prompt("rio", messages=prompt.messages, responses=answers)
print(ca.explain(several, model, ca.Contrast(), method="kernelshap", budget=8))
```

<!-- output: rio-several -->
```
rio · KernelSHAP(paired=True) · 8 masks · R² 0.99 on 128 test masks · noise 0 · 44 requests
  d3              +0.3250  Winter in Brazil runs from June to September.
  d1              -0.0285  Rio Carnival takes place in February or March.
  d2              -0.3023  Summer in Brazil runs from December to March.
```

- You can also let the model write the answers: `ca.sample(prompt, model, ca.Sampling(n=8, temperature=0.7), check)` adds eight of its own replies, each labelled by the check ([interface.md](interface.md) §4).
- How many answers are enough has not been measured. Use the ones that matter to you, in different words, and see whether the order of the parts stays the same when you add more.

**2. Read which part to change.** R² is 0.99 and the noise is 0, so the scores can be used (section 6). `d3` has the highest score: it pushes this model towards "cold". `d2` has the lowest: it protects against it, so it stays.

**3. Change the prompt and measure again.** Here `d3` is removed; rewriting it or moving it are other changes to try. The value of the whole prompt, before and after:

```python
better = ca.Prompt("rio-better", messages=[{"role": "user", "content": [
    ca.Unit("d1", "Rio Carnival takes place in February or March.", sep="\n"),
    ca.Unit("d2", "Summer in Brazil runs from December to March.", sep="\n"),
    "Question: What is the weather like during Rio Carnival?"]}], responses=answers)
before = ca.Value(several, model, ca.Contrast())([[1, 1, 1]])[0]
after = ca.Value(better, model, ca.Contrast())([[1, 1]])[0]
print(f"before {before:+.3f}, after {after:+.3f}")
```

<!-- output: rio-better -->
```
before -0.006, after -0.331
```

The value is how much more likely the model finds the unwanted answers than the wanted ones. It fell from −0.006 to −0.331: the change did what the scores said.

**4. Confirm it on the behaviour** with `ca.intervene` (section 8): 53 % of the replies say "cold" with `d3`, and none without it.

**5. Do it over many prompts.** One prompt can mislead. Give the parts a `tag` (`"doc"`, `"instruction"`, `"example"`), run a file of prompts, and read the summary by tag ([interface.md](interface.md) §8): it says which kind of part pushes towards the unwanted answers across all of them, and that is the kind to rewrite.

Two more things:

- A part with a score near zero can often be dropped, which makes the prompt shorter and cheaper. Measure the shorter prompt to check.
- A change that helps one prompt can hurt another: measure again on prompts you did not look at.
