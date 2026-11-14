# Reading a result

How to read what ContextAttr prints. In all the docs, `ca` is the package: `import contextattr as ca`.

## The rule

ContextAttr removes parts of the prompt and watches how the model's leaning towards an answer changes. Each part, called a *unit*, then gets one number, its score:

| Score | The unit |
|---|---|
| above 0 | pushes the model towards the answer you are explaining |
| below 0 | pushes the model away from it |
| near 0 | hardly matters |

The further a score is from 0, the stronger the push. The second line of every printed result repeats this for its own case.

The three cases below show it. They are the cases of `contextattr demo`: each has 8 short documents about an invented place and one question, so the answer can only come from the documents, and the right result is known in advance. Each explains the right answer.

```python
import contextattr as ca
from contextattr.demo import case, planted_toy

model = planted_toy()        # the demo's stand-in; put your own wrapper here
```

The stand-in follows fixed rules written for these cases. It is not a language model: its numbers show how to read a result, not how any real model behaves.

## 1. One document holds the answer

Only `d2` says when the bridge was opened. We ask which documents make the answer "1932" likely:

```python
prompt = case(1, [ca.Response("answer", "1932")])
e = ca.explain(prompt, model, ca.LogProb(), method="banzhaf", budget=256)
print(e)
```

<!-- output -->
```
planted-1 · Banzhaf() · 256 masks · R² 1 on 128 test masks · noise 0 · 259 requests
  above 0: the unit makes the answer more likely · below 0: less likely
  d2              +3.0000  The Velmar Bridge was opened in 1932.
  d1              +0.0000  The town of Velmar lies on the river Osk, forty kilometres …
  d3              +0.0000  Most visitors reach Velmar by the regional train from the c…
  d4              +0.0000  The town library of Velmar was rebuilt after a fire.
  d5              +0.0000  The Osk valley is known for its orchards and its cider.
  d6              +0.0000  The market of Velmar is held on the square every Saturday.
  d7              +0.0000  The river Osk freezes over in most winters.
  d8              +0.0000  The football club of Velmar plays in green and white.
```

**The second line** is the rule for this case: the answer "1932" is explained, so a score above 0 means the unit makes that answer more likely.

**The rows.** One row for each part of the prompt (a *unit*), highest score first: its id, its score and its text.

- `d2` has a score of +3.0: with `d2` in the prompt, the model finds "1932" much more likely. This is the document the answer comes from.
- The other seven score 0: the answer does not depend on them.

A score counts in *log-probability*, the model's own measure of how likely it finds the answer: 0 is certain, and the more negative, the less likely.

**The first line:**

| Part | Meaning |
|---|---|
| `planted-1` | the prompt's id |
| `Banzhaf()` | the method and its settings ([methods.md](methods.md)) |
| `256 masks` | how many copies of the prompt, each with some units removed, the method asked about. A *mask* is one such choice of units. Here `budget=256` covers every combination of the 8 documents |
| `R² 1 on 128 test masks` | how well the scores predict the model: 1 is perfect (section 5) |
| `noise 0` | how much the model's number changes when it is asked the same thing again. A score smaller than this means nothing |
| `259 requests` | what it cost: one request is one masked prompt with one answer to score |

## 2. Two documents hold the answer together

`d3` gives the month of the fair. `d2` says that the festival is in the same month as the fair. Neither names the festival's month alone.

```python
prompt = case(2, [ca.Response("answer", "October")])
print(ca.explain(prompt, model, ca.LogProb(), method="banzhaf", budget=256))
```

<!-- output -->
```
planted-2 · Banzhaf() · 256 masks · R² 0.986 on 128 test masks · noise 0 · 259 requests
  above 0: the unit makes the answer more likely · below 0: less likely
  d3              +2.6777  The Apple Fair of Senn takes place every October.
  d2              +0.3223  The Lantern Festival of Qarim is held in the same month as …
  d1              +0.0000  The town of Qarim lies on the river Senn.
  d4              +0.0000  The town library of Qarim was rebuilt after a fire.
  d5              +0.0000  Most visitors reach Qarim by the regional train from the ca…
  d6              +0.0000  The river Senn floods the lower town in some springs.
  d7              +0.0000  The market hall of Qarim dates from the last century.
  d8              +0.0000  The football club of Qarim plays in blue.
```

Both documents get a score, and that is all a method with one score per unit can say. It cannot show that `d2` works only through `d3`. A method that also scores units together can. `spex` is one; it needs `pip install "contextattr[interactions]"` and budgets of its own ([methods.md](methods.md) §3):

```python
print(ca.explain(prompt, model, ca.LogProb(), method="spex", budget=4032))
```

<!-- output -->
```
planted-2 · SPEX(degree=5, chase_depth=10) · 4032 masks · R² 1 on 128 test masks · noise 0 · 259 requests
  above 0: the unit makes the answer more likely · below 0: less likely
  d3              +2.3554  The Apple Fair of Senn takes place every October.
  d1              +0.0000  The town of Qarim lies on the river Senn.
  d2              +0.0000  The Lantern Festival of Qarim is held in the same month as …
  d4              +0.0000  The town library of Qarim was rebuilt after a fire.
  d5              +0.0000  Most visitors reach Qarim by the regional train from the ca…
  d6              +0.0000  The river Senn floods the lower town in some springs.
  d7              +0.0000  The market hall of Qarim dates from the last century.
  d8              +0.0000  The football club of Qarim plays in blue.
  d2 & d3         +0.6446  (together)
```

- `d3` alone: +2.36. The model already leans towards "October" when it sees the month of the fair.
- `d2` alone: 0. Without `d3` it says nothing about the month.
- `d2 & d3 (together)`: +0.64. This is what the two add when both are there, beyond their own scores.

A row marked `(together)` is called an *interaction*. A positive one means the units do more together than apart. A negative one means they overlap: each would do alone what both do together.

The first table shared that +0.64 between the two documents, half each: 2.36 + 0.32 = 2.68 for `d3` and 0.32 for `d2`. The simplest method, `loo`, gives the whole +0.64 to both, so it counts a joint effect twice. [methods.md](methods.md) §1 says which method to use, and its §2 works these numbers out by hand.

## 3. Two documents disagree

`d2` gives the tower's old height, 310 metres. `d4` says that it is now 450 metres. Here we explain the right answer, "450", against the outdated one, "310":

```python
prompt = case(3)
print(ca.explain(prompt, model, ca.Choice("450", "310"), method="banzhaf", budget=256))
```

<!-- output -->
```
planted-3 · Banzhaf() · 256 masks · R² 0.983 on 128 test masks · noise 0 · 518 requests
  above 0: the unit pushes towards '450' · below 0: towards '310'
  d4              +6.0066  Since its new spire was added last year, the Brask Tower is…
  d1              +0.0000  The Brask Tower stands in the old harbour of Brask.
  d3              +0.0000  The harbour of Brask is known for its fish market.
  d5              +0.0000  Most visitors reach Brask by ferry.
  d6              +0.0000  The old harbour of Brask is closed to cars.
  d7              +0.0000  The lighthouse of Brask is painted red and white.
  d8              +0.0000  The museum of Brask opens every day except Monday.
  d2              -3.8978  When it was built, the Brask Tower was 310 metres tall.
```

The rule is the same, and the second line spells it out for this case:

- `d4` is above 0: it pushes the model towards "450".
- `d2` is below 0: it pushes the model towards "310", the outdated answer.

## 4. What you explain: one answer, wanted answers, unwanted answers, or both

So far each case explained one right answer. You can also label answers: `positive` for an answer you want, `negative` for one you do not want. The same case, explained in the four ways:

```python
right = ca.Response("right", "450", label="positive")
outdated = ca.Response("outdated", "310", label="negative")
ways = [("one answer, no label", [ca.Response("answer", "450")], ca.LogProb()),
        ("wanted answers", [right], ca.Pooled("positive")),
        ("unwanted answers", [outdated], ca.Pooled("negative")),
        ("both: the contrast", [right, outdated], ca.Contrast())]
for name, answers, target in ways:
    e = ca.explain(case(3, answers), model, target, method="banzhaf", budget=256)
    print(f"{name:<22} d2 {e.scores['d2']:+.2f}   d4 {e.scores['d4']:+.2f}")
```

<!-- output -->
```
one answer, no label   d2 +0.00   d4 +3.03
wanted answers         d2 +0.00   d4 +3.03
unwanted answers       d2 +3.90   d4 -2.98
both: the contrast     d2 -3.90   d4 +6.01
```

| You explain | With | A score above 0 means the unit | What the case shows |
|---|---|---|---|
| one answer, no label | `ca.LogProb()` | makes that answer more likely | `d4` supports "450". `d2` scores 0: nothing here says it is a problem |
| wanted answers | `ca.Pooled("positive")` | makes the wanted answers more likely | the same, for all the answers you labelled `positive` together |
| unwanted answers | `ca.Pooled("negative")` | makes the unwanted answers more likely. This is the one to ask what causes a mistake | `d2` is what makes the model say "310", and `d4` protects against it |
| both | `ca.Contrast()` | pushes the model towards the wanted answers, and away from the unwanted ones | `d4` pushes towards "450". `d2`, below 0, pushes towards "310" |

**What the contrast shows.** It is one number for each unit: how much the unit raises the wanted answers, minus how much it raises the unwanted ones. So it finds everything that moves the model between the two kinds of answer. `d4` gets +6.01: +3.03 for making "450" more likely, and +2.98 for making "310" less likely. `d2` gets −3.90: it is the unit to fix. Explaining the wanted answer alone gave `d2` a score of 0 and missed it; the contrast does not. Use it whenever you have answers of both kinds.

The rule holds here as everywhere: above 0 pushes towards what is explained, the wanted answers. The units to look at when the model goes wrong are the ones with the lowest scores.

Two more targets follow the same rule: `ca.Choice(a, b)`, as in section 3, where above 0 pushes towards `a`; and `ca.Sampled(check)`, where above 0 means more replies get flagged by the check (section 7). No fixed size counts as large: compare a score with the other scores and with the noise.

## 5. How far to trust a result

| Look at | It tells you | Rule of thumb |
|---|---|---|
| `R²` in the first line | how well the scores predict the model's numbers on a separate draw of masks. 1 is perfect; 0 is no better than guessing the average | near 1: use the scores. Near 0 or below: do not. In between: trust the order of the top units more than the size of small scores |
| `noise` in the first line | how much the model's number moves when asked again | a score smaller than the noise means nothing |
| small scores on units that cannot matter | the scatter of the method itself | only scores that stand clear of the rest mean something |
| `removal=(1, 2)` in `ca.explain` | the 1 and the 2 units whose removal should change the model's number most are removed, and the model is asked again | if its number hardly moves, those units are not what drives it |
| `ca.intervene(...)` | the model writes fresh replies with and without the top units | the final check, on the behaviour itself (section 7) |

**An example of scatter.** `lime` on the first case, with few masks, gives small scores to documents that do not matter, and R² shows it. More masks cure it:

| `budget` | R² | score of `d2`, which holds the answer | largest score among the seven others |
|---|---|---|---|
| 24 | 0.72 | +1.60 | 0.55 |
| 64 | 0.92 | +2.27 | 0.32 |
| 128 | 0.99 | +2.71 | 0.09 |
| 256 | 1.00 | +2.87 | 0.03 |

**When R² is low**, try these in order:

1. **More masks.** Double `budget` and see whether R² rises, as in the table above.
2. **A steadier value.** If the noise is above 0, the model's numbers are shaky. With `ca.Sampled`, raise `samples`.
3. **A method that scores units together**, such as `spex`. With one score per unit, R² stays low when units matter only together.
4. **The right method for the size** ([methods.md](methods.md) §1).

If R² stays low, use only the order of the top units, and confirm it with `removal=` or `ca.intervene` before you act on it. On a real model R² is lower than on this stand-in: with Llama-3.2-3B-Instruct and 20 documents it was between 0.81 and 0.99, depending on the method.

## 6. Everything a result holds

| Use | Gives |
|---|---|
| `print(e)` | the printed result |
| `e.scores` | `{unit id: score}` |
| `e.top(k)` | the `k` units with the highest scores |
| `e.interactions` | `{(unit ids): score}` for the rows marked `(together)` |
| `e.r2`, `e.noise` | R², and the noise with the repeated values behind it |
| `e.requests` | requests sent to the model, and answers reused from the cache |
| `e.removal` | the rows of `removal=`: which units were removed and how far the model's number moved |
| `e.shapley()`, `e.banzhaf()` | one number per unit, with the rows marked `(together)` folded in |
| `e.to_html(path)` | a page showing the prompt with each unit shaded by its score |
| `e.as_dict()` | everything above, to save as JSON; `ca.Explanation.from_dict` reads it back |

## 7. Counting replies, and confirming a result

The scores above read the model's log-probabilities. You can also count what the model writes. A *check* looks at a reply; `ca.intervene` removes units and lets the model answer again, many times:

```python
right = ca.checks.regex(r"450")               # a check: flags a reply that gives the current height
found = ca.intervene(case(3), model, right, ["d2"], samples=64)           # remove d2, ask 64 times
print({k: found[k] for k in ("before", "after", "change", "se")})
```

<!-- output -->
```
{'before': 0.5625, 'after': 1.0, 'change': 0.4375, 'se': 0.0625}
```

With the full prompt, 56 % of 64 replies gave the current height. Without `d2`, all of them did. `se` is the standard error of the change: a change of several times `se` is real. This confirms on the behaviour itself what the score of `d2` said in section 3: it pushes the model away from the right answer.

Replies can also be the thing explained. `ca.Sampled(check, samples=8)` asks for 8 replies to each masked prompt and takes the share that the check flags:

```python
print(ca.explain(case(3), model, ca.Sampled(right, samples=8), method="banzhaf", budget=64))
```

<!-- output -->
```
planted-3 · Banzhaf() · 64 masks · R² 0.877 on 128 test masks · noise 0.188 · 135 requests
  above 0: the unit makes more replies get flagged by the check · below 0: fewer
  d4              +0.8040  Since its new spire was added last year, the Brask Tower is…
  d8              +0.0633  The museum of Brask opens every day except Monday.
  d3              +0.0274  The harbour of Brask is known for its fish market.
  d6              -0.0008  The old harbour of Brask is closed to cars.
  d7              -0.0120  The lighthouse of Brask is painted red and white.
  d1              -0.0264  The Brask Tower stands in the old harbour of Brask.
  d5              -0.0401  Most visitors reach Brask by ferry.
  d2              -0.2062  When it was built, the Brask Tower was 310 metres tall.
```

The picture is the same as in section 3, `d4` at the top and `d2` at the bottom, but rougher. A share of 8 replies changes from one draw to the next: the noise is 0.19, so the small scores of the six other documents mean nothing, and R² is lower. This costs 8 generations for every mask, so use it when replies are what you care about, and the log-probability targets otherwise.

## 8. From scores to a better prompt

The scores are for acting on: find the part that pushes the model away from the answer you want, change it, and measure again.

1. **Read which part to change.** In section 3, `d2` has a score below 0: it pushes the model away from the right answer.
2. **Change the prompt and measure again.** Here `d2` is removed; rewriting it is another change to try. `ca.Value` gives the model's number for any choice of units:

```python
value = ca.Value(case(3), model, ca.Choice("450", "310"))
before, after = value([[1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 1, 1, 1, 1, 1, 1]])      # 1 keeps a document, 0 removes it
print(f"before {before:+.2f}, after {after:+.2f}")
```

<!-- output -->
```
before +2.11, after +6.98
```

   The model's lean towards "450" rose, as the scores said it would.
3. **Confirm it on the behaviour**, as in section 7: every reply now gives the current height.
4. **Do it over many prompts.** One prompt can mislead. Give the parts a `tag` (`"doc"`, `"instruction"`, `"example"`), run a file of prompts, and read the summary by tag ([interface.md](interface.md) §8): it shows which kind of part helps or hurts across all of them.

A part with a score near zero can often be dropped, which makes the prompt shorter and cheaper. Measure the shorter prompt to check.
