# Methods and settings

Which method to use, exactly how each one computes its scores, how many masks to give it, and what every setting means. [results.md](results.md) shows how to read a result.

Words used here (`ca` is the package: `import contextattr as ca`):

| Word | Meaning |
|---|---|
| unit | a part of the prompt that may be removed. `n` is the number of units in a prompt |
| mask (`m`) | one choice of units to keep: one entry per unit, 1 to keep it and 0 to remove it. A prompt of n units has 2ⁿ masks |
| value (`f(m)`) | the number read from the model for the prompt with only the units that mask `m` keeps. The *target* decides which number ([interface.md](interface.md) §5) |
| method | decides which masks to ask about, and works out the scores from their values |
| budget | the number of masks a method may use for one prompt. Each mask costs model calls, so the budget is the cost |
| R² | how well the scores predict the value on test masks (section 2): 1 is perfect, 0 or below means the scores should not be trusted ([results.md](results.md) §5 says what to do then) |

## 1. Which method

| `method=` | A unit's score is | Use it for |
|---|---|---|
| `"loo"` | how far the value falls when that unit alone is removed from the full prompt | a first look at a short prompt |
| `"contextcite"` (the default) | its Banzhaf value (next row), estimated so that small scores come out as exactly zero | long prompts cut into sentences or documents |
| `"banzhaf"` | its Banzhaf value: the unit's effect, averaged over every combination of the other units | when every unit should get a score |
| `"lime"` | its effect near the full prompt: masks that remove few units count most | as `banzhaf` |
| `"kernelshap"` | its Shapley value: the unit's effect, averaged over every order of adding the units. The scores add up to the full prompt's value minus the empty prompt's | short prompts |
| `"faithbanzhaf"` | its own effect. Every pair of units gets a score too: what the two do together | prompts below about 64 units whose units may matter only together |
| `"faithshap"` | as `faithbanzhaf`, with the masks weighted as in `kernelshap` | the same |
| `"spex"` | its own effect. The few sets of up to 5 units that matter together get scores too, found without trying every set | prompts of tens to about 2,000 units |
| `"proxyspex"` | as `spex`, read off a tree model trained on about a tenth of the masks | prompts of 8 units or more, on models whose every call is costly |

- The first five come with `pip install contextattr`. The last four need `pip install "contextattr[interactions]"`.
- If unsure: `contextcite` for a first look, `banzhaf` when every unit should get a score, and `spex` (or `faithbanzhaf` on a short prompt) for units that matter only together.
- In the SPEX paper's tests (its section B.5.1), the Banzhaf value and LIME fit the model best among the scores with one number per unit, and Shapley values fit less well as prompts grow. The paper computes them in its own way (its section B.3), not with this package's code.

## 2. How each method computes its scores

### What every method fits

Every method asks the model for the value `f(m)` of some masks. It then fits a formula, the *surrogate*, that gives those values back as closely as it can:

```
one score per unit:    f̂(m) = b + Σ_i w_i·m_i
with interactions:     f̂(m) = b + Σ_i w_i·m_i + Σ_S w_S·Π_{i in S} m_i
```

| Symbol | Is |
|---|---|
| `m_i` | 1 when the mask keeps unit i and 0 when it removes it. So `Σ_i w_i·m_i` adds up the scores of the units kept |
| `w_i` | unit i's **score**: the number printed beside the unit |
| `b` | the base: what the surrogate predicts for the prompt with every unit removed |
| `w_S` | the **interaction** of a set S of two or more units, printed as a `(together)` row. The product `Π_{i in S} m_i` is 1 only when every unit of S is kept, so `w_S` is added only then: what the units do together, beyond their own scores |

*Least squares* means: choose `b` and the `w` that make the squared differences `(f(m) − f̂(m))²`, added up over the masks asked, as small as possible. The methods differ in three things, which the parts below give for each: which masks they ask about, how much each mask counts, and what they add to least squares.

**Exact or estimated.** `loo` is always exact. `banzhaf` (up to 16 units), `kernelshap` and `faithbanzhaf` are exact when the budget covers every one of the 2ⁿ masks (`budget=2**n`): nothing is drawn at random and the result is the quantity itself. `faithshap` then agrees with the exact quantity to about five decimal places: it holds the empty and the full prompt by a very large weight, not exactly. With fewer masks each gives an estimate from a random sample, the same on every run with the same `seed`. `contextcite` and `lime` are never exact: they add a charge on the size of the scores (below), which changes them whatever the number of masks.

**How a result is checked.** `f̂(m)` is what the scores predict for mask `m`.

```
R²    = 1 − Σ (f(m) − f̂(m))² / Σ (f(m) − f̄)²        over the test masks; f̄ is their mean value
noise = the standard deviation of the full prompt's value, asked noise_repeats + 1 times
Rem   = | f(full prompt) − f(prompt without the units) | / | f(full prompt) |
```

- **R²** (SPEX eq. 7) compares the squared error of the predictions with that of always predicting the mean: at R² 0.9 the first is a tenth of the second. The test masks are coin flips, drawn separately from the method's own. On a long prompt they are masks the method did not ask about. On a short one some or all of them are masks it did ask about, since n units have only 2ⁿ masks: R² then says how well the formula describes the values, not how well it predicts new ones.
- **The noise** is 0 for a model that always returns the same numbers. With `ca.Sampled` each asking draws fresh replies, so it is above 0.
- **Rem** (SPEX eq. 8) belongs to `removal=`. The units removed are chosen from the scores alone: of all ways to remove that many units, the one for which the surrogate predicts the largest change, up or down. Every way is tried up to 20,000 of them; beyond that, one unit at a time.

In the formulas below, `f(S)` is the value of the prompt that keeps exactly the units of the set S, and `S + i` is S with unit i added.

### A worked example

The second case of [results.md](results.md) has two documents that hold the answer together: `d3` names a month, and `d2` ties the question to it. The six other documents change nothing, so four values tell everything:

| Kept, of `d2` and `d3` | Value |
|---|---|
| neither | −3.0486 |
| only `d2` | −3.0486 |
| only `d3` | −0.6931 |
| both | −0.0486 |

`d2` adds nothing alone, and +0.645 when `d3` is there. `d3` adds +2.355 alone, and +3.000 when `d2` is there. Each method turns these changes into its numbers in its own way:

- **`loo`** reads the change with all the other units kept: +0.645 for `d2` and +3.000 for `d3`. The joint effect is in both, so `loo` counts it twice.
- **`banzhaf`** takes the plain mean of a unit's changes: (0 + 0.645) / 2 = +0.322 for `d2`, and (2.355 + 3.000) / 2 = +2.678 for `d3`. Half of the joint effect goes to each.
- **`kernelshap`** takes a weighted mean, in which the combinations with very few or very many of the other units count most. Here it equals `banzhaf`, because the only joint effect is one of two units. Its scores add up to the full prompt's value minus the empty prompt's: 0.322 + 2.678 = 3.000.
- **`contextcite` and `lime`** estimate such an average from fewer masks, with a pull towards zero.
- **The interaction methods** give each unit its effect alone, 0 for `d2` and +2.355 for `d3`, and the pair a term of its own: 3.000 − 2.355 = +0.645.

### `loo`: leave one out

- **Masks.** The full prompt and the n prompts that each lack one unit: n + 1 masks, nothing random.
- **The score.** `w_i = f(full prompt) − f(full prompt without unit i)`. There is no fit. The base is `b = f(full prompt) − Σ_i w_i`, so the surrogate gives these n + 1 values back exactly.
- **What it means.** Each unit is tested only with all the others present. Two limits follow. When two units hold the same fact, removing either changes little, and both score near zero. A joint effect of several units goes to each of them in full (the worked example). For a mask that removes several units the surrogate adds up the single losses, which is why R² is often lower than for the other methods.

### `banzhaf`

- **Masks.** `budget` masks. In each, every unit is kept or removed by a fair coin flip, so each of the 2ⁿ masks is equally likely.
- **The fit.** Least squares, every mask counting the same, nothing added.
- **The score.** With every mask, the fit has a closed form, the Banzhaf value:

  ```
  w_i = (mean of f over the masks that keep unit i) − (mean of f over the masks that remove it)
      = (1 / 2^(n−1)) · Σ_S [ f(S + i) − f(S) ]         S: every combination of the other n − 1 units
  ```

- **What it means.** The change in the value when unit i is added, averaged over every combination of the other units, each combination counting the same. A joint effect of k units is shared: each of them gets 1/2^(k−1) of it, so half each for a pair. With fewer masks the score is the least-squares estimate of this average.

### `contextcite`

- **Masks.** `budget` coin-flip masks, as `banzhaf`. They are always drawn at random.
- **The fit.** A Lasso: least squares with a charge on the size of every score. With M masks it makes this as small as possible:

  ```
  (1 / 2M) · Σ_m ( y(m) − b − Σ_i w_i·m_i )²  +  alpha · Σ_i |w_i|          alpha = 0.01
  ```

  With `standardize=True` (the default), `y(m)` is `f(m)` minus its mean, divided by the spread (the standard deviation) of the values asked, and the scores are multiplied by that spread afterwards. So `alpha` means the same whether the values are of size 0.1 or 100. With `standardize=False`, `y(m) = f(m)`, as in the ContextCite paper.
- **What it means.** The charge pulls every score towards zero by a fixed amount, and a score that the values support by less than that amount becomes exactly 0. With many masks the amount is about 4 × `alpha` × the spread of the values: 0.04 of the spread at the default (derived; measured as 0.015 to 0.017 against a derived 0.015, on a test function with 4,096 masks). With `standardize=False` it is 4 × `alpha`, in the value's own units. The result is a `banzhaf` estimate in which the units that matter stand out from few masks. The price: every non-zero score is a little too small, and a weak unit can vanish.

### `lime`

- **Masks.** The first mask is the full prompt. Every other mask removes k units chosen at random, with k drawn evenly from 1 to n. So masks that remove few units are as common as masks that remove many, where coin flips would nearly always remove about half.
- **The weights.** A mask counts more, the closer it is to the full prompt:

  ```
  weight(m) = sqrt( exp( −d² / kernel_width² ) )       d = 100 · (1 − sqrt(units kept / n)),  kernel_width = 25
  ```

  `d` is 100 times the cosine distance between the mask and the full prompt. By the share of units kept, the weight is 1.00 for all of them, 0.87 for three quarters, 0.50 for half, 0.14 for a quarter and 0.0003 for none.
- **The fit.** A weighted ridge: least squares with those weights and a charge on the squared scores. It makes this as small as possible, with `alpha = 1`:

  ```
  Σ_m weight(m) · ( f(m) − b − Σ_i w_i·m_i )²  +  alpha · Σ_i w_i²
  ```

- **What it means.** A local description around the full prompt: what removing a unit does while most of the others are still there. The charge shrinks every score a little and sets none to exactly zero. It shrinks by the same share whatever the size of the values, and by less as the budget grows. This is what LIME's own explainer for text does, without its optional selection of features.

### `kernelshap`

- **Masks.** The empty and the full prompt, always. Then whole groups by size, while the budget covers them: every mask that keeps 1 unit and every mask that keeps n − 1, then those that keep 2 and n − 2, and so on. What is left of the budget is drawn at random from the sizes not covered, a size s with probability in proportion to (n − 1) / (s·(n − s)), and each drawn mask is used together with its opposite (`paired=True`).
- **The weights.** The Shapley kernel. A mask that keeps s units counts

  ```
  weight(s) = (n − 1) / ( C(n, s) · s · (n − s) )          C(n, s): the number of masks that keep s units
  ```

  For 20 units that is 0.05 for a mask that keeps one unit and 0.000001 for one that keeps ten. Every mask drawn at random gets the same weight: the total kernel weight of all the sizes not covered, divided by the number of masks drawn.
- **The fit.** Weighted least squares with two things held exactly: the base is the empty prompt's value, `b = f(empty)`, and the scores add up to `f(full) − f(empty)`.
- **The score.** With every mask (`budget=2**n`), the fit is the Shapley value:

  ```
  w_i = Σ_S  [ |S|! · (n − |S| − 1)! / n! ] · [ f(S + i) − f(S) ]        S: every combination of the other n − 1 units
  ```

- **What it means.** The change in the value when unit i is added, averaged over all n! orders in which the units can be added one at a time. Like `banzhaf` it averages `f(S + i) − f(S)`, but each size of S gets the same total weight, 1/n. The few combinations with almost none or almost all of the other units therefore count far more each than the many of middle size. A joint effect of k units is shared equally, 1/k each.

### `faithbanzhaf` and `faithshap`

Both fit the surrogate with interactions, with one term for every set of up to `max_order` units. At the default, 2, that is every unit and every pair:

```
f̂(m) = b + Σ_i w_i·m_i + Σ_{i<j} w_ij·m_i·m_j
```

| | `faithbanzhaf` | `faithshap` |
|---|---|---|
| Masks | the empty and the full prompt, then different masks drawn at random, each equally likely | the empty and the full prompt, then the masks that keep fewer than `max_order` units or remove fewer than that, then different masks drawn by size as `kernelshap` |
| A mask counts | the same as any other. With `ridge=False` on sampled masks, shapiq corrects each mask's weight for how it was drawn | by the Shapley kernel. The empty and the full prompt get a very large weight (10⁸) |
| Fit | every mask asked: least squares. Fewer: a ridge (`ridge=True`), or least squares (`ridge=False`) | weighted least squares |

- **What they are.** The Faith-Banzhaf and Faith-Shap interaction indices (Tsai et al., 2023): the surrogate of that order which is closest to `f` over all 2ⁿ masks. For `faithbanzhaf` every mask counts the same in "closest"; for `faithshap` the masks count by the Shapley kernel. With `max_order=1` they are `banzhaf` and `kernelshap`.
- **A unit's score** is now its own effect, and what it does only together with another unit is in the pair's term. When no effect in the model needs more than `max_order` units, a pair's term is the direct measurement, `w_ij = f(i and j) − f(i) − f(j) + f(none)`. An effect that needs more units than `max_order` is spread over the terms there are.
- **`faithshap` adds up.** The base is the empty prompt's value, and the base, the scores and the interactions together give the full prompt's value, both held by the large weight of the table above.
- **One number per unit again.** `e.shapley()` and `e.banzhaf()` fold the interactions of any result back into the units, by the sharing rules of `kernelshap` and `banzhaf` above: a set of k units gives each of them 1/k of its term, or 1/2^(k−1). `e.shapley_taylor(order)` keeps the terms of sets smaller than `order`, and shares the term of each larger set equally among its subsets of `order` units.
- **The ridge** (`faithbanzhaf` on sampled masks). With nearly as many terms as masks, least squares fits the noise. The ridge adds a charge on the squared terms (not on the base), and picks its strength among 100 values, evenly spaced on a log scale from 10⁻⁶ to 10⁶, by leaving out one mask at a time and predicting it. The fit is done with the units coded as +1 and −1, where the charge treats every term alike, and its result is rewritten in the keep-or-remove form above. The SPEX paper fits its Faith-Banzhaf baseline by cross-validated ridge regression too (its section B.3).

### `spex`

Fitting a term for every set of up to 5 units is impossible in a long prompt: 100 units have 79 million such sets. SPEX finds the few sets that matter without listing them (SPEX §4 and §5, Appendix A).

1. **Another way to write the value.** Any function of n keep-or-remove switches can be written exactly as a sum over sets of units (the Fourier transform, SPEX eq. 1):

   ```
   f(m) = Σ_S F(S) · (−1)^(number of units of S that the mask keeps)
   ```

   `F(S)` is the Fourier coefficient of the set S. SPEX rests on two properties that language models show in practice (SPEX §3): most `F(S)` are close to zero, and the large ones belong to small sets. `degree` (5 by default) is the largest set SPEX is built to find.
2. **Masks from a code, not from coin flips** (§5.1, Algorithm 1). Three random tables of 0s and 1s, each with b rows and n columns, turn the 2ᵇ numbers of b bits into 2ᵇ masks each. Every such group of masks is asked p + 1 times: once as it is, and once with each of p fixed patterns of units flipped. The patterns are the rows of the parity matrix of a BCH code, an error-correcting code that corrects `degree` errors; at the default `degree`, p is 20 for up to 11 units and 55 for up to 1,992. That makes 3 × 2ᵇ × (p + 1) masks, with b the largest that the budget allows. This is why SPEX takes its budgets from a table (section 3), and why they hardly grow with the prompt.
3. **Bins** (eq. 4 and 6). A Fourier transform of each group of 2ᵇ values gives 2ᵇ sums, the bins. Every `F(S)` falls into exactly one bin of each group, and a bin holds the sum of whatever falls into it. Since few `F(S)` are far from zero, many bins hold one large term or none.
4. **Naming the set in a bin** (§5.2, Appendix A.4). Across the p flipped copies, a bin that holds one large term changes sign in a pattern that is the code's fingerprint of S. Decoding that pattern as a received code word names S directly, with no search over sets, and it still works when the small terms in the same bin get some of the signs wrong. When the decoding fails, the decoder takes the `chase_depth` least certain signs and tries again with some of them flipped: the `chase_depth` × (`chase_depth` + 1) / 2 most likely patterns of flips (55 at the default of 10). It keeps the decoding that has to correct the fewest signs.
5. **Peeling** (Algorithm 4). A term's coefficient is read from the bin it was found in: the bin's values over the p + 1 copies, with the fingerprint's signs undone, averaged (and averaged over the bins, when several name the same term in one pass). The term is then subtracted from its bin in each of the three groups. That leaves some bins with one large term, which is decoded in turn, and so on until no bin gives a new term. A term found again adds to its coefficient. There is no fit over all the masks at the end: the coefficients are the ones read from the bins.
6. **Rewriting as scores** (Table 1). The terms found are rewritten, exactly, in the keep-or-remove form of the surrogate: `w_S = (−2)^|S| · Σ F(T)` over the found sets T that contain S. The predictions stay the same; the result reads as a score per unit and a `(together)` row per set.

- **What it means.** A unit's score and each set's term have the same meaning as for `faithbanzhaf`, but only for the sets SPEX found; every other term is zero. When the value really has few terms, SPEX recovers them exactly. When it has many small ones, they are left out, and R² shows how much that costs.

### `proxyspex`

1. **Masks.** `budget` different masks drawn at random, each equally likely, with the empty and the full prompt among them.
2. **A tree model learns the value.** Gradient-boosted regression trees (LightGBM) are trained to predict `f(m)` from the n keep-or-remove switches. With `hpo=True` eight settings are tried (trees of depth 3 or 5, 500 or 1,000 of them, learning rate 0.01 or 0.1), and the one that predicts held-out fifths of the masks best is kept.
3. **From trees to sets.** A tree asks about one unit at a time, so each of its leaves depends only on the few units on the way to it. The Fourier coefficients of the whole model (step 1 of `spex`) follow exactly from its trees, with no further call to the language model.
4. **Keep the large terms and fit them again.** When the trees hold more terms than the prompt has units, the largest coefficients, those that hold 95 % of the total of the squared coefficients (the base left out of the total), are kept and fitted again to the masks asked, by a ridge whose strength is picked as for `faithbanzhaf`.
5. **Rewriting as scores**, as step 6 of `spex`, for sets of up to `max_order` units. The term of a larger set, which only trees deeper than `max_order` can give, is shared out among its smaller sets.

- **What it means.** The scores describe the language model through the tree model, which learns from about a tenth of `spex`'s masks. Trees split first on the units with the largest effect, so a weak joint effect may never get a split of its own, and then shows up inside one unit's score.

## 3. How many masks: `budget`

More masks give steadier scores and cost more model calls. `ca.explain(..., budget=64)` is the default.

| Method | Budget | Notes |
|---|---|---|
| `loo` | not used | always n + 1 masks: the full prompt, and one mask per unit |
| `contextcite`, `lime` | 2 or more; 32 to 256 is usual | |
| `banzhaf` | n + 1 or more | it raises an error when the masks drawn cannot tell the units apart (fewer different masks than n + 1, or two units always kept and removed together): give a larger budget |
| `kernelshap` | 2n + 2 or more (or all 2ⁿ masks, when that is fewer) | 2n + 2 covers the empty and the full prompt and every mask that keeps 1 or n − 1 units |
| `faithbanzhaf`, `faithshap` | a few times the number of scores | with pairs (`max_order=2`) there are 1 + n + n(n − 1)/2 scores: 211 for 20 units, 5,051 for 100 |
| `spex` | from the table below | the default of 64 is refused, as is any budget too small for its `degree` |
| `proxyspex` | about a tenth of `spex`'s | about 500 for 20 units, 800 for 100 |

`budget=2**n` asks about every mask, which makes some methods exact (section 2). That is practical up to about 16 units.

Budgets from the SPEX paper's Table 2, its row for b = 6, by the number of units. Each is 3 × 2⁶ × (p + 1), as in step 2 of `spex` above. The paper's main results use b = 8, four times as many masks:

| Units, up to | 11 | 36 | 92 | 215 | 466 | 973 | 1,992 |
|---|---|---|---|---|---|---|---|
| Masks | 4,032 | 5,376 | 6,912 | 7,872 | 8,832 | 9,792 | 10,752 |

For `spex`, pass a budget from this table, such as `budget=4032` for up to 11 units. On a short prompt the same few masks come up again and again and the model is asked about each only once: 4,032 masks of a four-unit prompt cost 16 requests.

**`proxyspex` needs a prompt of 8 units or more.** Its tree model (LightGBM) splits a group of masks only when each side keeps at least 20 of them, and a prompt of n units has only 2ⁿ different masks: 32 for 5 units. So it refuses a prompt of fewer than 6 units, and a budget below 40 masks: it could learn nothing from them. Measured on functions with known parts: with 6 units it finds single units only; with 7, pairs too; from 8, sets of three as well. For a shorter prompt use `faithbanzhaf` or `faithshap` with `budget=2**n`, or `spex`.

[interface.md](interface.md) §7 works out what a budget costs in requests.

## 4. Settings of `ca.explain`

```python
e = ca.explain(prompt, model, target, method="contextcite", budget=64, test_masks=128, seed=0,
               replace=None, noise_repeats=3, removal=(), cache=None, batch_size=32)
```

| Setting | Default | What it means |
|---|---|---|
| `prompt` | | the prompt, a `ca.Prompt` with its units marked ([interface.md](interface.md) §2) |
| `model` | | your wrapper ([wrapper.md](wrapper.md)) |
| `target` | `ca.LogProb()` | which number is read from the model for each mask ([interface.md](interface.md) §5). The default explains the prompt's one answer |
| `method` | `"contextcite"` | a name from section 1, or a method object with its own settings (section 5) |
| `budget` | 64 | how many masks the method may use (section 3) |
| `test_masks` | 128 | how many test masks R² is measured on (section 2). Each costs as much as one of the method's masks, unless the method asked about the same mask. They are the same for every method run with the same `seed`, so with a `cache`, comparing methods on one prompt asks about them only once. 0 skips R² |
| `seed` | 0 | decides which masks are drawn. The same seed gives the same masks, and so the same result, on every run |
| `replace` | `None` | what happens to a removed unit: deleted, or replaced by a text ([interface.md](interface.md) §6) |
| `noise_repeats` | 3 | how often the full prompt is asked again to measure the noise (section 2). 0 skips it |
| `removal` | `()` | for example `(1, 2, 3)`: also remove the 1, 2 and 3 units whose removal should change the value most, and report how far it really moves (section 2). One more request for each |
| `cache` | `None` | a file path. Every answer of the model is kept there, so running again, or running another method on the same prompt, does not ask the model twice |
| `batch_size` | 32 | how many masked prompts your wrapper gets in one call |

Two other ways in:

- `ca.explain(prompt, value=f, name="...")`, with the same settings, explains any function `f` of masked prompts in place of a model and a target, for example a classifier. `f` gets a list of masked prompts and returns one number for each. `name` is how the result calls it, and what a `cache` stores its answers under.
- `ca.attribute(f, n, unit_ids=None, ...)`, with `method`, `budget`, `test_masks`, `seed` and `removal`, explains any function of masks, for inputs that are not text. `f` gets an array with one row per mask and `n` columns, `True` where a unit is kept, and returns one number per row. `unit_ids` names the columns.

## 5. Each method's own settings

`method="spex"` uses the defaults below. To change one, pass the method as an object: `method=ca.methods.SPEX(degree=3)`. The defaults are the most accurate settings measured. Section 2 says what each setting does in the computation.

| Object | Setting | Default | What it means |
|---|---|---|---|
| `LeaveOneOut()` | | | no settings |
| `Banzhaf()` | | | no settings |
| `ContextCite(alpha, standardize)` | `alpha` | 0.01 | the charge on the size of the scores. Larger: more scores are exactly zero |
| | `standardize` | `True` | divide the values by their spread before the fit. `False` fits the raw values, as the ContextCite paper does |
| `LIME(kernel_width, alpha)` | `kernel_width` | 25.0 | how quickly a mask counts less as it removes more units. Smaller: only masks close to the full prompt count |
| | `alpha` | 1.0 | the charge on the squared scores. 0: none |
| `KernelSHAP(paired)` | `paired` | `True` | each mask drawn at random is used together with its opposite, which gives steadier scores |
| `FaithBanzhaf(max_order, ridge)` | `max_order` | 2 | the largest set of units that gets a score of its own: 2 means single units and pairs |
| | `ridge` | `True` | sampled masks are fitted by a ridge. `False` is plain least squares, which goes wrong when there are nearly as many scores as masks: R² was −0.08 against 0.79 in a test with noise |
| `FaithShap(max_order)` | `max_order` | 2 | as for `FaithBanzhaf` |
| `SPEX(degree, chase_depth)` | `degree` | 5 | the largest set of units SPEX is built to find |
| | `chase_depth` | `None` | how many of the least certain signs its decoder may flip. `None` means 2 × `degree`, the setting of the SPEX authors' own code (`spectralexplain`). `chase_depth=degree` is the shapiq library's setting: 1 to 11 s faster per run, with R² 0.003 to 0.070 lower in 8 tests with noise |
| `ProxySPEX(max_order, hpo)` | `max_order` | 5 | the largest set of units that can get a score of its own |
| | `hpo` | `True` | choose the tree model's settings among eight. `False` uses LightGBM's defaults (100 trees), which is faster |

## 6. How ContextAttr is checked

- Each method is tested against exact answers: the closed forms of section 2, and values computed over every mask. You can run such a check on your own model: `contextattr demo` ([wrapper.md](wrapper.md) §5).
- On a real model (Llama-3.2-3B-Instruct, on one GPU) and a prompt of 20 documents, every method puts first the document the question is about.
- With its default settings, this SPEX gives the same predictions as the SPEX authors' own code on the same masks: R² 1.0000 between the two in 8 of 8 test functions with noise.
- Against the SPEX paper's published results, on a small sample (15 reviews, 10 questions, one prompt):
  - Sentiment: the methods come in the paper's order. R² is within 0.04 of the paper's on reviews of 8 to 15 words, and within 0.15 on longer ones.
  - HotpotQA: the methods come in the paper's order, but every R² is higher than the paper's (SPEX 0.95–0.98 against 0.60–0.87). The task was rebuilt from the paper's description, so those levels are not comparable.
  - The trolley prompt of the paper's section 7.1: the model is right more often without the last sentence, as in the paper, by a larger margin (from 21 % to 86 %, against 12 % to 21 %).
- Where it has been run: Linux, with Python 3.12, 3.13 and 3.14. ContextAttr itself runs on the CPU and uses no GPU. The model has been run on a GPU (Llama-3.2-3B-Instruct) and on a CPU (Qwen2.5-0.5B-Instruct), each through a wrapper. Not tried: macOS, Windows, Apple MPS, and a model behind a remote server.

## References

The sections, equations, tables and algorithms cited as "SPEX" above are those of the SPEX paper.

| Method | Reference |
|---|---|
| `loo` | Leave-one-out is the plain removal test. For language models: J. Li, W. Monroe and D. Jurafsky. Understanding Neural Networks through Representation Erasure. arXiv:1612.08220, 2016. The ContextCite paper (below) uses it as a baseline |
| `banzhaf` | J. F. Banzhaf III. Weighted Voting Doesn't Work: A Mathematical Analysis. Rutgers Law Review 19(2):317–343, 1965. As a score in machine learning: J. T. Wang and R. Jia. Data Banzhaf: A Robust Data Valuation Framework for Machine Learning. AISTATS 2023, PMLR 206:6388–6421 |
| `contextcite` | B. Cohen-Wang, H. Shah, K. Georgiev and A. Mądry. ContextCite: Attributing Model Generation to Context. NeurIPS 2024. arXiv:2409.00729. Its fit, the Lasso: R. Tibshirani. Regression Shrinkage and Selection via the Lasso. Journal of the Royal Statistical Society B 58(1):267–288, 1996 |
| `lime` | M. T. Ribeiro, S. Singh and C. Guestrin. "Why Should I Trust You?": Explaining the Predictions of Any Classifier. KDD 2016, pp. 1135–1144 |
| `kernelshap` | S. M. Lundberg and S.-I. Lee. A Unified Approach to Interpreting Model Predictions. NeurIPS 2017, pp. 4768–4777. The Shapley value: L. S. Shapley. A Value for n-Person Games. In Contributions to the Theory of Games II, Annals of Mathematics Studies 28, pp. 307–317. Princeton University Press, 1953 |
| `faithbanzhaf`, `faithshap` | C.-P. Tsai, C.-K. Yeh and P. Ravikumar. Faith-Shap: The Faithful Shapley Interaction Index. Journal of Machine Learning Research 24(94):1–42, 2023 |
| `spex` | J. S. Kang, L. Butler, A. Agarwal, Y. E. Erginbas, R. Pedarsani, B. Yu and K. Ramchandran. SPEX: Scaling Feature Interaction Explanations for LLMs. ICML 2025, PMLR 267. https://proceedings.mlr.press/v267/kang25a.html |
| `proxyspex` | L. Butler, A. Agarwal, J. S. Kang, Y. E. Erginbas, B. Yu and K. Ramchandran. ProxySPEX: Inference-Efficient Interpretability via Sparse Feature Interactions in LLMs. NeurIPS 2025. arXiv:2505.17495 |

Also used:

- `e.shapley_taylor`: M. Sundararajan, K. Dhamdhere and A. Agarwal. The Shapley Taylor Interaction Index. ICML 2020, PMLR 119:9259–9268.
- The four interaction methods run through the shapiq library, and SPEX's decoding through its sparse-transform library: M. Muschalik, H. Baniecki, F. Fumagalli, P. Kolpaczki, B. Hammer and E. Hüllermeier. shapiq: Shapley Interactions for Machine Learning. NeurIPS 2024, Datasets and Benchmarks Track.
