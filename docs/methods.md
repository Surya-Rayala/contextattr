# Methods and settings

Which method to use, how each one works, how many masks to give it, and what every setting means. [results.md](results.md) shows the output of each method on one example and how to read it.

Words used here (`ca` is the package: `import contextattr as ca`):

| Word | Meaning |
|---|---|
| unit | a part of the prompt that may be removed. `n` is the number of units in a prompt |
| mask | one choice of units to keep. The model is asked about the prompt with only those units |
| value | the number read from the model for one mask. The *target* decides which number ([interface.md](interface.md) §5) |
| method | decides which masks to ask about, and works out the scores from their values |
| budget | the number of masks a method may use for one prompt. Each mask costs model calls, so the budget is the cost |
| R² | how well the scores predict the value on masks the method did not use: 1 is perfect, 0 or below means the scores should not be trusted ([results.md](results.md) §6 says what to do then) |

## 1. Which method, and how each works

Every method asks the model about a set of masks. It then looks for the numbers, the scores, that best reproduce the values it got when the scores of the units kept are added up. The methods differ in which masks they ask about, which masks count more, and whether small scores are set to zero.

| `method=` | How it works | Use it for |
|---|---|---|
| `"loo"` | asks about the full prompt, then about the prompt with one unit removed, for each unit in turn. A unit's score is how far the value falls without it | a first look at a short prompt |
| `"contextcite"` (the default) | keeps or removes each unit by coin flip in every mask, and sets small scores to exactly zero, so the few units that matter stand out | long prompts cut into sentences or documents |
| `"banzhaf"` | the same masks, with no score set to zero. A unit's score is its average effect over every combination of the other units | when every unit should get a score |
| `"lime"` | masks that remove from one unit to all of them. Masks near the full prompt count more, and all scores are shrunk a little | as `banzhaf` |
| `"kernelshap"` | masks that keep very few or very many units count most, and the scores are made to add up to the full prompt's value minus the empty prompt's. These are Shapley values, the most widely used score | short prompts |
| `"faithbanzhaf"` | as `banzhaf`, with one more number for every pair of units: what the two do together, beyond their own scores | prompts below about 64 units whose units may matter only together |
| `"faithshap"` | as `faithbanzhaf`, with the masks weighted as in `kernelshap` | the same |
| `"spex"` | asks about masks in a pattern taken from error-correcting codes. That lets it work out which few units and sets of units matter without trying every set | prompts of tens to about 2,000 units |
| `"proxyspex"` | trains a tree model on the values of random masks, and reads the units and sets that matter off its trees | prompts of 8 units or more, on models whose every call is costly |

- The first five come with `pip install contextattr`. The last four need `pip install "contextattr[interactions]"`.
- If unsure: `contextcite` for a first look, `banzhaf` when every unit should get a score, and `spex` (or `faithbanzhaf` on a short prompt) for units that matter only together.
- In the SPEX paper's tests (its section B.5.1), `banzhaf` and `lime` fit the model best among the methods with one score per unit, and `kernelshap` fits less well as prompts grow.

## 2. How many masks: `budget`

More masks give steadier scores and cost more model calls. `ca.explain(..., budget=64)` is the default.

| Method | Budget | Notes |
|---|---|---|
| `loo` | not used | always n + 1 masks: the full prompt, and one mask per unit |
| `contextcite`, `lime` | 2 or more; 32 to 256 is usual | |
| `banzhaf` | n + 1 or more | exact when the budget covers all 2ⁿ masks (up to 16 units). It raises an error when the masks drawn cannot tell the units apart: give a larger budget |
| `kernelshap` | 2n + 2 or more (or all 2ⁿ masks, when that is fewer) | exact when the budget covers all 2ⁿ masks |
| `faithbanzhaf`, `faithshap` | a few times the number of scores | with pairs (`max_order=2`) there are 1 + n + n(n − 1)/2 scores: 211 for 20 units, 5,051 for 100. Exact with `budget=2**n`, every mask |
| `spex` | from the table below | the default of 64 is refused, as is any budget too small for its `degree` |
| `proxyspex` | about a tenth of `spex`'s | about 500 for 20 units, 800 for 100 |

The budgets the SPEX paper uses (its Table 2), by the number of units:

| Units, up to | 11 | 36 | 92 | 215 | 466 | 973 | 1,992 |
|---|---|---|---|---|---|---|---|
| Masks | 4,032 | 5,376 | 6,912 | 7,872 | 8,832 | 9,792 | 10,752 |

For `spex`, pass a budget from this table, such as `budget=4032` for up to 11 units. On a short prompt the same few masks come up again and again and the model is asked about each only once: 4,032 masks of a four-unit prompt cost 16 requests.

**`proxyspex` needs a prompt of 8 units or more.** Its tree model (LightGBM) splits a group of masks only when each side keeps at least 20 of them, and a prompt of n units has only 2ⁿ different masks: 32 for 5 units. So it refuses a prompt of fewer than 6 units, and a budget below 40 masks: it could learn nothing from them. Measured on functions with known parts: with 6 units it finds single units only; with 7, pairs too; from 8, sets of three as well. For a shorter prompt use `faithbanzhaf` or `faithshap` with `budget=2**n`, or `spex`.

[interface.md](interface.md) §7 works out what a budget costs in requests.

## 3. Settings of `ca.explain`

```python
e = ca.explain(prompt, model, target, method="contextcite", budget=64, test_masks=128, seed=0,
               replace=None, noise_repeats=3, removal=(), cache=None, batch_size=32)
```

| Setting | Default | What it means |
|---|---|---|
| `prompt` | | the prompt, a `ca.Prompt` with its units marked ([interface.md](interface.md) §2) |
| `model` | | your wrapper ([wrapper.md](wrapper.md)) |
| `target` | `ca.LogProb()` | which number is read from the model for each mask ([interface.md](interface.md) §5). The default explains the prompt's one answer |
| `method` | `"contextcite"` | a name from section 1, or a method object with its own settings (section 4) |
| `budget` | 64 | how many masks the method may use (section 2) |
| `test_masks` | 128 | how many other masks R² is measured on. Each costs as much as one of the method's masks. 0 skips R² |
| `seed` | 0 | decides which masks are drawn. The same seed gives the same masks, and so the same result, on every run |
| `replace` | `None` | what happens to a removed unit. `None` deletes it. A text, such as `"[UNK]"`, is put in its place (the SPEX paper does this) |
| `noise_repeats` | 3 | how often the full prompt is asked again to measure the noise. Scores smaller than the noise mean nothing. 0 skips it |
| `removal` | `()` | for example `(1, 2, 3)`: also remove the 1, 2 and 3 units whose removal should change the value most, up or down, and report how far it really moves ([results.md](results.md) §6). One more request for each |
| `cache` | `None` | a file path. Every answer of the model is kept there, so running again, or running another method on the same prompt, does not ask the model twice |
| `batch_size` | 32 | how many masked prompts your wrapper gets in one call |

Two other ways in take the same settings:

- `ca.explain(prompt, value=f, name="...")` explains any function `f` of masked prompts in place of a model and a target, for example a classifier. `f` gets a list of masked prompts and returns one number for each. `name` is how the result calls it, and what a `cache` stores its answers under.
- `ca.attribute(f, n, unit_ids=None, ...)` explains any function of masks, for inputs that are not text. `f` gets an array with one row per mask and `n` columns, `True` where a unit is kept, and returns one number per row. `unit_ids` names the columns.

## 4. Each method's own settings

`method="spex"` uses the defaults below. To change one, pass the method as an object: `method=ca.methods.SPEX(degree=3)`. The defaults are the most accurate settings measured.

| Object | Setting | Default | What it means |
|---|---|---|---|
| `LeaveOneOut()` | | | no settings |
| `Banzhaf()` | | | no settings |
| `ContextCite(alpha, standardize)` | `alpha` | 0.01 | how strongly scores are pulled to zero. Larger: more scores are exactly zero |
| | `standardize` | `True` | makes `alpha` mean the same for every target, by dividing the values by their spread first. `False` uses the raw values, as the ContextCite paper does |
| `LIME(kernel_width, alpha)` | `kernel_width` | 25.0 | how quickly a mask counts less as it removes more units. Smaller: only masks close to the full prompt count |
| | `alpha` | 1.0 | how strongly scores are shrunk towards zero. 0: not at all |
| `KernelSHAP(paired)` | `paired` | `True` | each sampled mask is used together with its opposite, which gives steadier scores |
| `FaithBanzhaf(max_order, ridge)` | `max_order` | 2 | the largest set of units that gets a score of its own: 2 means single units and pairs |
| | `ridge` | `True` | keeps the scores steady when there are nearly as many scores as masks, as the SPEX paper does (B.3). Without it, R² was −0.08 against 0.79 in a test with noise |
| `FaithShap(max_order)` | `max_order` | 2 | as for `FaithBanzhaf` |
| `SPEX(degree, chase_depth)` | `degree` | 5 | the largest set of units that can get a score of its own |
| | `chase_depth` | `None` | how hard SPEX searches for sets of units. `None` means 2 × `degree`, the setting of the SPEX authors' own code. `chase_depth=degree` is 1 to 11 s faster per run, with R² 0.003 to 0.070 lower in 8 tests with noise |
| `ProxySPEX(max_order, hpo)` | `max_order` | 5 | the largest set of units that can get a score of its own |
| | `hpo` | `True` | tries several settings of the tree model and keeps the best. `False` uses LightGBM's defaults, which is faster |

## 5. How ContextAttr is checked

- Each method is tested against exact answers: formulas whose result is known, and values computed over every mask. You can run such a check on your own model: `contextattr demo` ([wrapper.md](wrapper.md) §10).
- On a real model (Llama-3.2-3B-Instruct, on one GPU) and a prompt of 20 documents, every method puts first the document the question is about.
- With its default settings, this SPEX gives the same predictions as the SPEX authors' own code on the same masks (8 of 8 test functions with noise).
- Against the SPEX paper's published results, on a small sample: the methods come in the paper's order, and R² is within 0.04 of the paper's on short reviews. On HotpotQA every R² is higher than the paper's (SPEX 0.95–0.98 against 0.60–0.87): the task was rebuilt from the paper's description, so those levels are not comparable.
- Where it has been run: Linux, with Python 3.12, 3.13 and 3.14. ContextAttr itself runs on the CPU and uses no GPU. The model has been run on a GPU (Llama-3.2-3B-Instruct) and on a CPU (Qwen2.5-0.5B-Instruct), each through a wrapper. Not tried: macOS, Windows, Apple MPS, and a real API service.

## References

- SPEX: J. S. Kang, L. Butler, A. Agarwal, Y. E. Erginbas, R. Pedarsani, B. Yu and K. Ramchandran. SPEX: Scaling Feature Interaction Explanations for LLMs. ICML 2025, PMLR 267. https://proceedings.mlr.press/v267/kang25a.html. "B.3", "B.5.1" and "Table 2" above are its sections and tables.
- LIME (Ribeiro et al., 2016), KernelSHAP (Lundberg and Lee, 2017), Faith-Banzhaf and Faith-Shap (Tsai et al., 2023), ContextCite (Cohen-Wang et al., 2024) and ProxySPEX (Butler et al., 2025). The interaction methods run through the shapiq library.
