# Methods and settings

Which method to use, how many masks to give it, and what every setting means. [results.md](results.md) shows the output of each method on one example and how to read it.

Words used here (`ca` is the package: `import contextattr as ca`):

| Word | Meaning |
|---|---|
| unit | a part of the prompt that may be removed. `n` is the number of units in a prompt |
| mask | one choice of units to keep. The model is asked about the prompt with only those units |
| value | the number read from the model for one mask. The *target* decides which number ([interface.md](interface.md) §5) |
| method | decides which masks to ask about, and works out the scores from their values |
| budget | the number of masks a method may use for one prompt. Each mask costs model calls, so the budget is the cost |
| R² | how well the scores predict the value on masks the method did not use: 1 is perfect, 0 or below means the scores should not be trusted |

## 1. Which method

| `method=` | Gives | Masks it needs | Use it for | Install |
|---|---|---|---|---|
| `"loo"` | each unit's effect when it alone is removed | n + 1, always | a first look at a short prompt | core |
| `"contextcite"` (the default) | one score per unit, most of them exactly zero | 32 to 256 | long prompts cut into sentences or documents | core |
| `"banzhaf"` | one score per unit: its average effect | 64 or more | with `lime`, the most faithful one-score-per-unit method in SPEX's tests (B.5.1) | core |
| `"lime"` | one score per unit, weighted towards masks near the full prompt | 64 or more | as `banzhaf` | core |
| `"kernelshap"` | one score per unit: its Shapley value | 64 or more | the most widely used score, though less faithful as prompts grow (B.5.1) | core |
| `"faithbanzhaf"`, `"faithshap"` | a score per unit and per pair of units | a few times the number of pairs: section 2 | prompts below about 64 units | `[interactions]` |
| `"spex"` | a score per unit, and the few sets of units that matter together | thousands: section 2 | prompts of tens to about 2,000 units | `[interactions]` |
| `"proxyspex"` | the same, estimated by a tree model | about a tenth of `spex`'s | prompts of 8 units or more (section 2 says why) on models whose every call is costly | `[interactions]` |

"Core" methods come with `pip install contextattr`. The others need `pip install "contextattr[interactions]"`.

A way to work: screen many prompts with `contextcite` at 64 masks, or `loo` for short prompts. Then run `spex` or `proxyspex` on the prompts worth a closer look.

## 2. How many masks: `budget`

More masks give steadier scores and cost more model calls. `ca.explain(..., budget=64)` is the default.

| Method | Budget | Notes |
|---|---|---|
| `loo` | not used | always n + 1 masks: the full prompt, and one mask per unit |
| `contextcite`, `lime` | 2 or more; 32 to 256 is usual | |
| `banzhaf` | n + 1 or more | exact when the budget covers all 2ⁿ masks (up to 16 units). Refuses masks that leave a score undetermined |
| `kernelshap` | 2n + 2 or more (or all 2ⁿ masks, when that is fewer) | exact when the budget covers all 2ⁿ masks |
| `faithbanzhaf`, `faithshap` | a few times the number of scores | with pairs (`max_order=2`) there are 1 + n + n(n − 1)/2 scores: 211 for 20 units, 5,051 for 100 |
| `spex` | see below | a budget too small for its `degree` is refused with a message, not run at a lower degree |
| `proxyspex` | about a tenth of `spex`'s | about 500 for 20 units, 800 for 100. Not for prompts of fewer than 8 units: see below |

The budgets SPEX's authors use (their Table 2, with their setting b = 6), by the number of units:

| Units, up to | 11 | 36 | 92 | 215 | 466 | 973 | 1,992 |
|---|---|---|---|---|---|---|---|
| Masks | 4,032 | 5,376 | 6,912 | 7,872 | 8,832 | 9,792 | 10,752 |

**`proxyspex` needs a prompt of 8 units or more.** It learns a tree model (LightGBM) from the values of the masks and reads the scores off its trees. LightGBM splits a group of masks only if each side keeps at least 20 of them, and a prompt of n units has only 2ⁿ different masks: 32 for 5 units. So a short prompt gives it too little to learn from. Measured on functions with known parts, with every mask asked:

| Units | Different masks | What `proxyspex` finds |
|---|---|---|
| 3, 4, 5 | 8, 16, 32 | nothing: every score is zero, and R² is about 0 |
| 6 | 64 | single units only (R² 0.66 to 0.80): what two units do together is missed or added to their own scores |
| 7 | 128 | single units and pairs (R² 0.94 to 1.00) |
| 8 or more | 256 or more | single units, pairs and sets of three: the functions exactly (R² 1.00) |

- A failed fit gives no error: all scores are zero and R² is near 0. Read R² first ([results.md](results.md) §6).
- For a short prompt use `spex`, `faithbanzhaf` or `faithshap`. Given every mask (`budget=2**n`, where n is the number of units), they are exact.
- On a long prompt the same rule applies to the budget: a set of k units that matter only together needs about 20 × 2ᵏ masks, which is 160 for sets of three and 320 for sets of four (worked out from the rule above, not measured beyond sets of three).

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
| `test_masks` | 128 | how many other masks R² is measured on. Each costs as much as a method's mask. They are the same for every method run with the same `seed`, so comparing methods on one prompt asks about them only once when a `cache` is given. 0 skips R² |
| `seed` | 0 | decides which masks are drawn. The same seed gives the same masks, and so the same result, on every run |
| `replace` | `None` | what happens to a removed unit. `None` deletes it. A text, such as `"[UNK]"`, is put in its place (SPEX does this) |
| `noise_repeats` | 3 | how often the full prompt is asked again to measure the noise. Scores smaller than the noise mean nothing. 0 skips it |
| `removal` | `()` | for example `(1, 2, 3)`: also remove the top 1, 2 and 3 units and report how far the value really moves ([results.md](results.md) §6). One more request for each |
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
| | `standardize` | `True` | the values are divided by their spread before the fit, so that `alpha` means the same for every target. `False` fits the raw values, as the ContextCite paper does |
| `LIME(kernel_width, alpha)` | `kernel_width` | 25.0 | how quickly a mask counts less as it removes more units. Smaller: only masks close to the full prompt count |
| | `alpha` | 1.0 | how strongly scores are shrunk towards zero. 0: not at all |
| `KernelSHAP(paired)` | `paired` | `True` | each sampled mask is used together with its opposite, which gives steadier scores |
| `FaithBanzhaf(max_order, ridge)` | `max_order` | 2 | the largest set of units that gets a score of its own: 2 means single units and pairs |
| | `ridge` | `True` | the fit is steadied against noise, as SPEX does (B.3). `False` is plain least squares, which goes wrong when there are nearly as many scores as masks (R² −0.08 against 0.79 in a test with noise). When every one of the 2ⁿ masks was asked, the fit is exact either way |
| `FaithShap(max_order)` | `max_order` | 2 | as for `FaithBanzhaf` |
| `SPEX(degree, chase_depth)` | `degree` | 5 | the largest set of units that can get a score of its own |
| | `chase_depth` | `None` | how hard its decoder searches. `None` means 2 × `degree`, the SPEX authors' own setting: this SPEX then gives what their code (`spectralexplain`) gives on the same masks. `chase_depth=degree` is the shapiq library's setting: 1 to 11 s faster per fit, with R² 0.003 to 0.070 lower in 8 tests with noise |
| `ProxySPEX(max_order, hpo)` | `max_order` | 5 | the largest set of units that can get a score of its own |
| | `hpo` | `True` | the tree model's own settings are chosen by cross-validation. `False` uses LightGBM's defaults, which is faster |

## 5. How ContextAttr is checked

- Each method is tested against exact answers: closed forms, and values computed over every mask.
- On a real model (Llama-3.2-3B-Instruct, on one GPU), `check_model` passes every check. On a prompt of 20 documents, every method puts first the document the question is about, with R² 0.81–0.99 on held-out masks.
- Where it has been run: on Linux with Python 3.12, 3.13 and 3.14. ContextAttr itself is NumPy and scikit-learn on the CPU and uses no GPU. The model was run on a GPU (Llama-3.2-3B-Instruct, an H100) and on a CPU (Qwen2.5-0.5B-Instruct), each through a wrapper. Not tried: macOS, Windows, Apple MPS, and a real API service (the guide's API wrapper was run against a stand-in for the service).
- With its default settings, this SPEX gives the same predictions as the SPEX authors' own code (`spectralexplain`) on the same masks: R² 1.0000 between the two in 8 of 8 noisy test functions.
- Against SPEX's published results, on a small sample (15 reviews, 10 questions, one prompt):
  - Sentiment: R² is within 0.04 of the paper's on reviews of 8–15 words and within 0.15 on longer ones, with the methods in the paper's order.
  - HotpotQA: the methods are in the paper's order, but every R² is higher (SPEX 0.95–0.98 against 0.60–0.87). The task here follows the paper's description, which the authors' released code does not, so the levels are not comparable.
  - The trolley prompt: the model is right more often without the last sentence, as in the paper, by a larger margin (21 % to 86 %, against 12 % to 21 %).

## References

- SPEX: J. S. Kang, L. Butler, A. Agarwal, Y. E. Erginbas, R. Pedarsani, B. Yu and K. Ramchandran. SPEX: Scaling Feature Interaction Explanations for LLMs. ICML 2025, PMLR 267. https://proceedings.mlr.press/v267/kang25a.html. "B.3", "B.5.1" and "Table 2" above are its sections and tables.
- LIME (Ribeiro et al., 2016), KernelSHAP (Lundberg and Lee, 2017), Faith-Banzhaf and Faith-Shap (Tsai et al., 2023), ContextCite (Cohen-Wang et al., 2024) and ProxySPEX (Butler et al., 2025). The interaction methods run through the shapiq library.
