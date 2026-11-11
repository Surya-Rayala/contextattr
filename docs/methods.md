# Methods: which to use, their settings, and reading a result

A method decides which copies of the prompt to ask the model about, each with some units removed (a *mask*), and fits a small model of the answers, the *surrogate*. The surrogate's terms are the scores. `budget` is the number of masks, and so of model calls, per prompt. Every result is then checked on `test_masks` other masks: R² near 1 means the surrogate predicts the model well, and R² near 0 or below means its scores should not be trusted (SPEX eq. 7).

## Which method

| `method=` | Gives | Masks it needs | Use it for | Install |
|---|---|---|---|---|
| `loo` | each unit's effect when it alone is removed | n + 1 | a first look at a short prompt | core |
| `contextcite` (the default) | one score per unit, most of them zero: ContextCite's sparse linear fit | 32–256 | long prompts cut into sentences or documents | core |
| `banzhaf` | Banzhaf values | 64 or more | with `lime`, the most faithful one-score-per-unit method in SPEX's tests (B.5.1) | core |
| `lime` | LIME's locally weighted linear fit | 64 or more | as `banzhaf` | core |
| `kernelshap` | Shapley values | 64 or more; exact from 2ⁿ | the most used score, though less faithful as prompts grow (B.5.1) | core |
| `faithbanzhaf`, `faithshap` | every interaction up to an order, 2 unless set | a few times the number of terms, which grows as nᵈ | prompts below about 64 units | `[interactions]` |
| `spex` | the few interactions that matter, up to order 5 unless set | thousands: see below | prompts of tens to about 2,000 units | `[interactions]` |
| `proxyspex` | the same, read from a tree model fitted to the answers | about a tenth of SPEX's | models whose every call is costly | `[interactions]` |

SPEX's usual budgets are those of its Table 2 (b = 6): 4,032 masks for up to 11 units, 5,376 up to 36, 6,912 up to 92, 7,872 up to 215, 8,832 up to 466, 9,792 up to 973 and 10,752 up to 1,992. A budget too small for its degree is refused, not run at a lower degree.

Screen many prompts with `contextcite` at 64 masks, or `loo` for short prompts. Then run `spex` or `proxyspex` on the prompts worth a closer look. [interface.md](interface.md) §7 works out the costs.

## Settings

`ca.explain(prompt, model, target, method="contextcite", budget=64, ...)` takes:

| Setting | Default | What it does |
|---|---|---|
| `budget` | 64 | masks per prompt for the method |
| `test_masks` | 128 | held-out masks for R², the same for every method run with the same seed (SPEX used 10,000 per example) |
| `seed` | 0 | which masks are drawn: the same seed, the same masks |
| `replace` | none: delete | a removed unit is deleted, or replaced by this text (`"[UNK]"`, as SPEX does) |
| `noise_repeats` | 3 | how often the full prompt is asked again to measure the noise: scores smaller than it mean nothing |
| `removal` | none | for example `(1, 2, 3)`: remove the top 1, 2 and 3 units and report how far the value moves (SPEX eq. 8) |
| `cache` | none | a file that keeps every model answer, so a rerun, or another method on the same masks, costs nothing |
| `batch_size` | 32 | masked prompts per call of the wrapper |

A method's own settings go through its object, `method=ca.methods.<Name>(...)`. The defaults are the most accurate settings measured:

| Object | Default | The other choice |
|---|---|---|
| `SPEX(degree=5, chase_depth=None)` | a Chase depth of 2 × degree, the SPEX authors' own: this SPEX then gives what their code (`spectralexplain`) gives on the same masks | `chase_depth=degree`, shapiq's own: R² 0.003–0.070 lower with noise in 8 test functions, and 1–11 s faster per fit |
| `FaithBanzhaf(max_order=2, ridge=True)` | fitted by cross-validated ridge regression, as SPEX does (B.3); by least squares when every mask was asked, which is then exact | `ridge=False`, least squares always: it overfits as the terms near the masks (R² −0.08 against ridge's 0.79 in a test with noise) |
| `FaithShap(max_order=2)` | shapiq's weighted least squares | |
| `ProxySPEX(max_order=5, hpo=True)` | the tree model's settings chosen by cross-validation, shapiq's default | `hpo=False`: LightGBM's own settings, faster |
| `ContextCite(alpha=0.01, standardize=True)` | a Lasso fit on the values divided by their spread | |
| `LIME(kernel_width=25.0, alpha=1.0)` | the `lime` library's masks and kernel, with a weighted ridge fit | |
| `KernelSHAP(paired=True)` | masks drawn in complementary pairs | |

## Reading a result

`ca.explain` and `ca.attribute` return an `Explanation`:

| Use | Gives |
|---|---|
| `print(e)`, `e.table()` | a header (method, masks, R², noise, requests), then every unit with its score, highest first, then the strongest interactions |
| `e.scores` | `{unit id: score}`. A positive score pushes the value up: towards the answer explained, or with `Contrast`, towards the unwanted answers |
| `e.top(k)` | the k units that push the value up most |
| `e.interactions`, `e.top_interactions(k)` | `{(unit ids): score}`: what units do together beyond their own scores (interaction methods only) |
| `e.r2` | the faithfulness: how well the scores predict the model on held-out masks (1 is perfect) |
| `e.noise` | how much the value moves when the same prompt is asked again: scores below its `std` mean nothing |
| `e.shapley()`, `e.banzhaf()`, `e.shapley_taylor(order)` | the usual indices, read from any method's result (SPEX App. C) |
| `e.to_html(path)`, `e.as_dict()` | a page with each unit shaded by its score; a dictionary to save as JSON |

To confirm a result on the behaviour itself, remove the top units and sample the model again: `ca.intervene(prompt, model, check, [u for u, _ in e.top(3)], samples=64)` gives the share of flagged replies before and after.

## How ContextAttr is checked

- Each method is tested against exact answers: closed forms, and values computed over every mask.
- On a real model (Llama-3.2-3B-Instruct, on one GPU), `check_model` passes every check. On a prompt of 20 documents, every method puts first the document the question is about, with R² 0.81–0.99 on held-out masks.
- With its default settings, this SPEX gives the same predictions as the SPEX authors' own code (`spectralexplain`) on the same masks: R² 1.0000 between the two in 8 of 8 noisy test functions.
- Against SPEX's published results, on a small sample (15 reviews, 10 questions, one prompt):
  - Sentiment: R² is within 0.04 of the paper's on reviews of 8–15 words and within 0.15 on longer ones, with the methods in the paper's order.
  - HotpotQA: the methods are in the paper's order, but every R² is higher (SPEX 0.95–0.98 against 0.60–0.87). The task here follows the paper's description, which the authors' released code does not, so the levels are not comparable.
  - The trolley prompt: the model is right more often without the last sentence, as in the paper, by a larger margin (21 % to 86 %, against 12 % to 21 %).

## References

- SPEX: J. S. Kang, L. Butler, A. Agarwal, Y. E. Erginbas, R. Pedarsani, B. Yu and K. Ramchandran. SPEX: Scaling Feature Interaction Explanations for LLMs. ICML 2025, PMLR 267. https://proceedings.mlr.press/v267/kang25a.html
- LIME (Ribeiro et al., 2016), KernelSHAP (Lundberg and Lee, 2017), Faith-Banzhaf and Faith-Shap (Tsai et al., 2023), ContextCite (Cohen-Wang et al., 2024) and ProxySPEX (Butler et al., 2025). The interaction methods run through the shapiq library.
