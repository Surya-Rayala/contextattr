# methods/: which masks to evaluate, and which fit ([docs/methods.md](../../../docs/methods.md))

A method designs the masks (`masks(n, budget, rng)`) and fits a surrogate to their values (`fit(masks, values)`); the surrogate's terms are the scores, and its predictions on held-out masks give the faithfulness R² (`evaluate.py`).

| File | Holds |
|---|---|
| `base.py` | `Method`, `Surrogate` (f̂(m) = intercept + Σ scores·m + Σ interactions·Π m), `all_masks`, least squares |
| `loo.py` | `LeaveOneOut`: the drop in the value when one unit alone is removed; n + 1 masks |
| `linear.py` | `Banzhaf` (least squares on uniform masks; exact from 2^n masks), `ContextCite` (Lasso, alpha 0.01, on the values divided by their spread unless `standardize=False`), `LIME` (its text explainer's masks and kernel, weighted ridge) |
| `kernelshap.py` | `KernelSHAP`: Shapley values; the heaviest size groups enumerated, the rest drawn in complementary pairs; exact from 2^n masks, and for games without interactions above pairs |
| `interactions.py` | through shapiq, imported only when one runs: `FaithBanzhaf` (by default refitted by cross-validated ridge, SPEX B.3, or shapiq's least squares when every mask was asked; `ridge=False`: least squares), `FaithShap` (every interaction up to an order; small n), `SPEX` (sparse, any order up to its degree; Chase depth 2t unless set, as the SPEX authors' code), `ProxySPEX` (sparse, LightGBM proxy); each answer read as product-basis (FBII/FSII) terms |
| `indices.py` | a surrogate read as Shapley values, Banzhaf values and the Shapley-Taylor index (App. C) |
| `__init__.py` | the methods by name: `loo`, `banzhaf`, `lime`, `contextcite`, `kernelshap`, `faithbanzhaf`, `faithshap`, `spex`, `proxyspex` |

A method refuses masks or a budget that would not give the fit it names: Banzhaf and KernelSHAP masks that leave the scores undetermined, SPEX a budget for which shapiq would lower its degree.

Importing shapiq takes 25–45 s and galois's first codes about 15 s more on a shared two-CPU machine, so shapiq is imported only when one of these methods runs, and the tests that need it are marked `shapiq` (5–7 minutes; `CA_TEST_SHAPIQ=1`).
