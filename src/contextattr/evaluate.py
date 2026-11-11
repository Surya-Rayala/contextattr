"""Checks of an explanation (docs/methods.md; SPEX §6.1): how well its surrogate predicts the value on held-out
masks (faithfulness R², eq. 7), how much the value changes when the units the surrogate ranks highest are removed
(top-r removal, eq. 8), how well its largest terms fall on human-labelled units (Recovery@r, eq. 9), and what removing
units does to the behaviour itself, sampled afresh (intervention, as SPEX B.5.2 did on the trolley prompt)."""

from __future__ import annotations

import itertools
from math import comb

import numpy as np

from .methods.base import Surrogate
from .methods.indices import terms as surrogate_terms

TEST_STREAM = 7919   # the test masks' own random stream, apart from every method's
EXACT_REMOVAL = 20_000


def held_out_masks(n: int, count: int, seed: int = 0) -> np.ndarray:
    """`count` masks keeping each unit with probability ½, from a stream of their own: the same for every method run
    on a prompt with this seed, so their values are asked of the model once and then come from the cache."""
    return np.random.default_rng([seed, TEST_STREAM]).random((count, n)) < 0.5


def faithfulness(predicted, values) -> float:
    """R² = 1 − Σ(f − f̂)² / Σ(f − f̄)² over held-out masks (SPEX eq. 7): 1 when the surrogate predicts every value,
    0 for the mean alone, below 0 for worse; nan when every value is the same."""
    predicted, values = np.asarray(predicted, dtype=float), np.asarray(values, dtype=float)
    spread = ((values - values.mean()) ** 2).sum()
    if spread == 0:
        return float("nan")
    return float(1 - ((values - predicted) ** 2).sum() / spread)


def removal_mask(surrogate: Surrogate, r: int) -> np.ndarray:
    """m* = argmax over masks with r units removed of |f̂(1) − f̂(m)| (SPEX eq. 8): exact over all C(n, r) masks up
    to 20,000 of them, else greedy, removing one unit at a time."""
    n = surrogate.n
    if not 0 <= r <= n:
        raise ValueError(f"r must lie in 0..{n}")
    full = np.ones(n, dtype=bool)
    top = surrogate.predict(full[None])[0]
    if comb(n, r) <= EXACT_REMOVAL:
        best, best_change = full, -1.0
        for chunk in _chunks(itertools.combinations(range(n), r), 4096):
            masks = np.ones((len(chunk), n), dtype=bool)
            for row, removed in zip(masks, chunk):
                row[list(removed)] = False
            change = np.abs(top - surrogate.predict(masks))
            k = int(change.argmax())
            if change[k] > best_change:
                best, best_change = masks[k], float(change[k])
        return best
    mask = full.copy()
    for _ in range(r):
        options = [i for i in range(n) if mask[i]]
        trial = np.repeat(mask[None], len(options), axis=0)
        trial[np.arange(len(options)), options] = False
        mask = trial[int(np.abs(top - surrogate.predict(trial)).argmax())]
    return mask


def removal(surrogate: Surrogate, value, rs=(1, 2, 3)) -> list[dict]:
    """Top-r removal (SPEX eq. 8): Rem(r) = |f(1) − f(m*)| / |f(1)|, with the value function asked for f(1) and each
    f(m*). `change` is f(m*) − f(1), how the value moves when the units go, and `predicted_change` the surrogate's
    f̂(m*) − f̂(1). Rem divides by |f(1)|, as SPEX's log-probabilities allow; for a value near 0 (a contrast) read
    the change instead."""
    n = surrogate.n
    masks = [removal_mask(surrogate, r) for r in rs]
    values = value(np.vstack([np.ones((1, n), dtype=bool), *[m[None] for m in masks]]))
    f1 = values[0]
    out = []
    for r, m, f in zip(rs, masks, values[1:]):
        out.append({"r": r, "removed": np.flatnonzero(~m).tolist(), "value": float(f), "change": float(f - f1),
                    "predicted_change": float(surrogate.predict(m[None])[0] - surrogate.predict(np.ones((1, n)))[0]),
                    "rem": float(abs(f1 - f) / abs(f1)) if f1 != 0 else float("nan")})
    return out


def recovery(surrogate: Surrogate, relevant, r: int) -> float:
    """Recovery@r (SPEX eq. 9): (1/r) Σ_{i ≤ r} |S* ∩ S_i| / |S_i|, the S_i being the surrogate's r most important
    terms, by absolute value (single units and interactions alike), and S* the human-labelled units. Only a non-zero
    term is one the method found: with fewer than r of them the slots left add 0 (a sparse fit is not credited with
    terms it gave no weight, in the order of the units). Terms tied in size at the r-th place share the slots left to
    them, each adding its mean, so the order of the units never decides the result."""
    if isinstance(r, bool) or not isinstance(r, (int, np.integer)) or r < 1:
        raise ValueError(f"r must be a positive integer, not {r!r}")
    relevant = {int(i) for i in relevant}
    terms = surrogate_terms(surrogate)
    if not all(np.isfinite(w) for w in terms.values()):
        raise ValueError("recovery needs finite terms: the surrogate has a NaN or infinite one")
    found = sorted(((abs(w), len(relevant & set(s)) / len(s)) for s, w in terms.items()
                    if s and w != 0), key=lambda t: -t[0])
    if len(found) <= r:
        return float(sum(h for _, h in found) / r)
    edge = found[r - 1][0]
    above = [h for size, h in found if size > edge]
    tied = [h for size, h in found if size == edge]
    return float((sum(above) + (r - len(above)) * sum(tied) / len(tied)) / r)


def _chunks(iterable, size):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def intervene(prompt, model, check, drop, *, samples: int = 64, sampling=None, replace: str | None = None,
              cache=None) -> dict:
    """Remove units and sample again: the mean of the check's values (the share flagged negative, for a labelling
    check) over `samples` fresh replies to the full prompt and to the prompt without `drop` (unit or group ids), the
    change, and its standard error. Needs only the wrapper's generate."""
    import math  # noqa: PLC0415

    from .responses.checks import verdict  # noqa: PLC0415
    from .targets.sampled import Sampled  # noqa: PLC0415
    from .value import Value  # noqa: PLC0415

    v = Value(prompt, model, Sampled(check, samples=samples, sampling=sampling), replace=replace, cache=cache)
    replies = v.replies(np.vstack([np.ones(prompt.n, dtype=bool), prompt.mask(drop=drop)]))
    rates = []
    for rs in replies:
        values = [x for x in (verdict(check(prompt, r)).value for r in rs) if x is not None]
        if not values:
            raise ValueError("the check labelled none of the replies")
        mean = math.fsum(values) / len(values)
        spread = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        rates.append((mean, spread / math.sqrt(len(values)), len(values)))
    (before, se0, k0), (after, se1, k1) = rates
    return {"dropped": [drop] if isinstance(drop, str) else list(drop), "before": before, "after": after,
            "change": after - before, "se": math.hypot(se0, se1), "replies": [k0, k1], "requests": dict(v.stats)}
