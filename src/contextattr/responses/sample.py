"""Responses sampled from the full prompt and labelled (docs/interface.md §4): `sample` adds k replies of the model to
a prompt's responses, each labelled by a check, so that scored targets (LogProb, Pooled, Contrast) can explain them."""

from __future__ import annotations

import numpy as np

from ..cache import Cache
from ..prompt import Prompt, Response
from ..wrapper.contract import Sampling
from .checks import verdict

DEFAULT = Sampling(n=8, temperature=0.7, top_p=0.95, max_tokens=256, seed=0)


def sample(prompt: Prompt, model, sampling: Sampling = DEFAULT, check=None, *, cache: Cache | str | None = None,
           replace: str | None = None) -> Prompt:
    """The prompt with `sampling.n` replies to its full text added to its responses, as sample-0, sample-1, …, each
    labelled by `check` (none: unlabelled). Empty replies are left out. The replies are cached like every answer."""
    from ..value import Value     # noqa: PLC0415 - value imports the targets, which import this package
    from ..targets.sampled import Sampled

    replies = Value(prompt, model, Sampled(lambda p, r: 0.0, samples=sampling.n, sampling=sampling), replace=replace,
                    cache=cache).replies(np.ones((1, prompt.n), dtype=bool))[0]
    taken = {r.id for r in prompt.responses}
    added = []
    for k, text in enumerate(replies):
        if not text.strip():
            continue
        rid = f"sample-{k}"
        if rid in taken:
            raise ValueError(f"prompt {prompt.id!r} already has a response {rid!r}")
        v = verdict(check(prompt, text)) if check is not None else None
        added.append(Response(rid, text, label=None if v is None else v.label, span=None if v is None else v.span))
    return Prompt(prompt.id, messages=prompt.messages, text=prompt.text, reference=prompt.reference,
                  responses=(*prompt.responses, *added))
