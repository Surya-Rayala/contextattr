"""ToyModel: a compatible wrapper with known answers and no real model behind it, for tests and a first try."""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections.abc import Mapping, Sequence

from .contract import Rendered, Sampling, Scored

_TOKEN = re.compile(r"\s*\S+|\s+$")   # each word with the spaces before it; trailing spaces as a last token
_EDGE = re.compile(r"^\W+|\W+$")


class ToyModel:
    """A stand-in model whose answers are known exactly.

    A response is cut into words, each with the spaces before it. Word w gets the log-probability
    log σ(bias + Σ rules[cue][w] + Σ pairs[(cue1, cue2)][w]), summed over the cues present in the prompt, and
    over the pairs whose two cues are both present. A cue is present when its text occurs in the prompt, case
    ignored (a chat prompt's message contents are joined with newlines); a word is matched lower-cased, without
    punctuation around it.

    `generate` at temperature 0 gives the reply of the first cue in `replies` that is present, else `default_reply`,
    sampling.n times. Above temperature 0 each reply is drawn uniformly from the replies of all the cues present (the
    default when none is), so its rate is known exactly; the draw is seeded by the seed and the prompt's text, and
    seed None draws afresh."""

    def __init__(self, rules: Mapping[str, Mapping[str, float]] | None = None, *,
                 pairs: Mapping[tuple[str, str], Mapping[str, float]] | None = None, bias: float = 0.0,
                 replies: Mapping[str, str] | None = None, default_reply: str = "I do not know.",
                 name: str = "toy"):
        self.rules = {_cue(c): _weights(w, c) for c, w in (rules or {}).items()}
        self.pairs = {}
        for key, w in (pairs or {}).items():
            if not (isinstance(key, tuple) and len(key) == 2):
                raise TypeError(f"ToyModel: a pair is a tuple of two cues, not {key!r}")
            self.pairs[(_cue(key[0]), _cue(key[1]))] = _weights(w, key)
        if isinstance(bias, bool) or not isinstance(bias, (int, float)) or not math.isfinite(bias):
            raise TypeError("ToyModel: bias must be a finite number")
        self.bias = float(bias)
        self.replies = {_cue(c): r for c, r in (replies or {}).items()}
        if not all(isinstance(r, str) for r in self.replies.values()) or not isinstance(default_reply, str):
            raise TypeError("ToyModel: replies must be strings")
        self.default_reply = default_reply
        self.name = name

    @classmethod
    def demo(cls) -> ToyModel:
        """A toy model of the interface's Rio example: Brazil's winter makes "cold" likely, its summer "hot" and
        "humid", more so when the prompt also says Carnival falls in February or March; it replies "cold" when it
        reads only of the winter."""
        return cls(rules={"winter in brazil": {"cold": 2.0, "winter": 2.0},
                          "summer in brazil": {"hot": 2.0, "humid": 1.0, "summer": 2.0}},
                   pairs={("february or march", "summer in brazil"): {"hot": 1.5, "summer": 1.5}},
                   bias=-1.0,
                   replies={"summer in brazil": "It is hot and humid: it is summer in Brazil.",
                            "winter in brazil": "It is cold, because Carnival is in winter."},
                   name="toy-demo")

    def score(self, prompts: Sequence[Rendered], responses: Sequence[str]) -> list[Scored]:
        if len(prompts) != len(responses):
            raise ValueError(f"ToyModel.score: {len(prompts)} prompts but {len(responses)} responses")
        out = []
        for prompt, response in zip(prompts, responses):
            if not isinstance(response, str):
                raise TypeError(f"ToyModel.score: a response is a string, not {type(response).__name__}")
            text = _text(prompt)
            tokens = _TOKEN.findall(response)
            out.append(Scored(tuple(self._logprob(text, t) for t in tokens), tuple(tokens)))
        return out

    def generate(self, prompts: Sequence[Rendered], sampling: Sampling) -> list[list[str]]:
        out = []
        for prompt in prompts:
            text = _text(prompt)
            present = [r for c, r in self.replies.items() if c in text] or [self.default_reply]
            if sampling.temperature == 0 or len(present) == 1:
                out.append([present[0]] * sampling.n)
                continue
            rng = random.Random(None if sampling.seed is None else _seed(text, sampling.seed))
            out.append([rng.choice(present) for _ in range(sampling.n)])
        return out

    def _logprob(self, text: str, token: str) -> float:
        word = _EDGE.sub("", token.strip()).lower()
        x = self.bias
        x += sum(w.get(word, 0.0) for c, w in self.rules.items() if c in text)
        x += sum(w.get(word, 0.0) for (c1, c2), w in self.pairs.items() if c1 in text and c2 in text)
        return log_sigmoid(x)


def _seed(text: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}\0{text}".encode()).digest()[:8], "big")


def log_sigmoid(x: float) -> float:
    """log σ(x) = log(1 / (1 + e^-x)), without overflow."""
    return min(x, 0.0) - math.log1p(math.exp(-abs(x)))


def _text(prompt: Rendered) -> str:
    """The prompt's text, lower-cased; a chat prompt's contents joined with newlines. Checks the rendered shape."""
    if isinstance(prompt, str):
        return prompt.lower()
    if not isinstance(prompt, list) or not all(isinstance(m, dict) and set(m) == {"role", "content"} and
                                               isinstance(m["role"], str) and isinstance(m["content"], str)
                                               for m in prompt):
        raise TypeError("ToyModel: a prompt is a string or a list of {'role', 'content'} string dicts")
    return "\n".join(m["content"] for m in prompt).lower()


def _cue(cue) -> str:
    if not isinstance(cue, str) or not cue:
        raise TypeError(f"ToyModel: a cue is a non-empty string, not {cue!r}")
    return cue.lower()


def _weights(w, cue) -> dict[str, float]:
    if not isinstance(w, Mapping):
        raise TypeError(f"ToyModel: the weights of {cue!r} are a mapping word → number")
    out = {}
    for word, x in w.items():
        if not isinstance(word, str) or isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
            raise TypeError(f"ToyModel: weight {word!r}: {x!r} of {cue!r} must be a word and a finite number")
        out[word.lower()] = float(x)
    return out
