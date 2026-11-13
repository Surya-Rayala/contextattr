"""Your model's wrapper, to fill in (`contextattr template my_wrapper.py` wrote this file). ContextAttr reaches a
model only through an object like this one: a `name`, `score` and, optionally, `generate`.

1. Fill in `score`. Fill in `generate` too if you want the model's own replies, or delete it.
     score      gives the log-probability of each token of a reply you supply. Any model whose log-probabilities
                you can read can do this, in your own process or behind a server. The targets LogProb, Pooled,
                Contrast and Choice read it.
     generate   optional: writes replies to a prompt. ca.sample, ca.intervene and the Sampled target use it.
2. Run this file: `python my_wrapper.py`. check_model calls your wrapper about a dozen times and says what is wrong.
3. Check the results: `contextattr demo my_wrapper.py:MyModel` explains three prompts with planted facts, where the
   right result is known, and says whether your model's results agree.
4. Use it: `ca.explain(prompt, MyModel(), ...)`, or `use = "my_wrapper.py:MyModel"` in a run file.

The guide, with a complete wrapper to copy from and the recipe for a model behind a server:
https://github.com/Surya-Rayala/contextattr/blob/main/docs/wrapper.md
"""

from contextattr import Sampling, Scored, check_model  # noqa: F401


class MyModel:
    # Every answer is cached under this name, so change it whenever the answers could change: another model,
    # revision, precision or prompt format.
    name = "my-model"

    def __init__(self):
        ...   # load the model, or open the client of your server

    def score(self, prompts, responses):
        """Say how likely the model is to give responses[i] as its reply to prompts[i].

        prompts    a list. Each prompt is chat messages, [{"role": "user", "content": "..."}, ...], or a plain string.
        responses  a list of strings, one per prompt.
        return     one Scored per pair: Scored(logprobs), the natural-log probability of each token of the response,
                   in order. For a response of three tokens: Scored([-0.2, -1.3, -0.1])

        Format the prompt as your model expects (its chat template), put the response where the model's reply would
        begin, and read the log-probability the model gives each token of the response. Leave out the end-of-turn
        token. Optional: Scored(logprobs, tokens), with each token's text, lets a target cover part of a response.
        """
        raise NotImplementedError("fill in score")

    def generate(self, prompts, sampling: Sampling):
        """Reply to each prompt.

        prompts    as for score.
        sampling   .n replies per prompt, .temperature (0: always the most likely token), .top_p, .max_tokens,
                   .seed, .stop.
        return     one list per prompt, holding exactly sampling.n replies (the reply's text only).
                   For two prompts and n = 1: [["first reply"], ["second reply"]]
        """
        raise NotImplementedError("fill in generate, or delete it")

    # In both: raise an error on a failure or on a prompt that is too long. Never cut a prompt short.


if __name__ == "__main__":
    print(check_model(MyModel()))
