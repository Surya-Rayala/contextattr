"""check_model (docs/wrapper.md) on the toy model, and on wrappers that each break one rule."""

from contextattr import Prompt, Scored, ToyModel, Unit, check_model


def statuses(report):
    return {c.name: c.status for c in report.checks}


class NoTokens(ToyModel):
    def score(self, prompts, responses):
        return [Scored(s.logprobs) for s in super().score(prompts, responses)]


class WrongTokens(ToyModel):
    def score(self, prompts, responses):
        return [Scored(s.logprobs, tuple(t.strip() for t in s.tokens)) for s in super().score(prompts, responses)]


class Drifting(ToyModel):
    """Each call scores a little lower than the last: answers that vary between calls."""
    calls = 0

    def score(self, prompts, responses):
        self.calls += 1
        return [Scored(tuple(x - 1e-3 * self.calls for x in s.logprobs), s.tokens)
                for s in super().score(prompts, responses)]


class Down(ToyModel):
    def score(self, prompts, responses):
        raise RuntimeError("server down")


class Unseeded(ToyModel):
    draws = 0

    def generate(self, prompts, sampling):
        if sampling.temperature == 0:
            return super().generate(prompts, sampling)
        self.draws += 1
        return [[f"reply {self.draws}"] * sampling.n for _ in prompts]


class Counting(ToyModel):
    calls = 0

    def score(self, prompts, responses):
        self.calls += 1
        return super().score(prompts, responses)

    def generate(self, prompts, sampling):
        self.calls += 1
        return super().generate(prompts, sampling)


def test_the_demo_toy_model_is_ready():
    r = check_model(ToyModel.demo())
    assert r.ok and r.scores and r.generates and r.spans
    s = statuses(r)
    assert {s[k] for k in ("generate", "greedy repeats", "seed", "score", "score repeats", "token texts",
                           "prompt matters")} == {"ok"}
    assert s["reply above reversed"] == "note"          # the toy model is blind to word order
    assert r.targets == ["LogProb", "Pooled", "Contrast", "Choice", "Sampled"]
    assert "ready: every check passed" in str(r)


def test_a_model_that_ignores_the_prompt_is_not_ready():
    r = check_model(ToyModel())
    assert not r.ok and statuses(r)["prompt matters"] == "FAIL" and "NOT ready" in str(r)


def test_no_name_or_no_method():
    class NoName:
        def score(self, prompts, responses):
            return []

    class Nothing:
        name = "nothing"

    r = check_model(NoName())
    assert not r.ok and statuses(r) == {"name": "FAIL"}
    r = check_model(Nothing())
    assert not r.ok and statuses(r)["methods"] == "FAIL"


def test_each_broken_rule_shows():
    r = check_model(NoTokens.demo())
    assert r.ok and not r.spans and statuses(r)["token texts"] == "note"
    r = check_model(WrongTokens.demo())
    assert not r.ok and statuses(r)["token texts"] == "FAIL"
    r = check_model(Drifting.demo())
    assert r.ok and statuses(r)["score repeats"] == "note"
    r = check_model(Down.demo())
    assert not r.ok and statuses(r)["score"] == "FAIL" and "server down" in str(r) and r.generates
    r = check_model(Unseeded.demo())
    assert r.ok and statuses(r)["seed"] == "note"


def test_a_generate_only_wrapper_serves_the_sampled_target():
    class GenerateOnly:
        name = "gen"

        def generate(self, prompts, sampling):
            return [["a reply"] * sampling.n for _ in prompts]

    r = check_model(GenerateOnly())
    assert r.ok and r.generates and not r.scores and r.targets == ["Sampled"] and "targets: Sampled" in str(r)


def test_a_text_prompt_and_a_given_response():
    p = Prompt("t", text=["Docs: ", Unit("w", "Winter in Brazil runs from June.")])
    r = check_model(ToyModel.demo(), prompt=p, response="It is cold.")
    assert r.ok and statuses(r)["prompt matters"] == "ok"


def test_about_a_dozen_calls():
    m = Counting.demo()
    check_model(m)
    assert m.calls <= 12
