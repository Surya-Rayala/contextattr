"""The demonstration on planted facts (src/contextattr/demo.py, docs/wrapper.md §10). On its stand-in every line says
ok. A wrapper that can only score gets the three cases without the replies; one that can only generate gets the first
case, by counting replies. The four interaction methods join with CA_TEST_SHAPIQ=1 (marked shapiq)."""

import pytest

import contextattr as ca
from contextattr import cli
from contextattr.demo import DOCUMENTS, case, planted_toy


def run(model=None, **settings):
    lines = []
    passed = ca.demo(model, show=lines.append, **settings)
    return passed, "\n".join(lines)


class OnlyScores:
    name = "only-scores"

    def __init__(self):
        self.toy = planted_toy()

    def score(self, prompts, responses):
        return self.toy.score(prompts, responses)


class OnlyGenerates:
    name = "only-generates"

    def __init__(self):
        self.toy = planted_toy()

    def generate(self, prompts, sampling):
        return self.toy.generate(prompts, sampling)


def test_the_cases_have_eight_documents_and_keep_the_question():
    for k in (1, 2, 3):
        prompt = case(k)
        assert prompt.n == len(DOCUMENTS[k]) == 8 and [u.id for u in prompt.units] == [f"d{i}" for i in range(1, 9)]
        assert prompt.render([0] * 8)[-1]["content"].startswith("Documents:\nQuestion: ")


def test_the_stand_in_passes_every_check_with_the_core_methods():
    passed, text = run(interactions=False)
    assert passed and "FAIL" not in text and "\n  note " not in text
    assert "5 of 5 checks of the package passed." in text and "8 of 8 expectations about the model held." in text
    assert "fixed rules, no model and no weights" in text and 'pip install "contextattr[interactions]"' in text
    assert text.count("loo, banzhaf, kernelshap = brute force") == 3 and "over all 256 combinations" in text
    assert "first: d2, d2, d2, d2, d2" in text and "highest/lowest: d4/d2, d4/d2, d4/d2, d4/d2, d4/d2" in text
    assert "Together they add +0.645 beyond their separate effects." in text      # the stand-in's planted pair
    assert "its answer: '1932'" in text and "without d2: 'I do not know.'" in text


def test_a_wrapper_that_only_scores_gets_the_cases_without_the_replies():
    passed, text = run(OnlyScores(), interactions=False)
    assert passed and "this wrapper cannot generate, so no replies are read" in text
    assert "its answer" not in text and "5 of 5 checks of the package passed." in text
    assert "4 of 4 expectations about the model held." in text


def test_a_wrapper_that_only_generates_gets_the_first_case_by_counting_replies():
    passed, text = run(OnlyGenerates(), interactions=False)
    assert passed and "This wrapper can only generate" in text and "== 2." not in text
    assert "ok    loo = the measured change of the share" in text and "ok    d2 comes first" in text
    assert "100% of 16 replies before, 0% after" in text
    assert "1 of 1 checks of the package passed." in text and "2 of 2 expectations about the model held." in text


def test_a_model_that_reads_the_documents_otherwise_gets_notes_and_still_passes():
    blind = ca.ToyModel(bias=-1.0, default_reply="I do not know.", name="reads-nothing")
    passed, text = run(blind, interactions=False)
    assert passed and "FAIL" not in text and text.count("\n  note ") >= 5
    assert "5 of 5 checks of the package passed." in text and "They say how this model read the documents" in text


def test_an_object_that_is_no_wrapper_fails():
    passed, text = run(type("Nothing", (), {"name": "nothing"})(), interactions=False)
    assert not passed and "FAIL  the wrapper" in text


def test_the_cache_file_makes_a_second_run_free(tmp_path):
    counted = OnlyScores()
    asked = []
    score = counted.score
    counted.score = lambda prompts, responses: (asked.append(len(prompts)), score(prompts, responses))[1]
    assert run(counted, interactions=False, cache=tmp_path / "demo.sqlite")[0] and sum(asked) == 1024
    asked.clear()
    assert run(counted, interactions=False, cache=tmp_path / "demo.sqlite")[0] and not asked


@pytest.mark.shapiq
def test_the_stand_in_passes_every_check_with_all_nine_methods(capfd):
    passed, text = run()
    printed = capfd.readouterr()
    assert passed and "FAIL" not in text and "\n  note " not in text
    assert "6 of 6 checks of the package passed." in text and "8 of 8 expectations about the model held." in text
    assert "first: " + ", ".join(["d2"] * 9) in text and "highest/lowest: " + ", ".join(["d4/d2"] * 9) in text
    assert "faithbanzhaf with every set +0.645; spex +0.645" in text
    assert "LightGBM" not in printed.out + printed.err          # ProxySPEX's tree library prints nothing any more


@pytest.mark.shapiq
def test_the_command_runs_it_on_the_stand_in_and_on_a_named_wrapper(capsys):
    assert cli.main(["demo"]) == 0
    assert "contextattr demo: toy-planted" in capsys.readouterr().out
    assert cli.main(["demo", "contextattr.demo:planted_toy"]) == 0
    out = capsys.readouterr().out
    assert "fixed rules, no model and no weights" not in out and "6 of 6 checks of the package passed." in out
