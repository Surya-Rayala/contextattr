"""prompts.jsonl (interface §2): the interface's example line, split pieces, exact write-read round trips, and the
line and place named in every error."""

import json

import pytest
from hypothesis import given
from prompt_helpers import RIO_LINE, prompts, rio

from contextattr import Group, Prompt, Response, Unit, read_prompts, write_prompts
from contextattr.prompt import prompt_from_dict, prompt_to_dict


def test_the_interface_example_line(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text(RIO_LINE + "\n\n", encoding="utf-8")       # a blank line is skipped
    (p,) = read_prompts(path)
    assert p == rio()
    path.write_text(RIO_LINE + "\r\n", encoding="utf-8-sig")   # a byte-order mark and Windows line ends
    assert read_prompts(path) == [rio()]


def test_a_split_piece_becomes_a_group():
    p = prompt_from_dict({"id": "p", "text": ["Docs: ", {"unit": "d", "text": "A b. C d.", "split": "sentence",
                                                         "sep": "\n", "tag": "doc"}, "Q?"]})
    (g,) = [x for x in p.text if isinstance(x, Group)]
    assert p.unit_ids == ("d.0", "d.1") and dict(p.groups) == {"d": (0, 1)}
    assert [u.tag for u in p.units] == ["doc", "doc"] and g.sep == "\n"
    assert p.render() == "Docs: A b. C d.\nQ?"
    assert prompt_to_dict(p)["text"][1] == {"group": "d", "units": [{"unit": "d.0", "text": "A b.", "sep": " "},
                                                                     {"unit": "d.1", "text": "C d."}],
                                            "sep": "\n", "tag": "doc"}


def test_write_then_read_gives_equal_prompts(tmp_path):
    ps = [rio(),
          Prompt("t", text=["Review: ", Group("w", (Unit("w.0", "great", sep=" ", tag="adj"), Unit("w.1", "film")),
                                              lead=">", sep="!", tag="word", replace="_")],
                 reference="positive", responses=[Response("r", "positive", label="positive", span=(0, 3))])]
    path = tmp_path / "out.jsonl"
    write_prompts(ps, path)
    assert read_prompts(path) == ps
    assert not (tmp_path / "out.jsonl.tmp").exists()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and all(json.loads(line) for line in lines)


@given(prompts())
def test_any_prompt_round_trips_through_json(prompt):
    data = json.loads(json.dumps(prompt_to_dict(prompt), ensure_ascii=False))
    assert prompt_from_dict(data) == prompt


@pytest.mark.parametrize("line, message", [
    ('{"id": "p", "text": [{"unit": "a", "text": "A", "seperator": " "}]}', "text[0]: unknown field 'seperator'"),
    ('{"id": "p", "text": [{"unit": "a"}]}', "text[0]: missing field 'text'"),
    ('{"id": "p", "text": [{"unit": "a", "text": "A", "split": "words"}]}', "split must be one of"),
    ('{"id": "p", "text": [{"unit": "a", "group": "g", "text": "A"}]}', "a unit or a group, not both"),
    ('{"id": "p", "text": [{"text": "A"}]}', "needs a 'unit' or a 'group' field"),
    ('{"id": "p", "text": [{"unit": "a", "text": "A", "sep": 3}]}', "text[0]: unit 'a': sep must be a string"),
    ('{"id": "p", "messages": [{"role": "user"}]}', "messages[0]: missing field 'content'"),
    ('{"id": "p", "messages": {"role": "user"}}', "messages must be a list"),
    ('{"id": "p", "messages": [{"role": "user", "content": [{"unit": "a", "text": "A"}, 4]}]}',
     "messages[0].content[1]"),
    ('{"id": "p", "text": [{"group": "g", "units": [{"unit": "a", "text": "A", "split": "word"}]}]}',
     "text[0].units[0]: unknown field 'split'"),
    ('{"id": "p", "text": [{"unit": "a", "text": "A"}], "responses": [{"id": "r", "text": "x", "label": "good"}]}',
     "responses[0]: response 'r': label must be"),
    ('{"id": "p", "text": [{"unit": "a", "text": "A"}], "responses": [{"id": "r", "text": "x", "span": 1}]}',
     "responses[0]: span must be a list"),
    ('{"id": "p", "text": "no units here"}', "has no units"),
    ('{"text": [{"unit": "a", "text": "A"}]}', "missing field 'id'"),
    ('{"id": "p", "text": [{"unit": "a", "text": "A"}], "extra": 1}', "unknown field 'extra'"),
    ('["not", "an", "object"]', "must be a JSON object"),
    ('{"id": "p", "text": [', "not valid JSON"),
])
def test_errors_name_the_line_and_the_place(tmp_path, line, message):
    path = tmp_path / "bad.jsonl"
    path.write_text(RIO_LINE + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(ValueError) as e:
        read_prompts(path)
    assert "line 2" in str(e.value) and message in str(e.value), str(e.value)


def test_prompt_ids_must_be_unique(tmp_path):
    path = tmp_path / "dup.jsonl"
    path.write_text(RIO_LINE + "\n" + RIO_LINE + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2: prompt id 'rio-01' is already used on line 1"):
        read_prompts(path)
    with pytest.raises(ValueError, match="used twice"):
        write_prompts([rio(), rio()], tmp_path / "x.jsonl")
    assert not (tmp_path / "x.jsonl").exists()
    with pytest.raises(TypeError):
        write_prompts([{"id": "p"}], tmp_path / "x.jsonl")
