"""The cache: stable keys, exact answers, the first answer kept, persistence and many keys at once."""

import math

import pytest

from contextattr import Cache


def test_keys_ignore_field_order_and_tell_requests_apart():
    a = Cache.key({"kind": "score", "prompt": [{"role": "user", "content": "p"}], "response": "r"})
    assert a == Cache.key({"response": "r", "prompt": [{"content": "p", "role": "user"}], "kind": "score"})
    assert a != Cache.key({"kind": "score", "prompt": [{"role": "user", "content": "p"}], "response": "r "})
    assert len(a) == 64


def test_answers_round_trip_exactly(tmp_path):
    c = Cache(tmp_path / "c.sqlite")
    answers = {"k1": {"logprobs": [-0.1, -1e-300, -123.45678901234568], "tokens": ["a", "é", "\n"]},
               "k2": 0.30000000000000004, "k3": None}
    c.put_many("score", answers)
    assert c.get_many(["k1", "k2", "k3", "absent"]) == answers


def test_the_first_answer_stays():
    c = Cache()
    c.put_many("x", {"k": 1})
    c.put_many("x", {"k": 2})
    assert c.get_many(["k"]) == {"k": 1} and len(c) == 1


def test_it_persists_and_handles_many_keys(tmp_path):
    path = tmp_path / "new" / "dir" / "c.sqlite"
    with Cache(path) as c:
        c.put_many("x", {f"k{i}": i for i in range(1234)})
    again = Cache(path)
    assert len(again) == 1234
    assert again.get_many([f"k{i}" for i in range(1234)] + ["absent"]) == {f"k{i}": i for i in range(1234)}


def test_nan_is_refused():
    with pytest.raises(ValueError):
        Cache().put_many("x", {"a": math.nan})
    with pytest.raises(ValueError):
        Cache.key({"a": math.inf})
