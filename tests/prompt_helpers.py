"""Shared by the test_prompt_* files: the example prompt of the interface, texts that stress the splitters, and
Hypothesis strategies for random prompts and masks."""

from hypothesis import strategies as st

from contextattr import Group, Message, Prompt, Response, Unit

RIO_LINE = (
    '{"id": "rio-01", "messages": [{"role": "system", "content": "Answer in one sentence."}, {"role": "user", '
    '"content": ["Documents:\\n", {"unit": "d1", "text": "Rio Carnival takes place before Lent, in February or '
    'March.", "sep": "\\n", "tag": "doc"}, {"unit": "d2", "text": "Summer in Brazil runs from December to March.", '
    '"sep": "\\n", "tag": "doc"}, {"unit": "d3", "text": "Winter in Brazil runs from June to September.", "sep": '
    '"\\n", "tag": "doc"}, "Question: What is the weather like during Rio Carnival?"]}], "reference": "Hot and '
    'humid: it is summer in Brazil.", "responses": [{"id": "bad-1", "text": "It is cold, because Carnival is in '
    'winter.", "label": "negative"}]}')


def rio():
    """The example of docs/interface.md §2, built in Python (RIO_LINE is the same prompt as a prompts.jsonl line)."""
    return Prompt("rio-01", messages=[
        {"role": "system", "content": "Answer in one sentence."},
        {"role": "user", "content": [
            "Documents:\n",
            Unit("d1", "Rio Carnival takes place before Lent, in February or March.", sep="\n", tag="doc"),
            Unit("d2", "Summer in Brazil runs from December to March.", sep="\n", tag="doc"),
            Unit("d3", "Winter in Brazil runs from June to September.", sep="\n", tag="doc"),
            "Question: What is the weather like during Rio Carnival?"]}],
        reference="Hot and humid: it is summer in Brazil.",
        responses=[{"id": "bad-1", "text": "It is cold, because Carnival is in winter.", "label": "negative"}])

# Letters in both cases, digits, sentence punctuation, quotes and brackets, every kind of space and line break, and
# non-Latin characters: what the splitters decide on.
SPLIT_ALPHABET = list("abcXYZ019.!?…,;:\"'”’»«()[]é。！？дЖ" " \t\n\r  　\x0b")
SPLIT_TEXT = st.text(st.sampled_from(SPLIT_ALPHABET), max_size=80) | st.text(max_size=40)
SHORT = st.text(max_size=4)
TAGS = st.none() | st.sampled_from(["doc", "rule"])


@st.composite
def prompts(draw, chat=None):
    """A random prompt: fixed strings, units and groups in one to three messages (or a text prompt), ids unique by
    construction, at least one unit, and a few responses."""
    counter = iter(range(1_000_000))

    def unit():
        return Unit(f"u{next(counter)}", draw(st.text(min_size=1, max_size=10)), sep=draw(SHORT), tag=draw(TAGS),
                    replace=draw(st.none() | SHORT))

    def group():
        gid = f"g{next(counter)}"
        return Group(gid, tuple(unit() for _ in range(draw(st.integers(1, 4)))), lead=draw(SHORT), sep=draw(SHORT),
                     tag=draw(TAGS), replace=draw(st.none() | SHORT))

    def pieces():
        out = [draw(st.sampled_from([lambda: draw(st.text(max_size=8)), unit, group]))()
               for _ in range(draw(st.integers(0, 4)))]
        return out

    def with_a_unit(contents):
        if not any(isinstance(p, (Unit, Group)) for c in contents for p in c):
            k = draw(st.integers(0, len(contents) - 1))
            contents[k].insert(draw(st.integers(0, len(contents[k]))), unit())
        return contents

    responses = []
    for k in range(draw(st.integers(0, 2))):
        text = draw(st.text(min_size=1, max_size=12))
        span = draw(st.none() | st.tuples(st.integers(0, len(text) - 1), st.integers(1, len(text)))
                    .filter(lambda s: s[0] < s[1]))
        responses.append(Response(f"r{k}", text, label=draw(st.sampled_from([None, "positive", "negative"])),
                                  span=span))
    reference = draw(st.none() | st.text(max_size=10))
    pid = draw(st.text(min_size=1, max_size=6))
    if draw(st.booleans()) if chat is None else chat:
        contents = with_a_unit([pieces() for _ in range(draw(st.integers(1, 3)))])
        roles = [draw(st.sampled_from(["system", "user", "assistant", "tool"])) for _ in contents]
        return Prompt(pid, messages=tuple(Message(r, tuple(c)) for r, c in zip(roles, contents)),
                      reference=reference, responses=tuple(responses))
    (content,) = with_a_unit([pieces()])
    return Prompt(pid, text=tuple(content), reference=reference, responses=tuple(responses))


def masks_for(prompt):
    """A strategy for masks of this prompt, as tuples of booleans."""
    return st.tuples(*([st.booleans()] * prompt.n))
