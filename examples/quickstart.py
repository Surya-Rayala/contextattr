"""A first try, on any machine in seconds: the interface's Rio prompt explained through ToyModel.demo(), a stand-in
model whose answers are known. Swap in your own wrapper to explain a real model (`contextattr template my_wrapper.py`
writes one to fill in).

    python examples/quickstart.py [report.html]
"""

import sys

import contextattr as ca

prompt = ca.Prompt("rio-01", messages=[
    {"role": "system", "content": "Answer in one sentence."},
    {"role": "user", "content": [
        "Documents:\n",
        ca.Unit("d1", "Rio Carnival takes place before Lent, in February or March.", sep="\n", tag="doc"),
        ca.Unit("d2", "Summer in Brazil runs from December to March.", sep="\n", tag="doc"),
        ca.Unit("d3", "Winter in Brazil runs from June to September.", sep="\n", tag="doc"),
        "Question: What is the weather like during Rio Carnival?"]}],
    responses=[ca.Response("bad-1", "It is cold, because Carnival is in winter.", label="negative"),
               ca.Response("good-1", "It is hot and humid: it is summer in Brazil.", label="positive")])

model = ca.ToyModel.demo()
print(ca.check_model(model), "\n")

# What moves the model from the good answer to the bad one? Three units have only 8 masks: exact Shapley values.
e = ca.explain(prompt, model, ca.Contrast(), method="kernelshap", budget=8, removal=(1,))
print(e, "\n")

# The behaviour itself: how often the model says "cold", sampled afresh, with d3 and without it.
cold = ca.checks.regex(r"\bcold\b")
print("removing", e.top(1)[0][0], "→", ca.intervene(prompt, model, cold, [u for u, _ in e.top(1)], samples=200))

if len(sys.argv) > 1:
    e.to_html(sys.argv[1])
    print("report:", sys.argv[1])
