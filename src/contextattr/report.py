"""Reports (docs/interface.md §8): one self-contained HTML page (no script, no file beside it) showing each prompt with
its units shaded by score where they stand, then the scores, the top interactions, the removal check and what the
explanation cost."""

from __future__ import annotations

import html
import math

from .prompt import Group, Unit

CSS = """
:root { --bg: #ffffff; --fg: #1f2328; --muted: #57606a; --line: #d0d7de; --card: #f6f8fa;
        --up: 214, 69, 65; --down: 47, 109, 214; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #0d1117; --fg: #e6edf3; --muted: #9198a1; --line: #30363d; --card: #161b22;
          --up: 248, 81, 73; --down: 88, 166, 255; } }
* { box-sizing: border-box; }
body { margin: 0; padding: 24px 16px; background: var(--bg); color: var(--fg);
       font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 960px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px; } h2 { font-size: 18px; margin: 32px 0 6px; }
.meta, .legend { color: var(--muted); font-size: 13px; }
.prompt { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px;
          white-space: pre-wrap; overflow-wrap: anywhere; margin: 10px 0; }
.role { display: block; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted);
        margin-top: 6px; }
.u { border-radius: 3px; padding: 0 1px; }
table { border-collapse: collapse; width: 100%; font-size: 14px; margin: 8px 0; }
th, td { text-align: left; padding: 4px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.wrap { overflow-x: auto; }
"""


def page(explanations, title: str = "ContextAttr report") -> str:
    """The HTML page of a list of explanations."""
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
             f"content='width=device-width, initial-scale=1'><title>{_e(title)}</title><style>{CSS}</style></head>"
             f"<body><main><h1>{_e(title)}</h1><p class='legend'>A unit's shade shows its score: "
             f"<span class='u' style='background: rgba(var(--up), .45)'>pushes the value up</span>, "
             f"<span class='u' style='background: rgba(var(--down), .45)'>pushes it down</span>; hover for the "
             f"number.</p>"]
    for e in explanations:
        parts.append(_section(e))
    parts.append("</main></body></html>")
    return "".join(parts)


def write(explanations, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(page(explanations))


def _section(e) -> str:
    scores = dict(e.scores)
    top = max([abs(v) for v in scores.values()] + [1e-300])
    noise = e.noise["std"] if e.noise else None
    out = [f"<h2>{_e(e.prompt_id or 'attribute')}</h2><p class='meta'>{_e(e.method)} · target {_e(e.target or '')} · "
           f"{e.masks} masks · R² {_fmt(e.r2)} on {e.test_masks} held-out masks · noise {_fmt(noise)} · "
           f"{e.requests.get('requests', 0)} requests</p>"]
    if e.prompt is not None:
        out.append(f"<div class='prompt'>{_prompt_html(e.prompt, scores, top)}</div>")
    rows = sorted(scores.items(), key=lambda t: -t[1])
    texts = dict(zip(e.unit_ids, e.unit_texts or [""] * len(e.unit_ids)))
    out.append("<div class='wrap'><table><tr><th>unit</th><th>score</th><th>text</th></tr>")
    for uid, v in rows:
        out.append(f"<tr><td>{_e(uid)}</td><td class='num'>{v:+.4f}</td><td>{_e(_short(texts.get(uid, '')))}</td></tr>")
    out.append("</table></div>")
    inter = e.top_interactions(10)
    if inter:
        out.append("<div class='wrap'><table><tr><th>units together</th><th>score</th></tr>")
        out += [f"<tr><td>{_e(' & '.join(s))}</td><td class='num'>{w:+.4f}</td></tr>" for s, w in inter]
        out.append("</table></div>")
    if e.removal:
        out.append("<div class='wrap'><table><tr><th>removing the top r</th><th>units removed</th>"
                   "<th>the value moves by</th><th>Rem(r)</th></tr>")
        for r in e.removal:
            removed = ", ".join(e.unit_ids[i] for i in r["removed"])
            out.append(f"<tr><td>{r['r']}</td><td>{_e(removed)}</td><td class='num'>{r['change']:+.4f}</td>"
                       f"<td class='num'>{_fmt(r['rem'])}</td></tr>")
        out.append("</table></div>")
    return "".join(out)


def _prompt_html(prompt, scores, top) -> str:
    def unit(u: Unit) -> str:
        v = scores.get(u.id, 0.0)
        alpha = 0.6 * abs(v) / top
        colour = "--up" if v >= 0 else "--down"
        return (f"<span class='u' title='{_e(u.id)}: {v:+.4f}' style='background: rgba(var({colour}), "
                f"{alpha:.3f})'>{_e(u.text)}</span>{_e(u.sep)}")

    def pieces(content) -> str:
        out = []
        for p in content:
            if isinstance(p, str):
                out.append(_e(p))
            elif isinstance(p, Unit):
                out.append(unit(p))
            elif isinstance(p, Group):
                out.append(_e(p.lead) + "".join(unit(u) for u in p.units) + _e(p.sep))
        return "".join(out)

    if prompt.messages is None:
        return pieces(prompt.text)
    return "".join(f"<span class='role'>{_e(m.role)}</span>{pieces(m.content)}" for m in prompt.messages)


def _e(text) -> str:
    return html.escape(str(text), quote=True)


def _short(text: str, width: int = 120) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= width else text[: width - 1] + "…"


def _fmt(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3g}"
