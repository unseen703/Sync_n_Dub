#!/usr/bin/env python
"""Does every advertised edit phrasing actually parse, and to the right op?

The edit agent advertises its vocabulary in three places — `OP_SPECS.examples`
(the "See all actions" list), `USER_ACTIONS.fill` (the quick-action chips) and
the system prompt. Only the first two are text a user can end up sending, and
for a long time several of them parsed to nothing, or to the WRONG op:
`translate the word "conditional" into Hindi` came back as a whole-dub
language switch.

This runs every one of them through the rule parser — no network, no LLM, no
job folder — and prints the op and scope each produces. A row is OK when the
op the phrasing advertises is among the ops it produced.

Placeholders (`<voice>`, `<style>`, `<language>`, `<speaker>`, `<new wording>`)
are filled from the live catalogues, exactly as the UI's ValuePicker fills
them, so a chip is tested as it is actually sent.

Each phrasing is tested twice: bare, and prefixed with `SEG_0004:` the way the
workspace prefixes a selection. The second run is what proves cue scope
survives — it did not before `_scope_from` learned SEG ids.

    $PY scripts/audit_edit_ops.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.edit_agent import _rule_parse                      # noqa: E402
from backend.edits import OP_SPECS, USER_ACTIONS, tone_styles   # noqa: E402


def _a_voice() -> str:
    """A real voice id, so `use <voice> here` is tested as it is sent."""
    try:
        from models.voices import catalog            # type: ignore
        names = [v for v in catalog() if v]
        if names:
            return str(names[0])
    except Exception:
        pass
    return "Amit"


FILLERS = {
    "voice": _a_voice(),
    "tone": tone_styles()[1],          # [0] is the `neutral` sentinel
    "language": "Marathi",
    "speaker": "SPEAKER_01",
    "text": "यह नया वाक्य है",
}


def fill_placeholders(phrase: str, needs: str | None, options=None) -> str:
    """Fill as the workspace fills: a narrowed `options` list wins."""
    import re
    value = (options or [None])[0] or FILLERS.get(needs or "", "…")
    return re.sub(r"<[^<>]+>", value, phrase)


def describe(op) -> str:
    sc = op.scope
    if sc.segments:
        where = ",".join(sc.segments)
    elif sc.speaker:
        where = sc.speaker
    elif sc.start_s is not None:
        where = f"{sc.start_s}–{sc.end_s}" if sc.end_s is not None else f"@{sc.start_s}"
    elif sc.all:
        where = "ALL"
    else:
        where = "-"
    return f"{op.op}[{where}]"


def main() -> int:
    rows: list[tuple[str, str, str, str, bool]] = []

    for spec in OP_SPECS:
        meta = USER_ACTIONS.get(spec.name) or {}
        cases = [("example", e) for e in (spec.examples or [])]
        if meta.get("fill"):
            cases.append(("fill", fill_placeholders(
                meta["fill"], meta.get("needs"), meta.get("options"))))
        for kind, phrase in cases:
            for label, text in (("bare", phrase), ("scoped", f"SEG_0004: {phrase}")):
                ops = _rule_parse(text)
                got = " + ".join(describe(o) for o in ops) or "(nothing)"
                ok = any(o.op == spec.name for o in ops)
                if ok and label == "scoped":
                    # The selection has to survive, not merely be tolerated.
                    ok = any(o.op == spec.name and "SEG_0004" in o.scope.segments
                             for o in ops)
                rows.append((spec.name, f"{kind}/{label}", text, got, ok))

    width = max(len(r[2]) for r in rows)
    bad = 0
    for op, kind, text, got, ok in rows:
        if not ok:
            bad += 1
        print(f"{'ok ' if ok else 'FAIL'}  {op:<22} {kind:<16} "
              f"{text:<{width}}  ->  {got}")

    print()
    print(f"{len(rows) - bad}/{len(rows)} advertised phrasings parse to their "
          f"own op, with the selection intact.")
    if bad:
        print(f"{bad} still do not. Each one is a chip or an example the UI "
              f"shows and the parser cannot read.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
