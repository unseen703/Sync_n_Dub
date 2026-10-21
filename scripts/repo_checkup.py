#!/usr/bin/env python3
"""Structural checkup for the dubbing-studio repo.

Snapshots what the repo *is* — directories, stage modules, adapter registries,
pipeline stages, CLI flags, scripts, skills, envs, GPUs — diffs that against the
last snapshot, and when something moved:

  1. regenerates the AUTOGEN block in README.md (deterministic, always current)
  2. appends a human-readable entry to CHANGES-PENDING.md
  3. stores the new snapshot in .repo-state.json

CHANGES-PENDING.md is a buffer, not a log: Claude folds its entries into the
prose parts of README.md and then clears it. The AUTOGEN block needs no such
help — it is rewritten from the live tree on every run.

Parses source rather than importing it; a cron job must not pull in torch.

    python scripts/repo_checkup.py            # check, update if changed
    python scripts/repo_checkup.py --dry-run  # report only, write nothing
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / ".repo-state.json"
README = ROOT / "README.md"
PENDING = ROOT / "CHANGES-PENDING.md"

BEGIN = "<!-- BEGIN AUTOGEN:structure -->"
END = "<!-- END AUTOGEN:structure -->"

SKIP_DIRS = {"__pycache__", ".git", "node_modules", ".ipynb_checkpoints"}

# This script's own outputs — listing them would make every run report itself.
SELF_MANAGED = {"CHANGES-PENDING.md", ".repo-state.json"}


# --------------------------------------------------------------------------
# snapshot
# --------------------------------------------------------------------------

def _pyfiles(d: Path) -> list[str]:
    if not d.is_dir():
        return []
    return sorted(
        p.stem for p in d.glob("*.py")
        if p.stem != "__init__" and "__pycache__" not in p.parts
    )


def _registry(init_py: Path) -> list[str]:
    """Pull adapter keys out of a _REGISTRY_SPEC dict literal without importing."""
    if not init_py.is_file():
        return []
    try:
        tree = ast.parse(init_py.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return []
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        for t in targets:
            if isinstance(t, ast.Name) and t.id == "_REGISTRY_SPEC":
                if isinstance(node.value, ast.Dict):
                    return sorted(
                        k.value for k in node.value.keys
                        if isinstance(k, ast.Constant) and isinstance(k.value, str)
                    )
    return []


def _stage_key(stage: str) -> tuple:
    """Sort '4b' after '4' and '10a' after '9h' — stage ids are not plain ints."""
    m = re.match(r"\s*(\d+)([a-z0-9]*)", stage)
    return (int(m.group(1)), m.group(2)) if m else (999, stage)


def _stages() -> list[str]:
    p = ROOT / "backend" / "pipeline.py"
    if not p.is_file():
        return []
    out, seen = [], set()
    for m in re.finditer(r'"=== stages? (.+?) ==="',
                         p.read_text(encoding="utf-8", errors="replace")):
        s = m.group(1)
        s = re.sub(r"\s*\([^)]*%s[^)]*\)", "", s)   # drop "(%s, full_audio=%s)"
        s = s.replace("%s", "").replace("  ", " ")
        s = re.sub(r"\s*(→|,|script=|source=)\s*$", "", s.strip()).strip(" ,→")
        if s and s not in seen and "SKIPPED" not in s:
            seen.add(s)
            out.append(s)
    return sorted(out, key=_stage_key)


def _cli_flags() -> list[str]:
    p = ROOT / "backend" / "cli.py"
    if not p.is_file():
        return []
    txt = p.read_text(encoding="utf-8", errors="replace")
    return sorted(set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', txt)))


def _gpus() -> list[str]:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
        if r.returncode == 0:
            return [l.strip() for l in r.stdout.strip().splitlines() if l.strip()]
    except (OSError, subprocess.SubprocessError):
        pass
    return []


def _envs() -> list[str]:
    envs = Path("/data1/dipak_data/miniconda3/envs")
    out = sorted(p.name for p in envs.iterdir() if p.is_dir()) if envs.is_dir() else []
    out += sorted(p.name for p in ROOT.glob(".venv*") if p.is_dir())
    return out


def _toplevel() -> list[str]:
    out = []
    for p in sorted(ROOT.iterdir()):
        if p.name.startswith(".") or p.name in SKIP_DIRS or p.name in SELF_MANAGED:
            continue
        out.append(p.name + ("/" if p.is_dir() else ""))
    return out


def snapshot() -> dict:
    models = ROOT / "models"
    reg = {}
    if models.is_dir():
        for d in sorted(models.iterdir()):
            if not d.is_dir() or d.name in SKIP_DIRS:
                continue
            keys = _registry(d / "__init__.py")
            if keys:
                reg[d.name] = keys
            for sub in sorted(x for x in d.iterdir() if x.is_dir()):
                if sub.name in SKIP_DIRS:
                    continue
                k = _registry(sub / "__init__.py")
                if k:
                    reg[f"{d.name}.{sub.name}"] = k

    skills = ROOT / ".claude" / "skills"
    return {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "toplevel": _toplevel(),
        "backend": _pyfiles(ROOT / "backend"),
        "registries": reg,
        "stages": _stages(),
        "cli_flags": _cli_flags(),
        "scripts": sorted(
            p.name for p in (ROOT / "scripts").glob("*")
            if p.is_file() and p.suffix in {".py", ".sh"}
        ) if (ROOT / "scripts").is_dir() else [],
        "tests": _pyfiles(ROOT / "tests"),
        "skills": sorted(
            p.name for p in skills.iterdir() if p.is_dir()
        ) if skills.is_dir() else [],
        "third_party": sorted(
            p.name for p in (ROOT / "third_party").iterdir() if p.is_dir()
        ) if (ROOT / "third_party").is_dir() else [],
        "envs": _envs(),
        "gpus": _gpus(),
    }


# --------------------------------------------------------------------------
# diff
# --------------------------------------------------------------------------

LABELS = {
    "toplevel": "Top-level entries", "backend": "backend/ modules",
    "stages": "Pipeline stages", "cli_flags": "CLI flags",
    "scripts": "scripts/", "tests": "tests/", "skills": "Project skills",
    "third_party": "third_party/ clones", "envs": "Environments", "gpus": "GPUs",
}


def diff(old: dict, new: dict) -> list[str]:
    if not old:
        return ["Initial snapshot — no previous state to compare against."]
    lines = []
    for key, label in LABELS.items():
        a, b = set(old.get(key, [])), set(new.get(key, []))
        if added := sorted(b - a):
            lines.append(f"- **{label}** added: {', '.join(f'`{x}`' for x in added)}")
        if gone := sorted(a - b):
            lines.append(f"- **{label}** removed: {', '.join(f'`{x}`' for x in gone)}")

    oreg, nreg = old.get("registries", {}), new.get("registries", {})
    for stage in sorted(set(oreg) | set(nreg)):
        a, b = set(oreg.get(stage, [])), set(nreg.get(stage, []))
        if added := sorted(b - a):
            lines.append(f"- **{stage} adapters** registered: "
                         f"{', '.join(f'`{x}`' for x in added)}")
        if gone := sorted(a - b):
            lines.append(f"- **{stage} adapters** removed: "
                         f"{', '.join(f'`{x}`' for x in gone)}")
    return lines


# --------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------

def render(s: dict) -> str:
    L = [BEGIN, "",
         f"*Regenerated by `scripts/repo_checkup.py` — {s['generated']}. "
         f"Edits inside this block are overwritten; write prose outside it.*", ""]

    L += ["### Layout", "", "```"]
    L += [f"{e}" for e in s["toplevel"]]
    L += ["```", ""]

    if s["backend"]:
        L += ["### `backend/` modules", "",
              "  ".join(f"`{m}`" for m in s["backend"]), ""]

    if s["registries"]:
        L += ["### Adapter registries", "",
              "Every stage resolves its implementation through a `_REGISTRY_SPEC`, "
              "so adding a model means writing one adapter and registering it.", "",
              "| Stage | Registered |", "|---|---|"]
        for stage, keys in s["registries"].items():
            L.append(f"| `{stage}` | {', '.join(f'`{k}`' for k in keys)} |")
        L.append("")

    if s["stages"]:
        L += [f"### Pipeline stages ({len(s['stages'])})", "",
              "In execution order, as logged by `backend/pipeline.py`:", "", "```"]
        for st in s["stages"]:
            sid, _, desc = st.partition(":")
            L.append(f"{sid.strip():<5} {desc.strip()}" if desc else st)
        L += ["```", ""]

    for key, head in (("scripts", "### `scripts/`"),
                      ("tests", "### `tests/`"),
                      ("third_party", "### `third_party/` clones")):
        if s[key]:
            L += [head, "", "  ".join(f"`{x}`" for x in s[key]), ""]

    if s["skills"]:
        L += ["### Project skills (`.claude/skills/`)", "",
              "  ".join(f"`{x}`" for x in s["skills"]), ""]

    if s["envs"] or s["gpus"]:
        L += ["### Host", ""]
        if s["gpus"]:
            L.append(f"GPUs: {'; '.join(s['gpus'])}")
            L.append("")
        if s["envs"]:
            L.append(f"Environments: {', '.join(f'`{e}`' for e in s['envs'])}")
            L.append("")

    L += [f"CLI surface: {len(s['cli_flags'])} flags on `backend/cli.py` "
          f"(`python -m backend.cli --help`).", "", END]
    return "\n".join(L)


def splice(block: str) -> bool:
    txt = README.read_text(encoding="utf-8") if README.is_file() else ""
    if BEGIN in txt and END in txt:
        new = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), block, txt, flags=re.S)
    else:
        new = (txt.rstrip() + "\n\n" + block + "\n") if txt.strip() else block + "\n"
    if new == txt:
        return False
    README.write_text(new, encoding="utf-8")
    return True


PENDING_HEADER = (
    "# Pending structural changes\n\n"
    "Buffer written by `scripts/repo_checkup.py`. Entries sit here until they\n"
    "are folded into the prose sections of [README.md](README.md), after which\n"
    "this file is cleared with `python scripts/repo_checkup.py --clear`.\n\n"
    "The generated structure block in README.md is already current — it is\n"
    "rewritten on every checkup. What lives here is the *narrative* follow-up:\n"
    "what changed, so someone can say why it changed and what it means.\n"
)


def clear_pending() -> None:
    PENDING.write_text(PENDING_HEADER + "\n---\n\n*Nothing pending.*\n",
                       encoding="utf-8")


def log_pending(changes: list[str]) -> None:
    head = PENDING_HEADER
    prev = PENDING.read_text(encoding="utf-8") if PENDING.is_file() else ""
    body = prev[len(head):] if prev.startswith(head) else (
        prev.split("\n---\n", 1)[1] if "\n---\n" in prev else ""
    )
    body = re.sub(r"^\s*(?:---\s*)+", "", body)   # don't stack separators
    body = body.replace("*Nothing pending.*", "").strip()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    PENDING.write_text(
        head + "\n---\n" + body.rstrip()
        + f"\n\n## {stamp}\n\n" + "\n".join(changes) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="report what changed, write nothing")
    ap.add_argument("--clear", action="store_true",
                    help="empty CHANGES-PENDING.md, after folding it into README prose")
    args = ap.parse_args()

    if args.clear:
        clear_pending()
        print(f"repo-checkup: cleared {PENDING.name}")
        return 0

    new = snapshot()
    old = {}
    if STATE.is_file():
        try:
            old = json.loads(STATE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    comparable = {k: v for k, v in new.items() if k != "generated"}
    unchanged = old and {k: v for k, v in old.items() if k != "generated"} == comparable

    if unchanged:
        print("repo-checkup: no structural change")
        return 0

    changes = diff(old, new)
    print(f"repo-checkup: {len(changes)} structural change(s)")
    for c in changes:
        print("  " + c)

    if args.dry_run:
        print("repo-checkup: --dry-run, nothing written")
        return 0

    if splice(render(new)):
        print(f"repo-checkup: rewrote AUTOGEN block in {README.name}")
    log_pending(changes)
    print(f"repo-checkup: logged to {PENDING.name}")
    STATE.write_text(json.dumps(new, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
