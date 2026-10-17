# Pending structural changes

Buffer written by `scripts/repo_checkup.py`. Entries sit here until they
are folded into the prose sections of [README.md](README.md), after which
this file is cleared with `python scripts/repo_checkup.py --clear`.

The generated structure block in README.md is already current — it is
rewritten on every checkup. What lives here is the *narrative* follow-up:
what changed, so someone can say why it changed and what it means.

---


## 2026-10-17 14:25

- **backend/ modules** added: `speaker_id`
- **Pipeline stages** added: `4c: speaker label refinement`
- **CLI flags** added: `--refine-speaker-labels`, `--speaker-min-speech-s`, `--speaker-purity-floor`
- **scripts/** added: `benchmark_gender.py`
- **tests/** added: `test_speaker_id`
