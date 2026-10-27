"""Every version of the merged ASR+MT JSON is kept, and none of them can be
mistaken for the live one.

The merged JSON is rewritten IN PLACE by every post-MT stage, so without
snapshots the original translation stops existing as soon as 9b runs. The
naming is the delicate half: `edit_agent._find_merged_json` globs
`mt/*.mt.json` and, when several match, takes **the newest by mtime** with
only a log line — so a version file in the wrong place with the wrong suffix
would silently become the document every later edit is applied to.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from backend import mt_versions  # noqa: E402


def _live(tmp_path: Path, n=2) -> Path:
    mt = tmp_path / "mt"
    mt.mkdir(parents=True, exist_ok=True)
    p = mt / "clip_0to60.bodhan.hi.mt.json"
    p.write_text(json.dumps(
        {"segments": [{"id": f"SEG_{i:04d}", "mt_text": "x"} for i in range(n)]}))
    return p


# --- the collision guarantees ---------------------------------------------

def test_a_version_is_never_picked_up_as_the_live_document(tmp_path):
    """The regression this naming exists for.

    `_find_merged_json` globs `mt/*.mt.json` and prefers the NEWEST on a
    tie. A snapshot is newer than the live file by construction, so if it
    answered to that pattern it would win every time.
    """
    live = _live(tmp_path)
    for producer in ("original", "codemix", "condense", "edit_000"):
        mt_versions.snapshot(live, producer)

    mt_dir = live.parent
    assert sorted(p.name for p in mt_dir.glob("*.mt.json")) == [live.name]
    # Even recursively, no snapshot answers to the live pattern.
    assert [p.name for p in mt_dir.rglob("*.mt.json")] == [live.name]


def test_versions_live_in_a_subdirectory(tmp_path):
    live = _live(tmp_path)
    dest = mt_versions.snapshot(live, "original")
    assert dest.parent.name == "versions"
    assert dest.parent.parent == live.parent


def test_version_numbers_are_zero_padded_so_they_sort(tmp_path):
    """`v10` sorts before `v9` unpadded, which is how a history silently
    starts reading in the wrong order once it passes ten entries."""
    live = _live(tmp_path)
    for i in range(12):
        mt_versions.snapshot(live, f"stage_{i}")
    names = sorted(p.name for p in mt_versions.versions_dir(live).iterdir()
                   if p.name != mt_versions.INDEX_NAME)
    # Lexicographic order == chronological order.
    assert ".v000." in names[0]
    assert ".v011." in names[-1]
    hist = mt_versions.history(live)
    assert [p.name for p in hist] == names


def test_a_snapshot_never_overwrites_an_earlier_one(tmp_path):
    """Same producer twice -- a re-run, or a replay -- must not clobber."""
    live = _live(tmp_path)
    a = mt_versions.snapshot(live, "condense")
    b = mt_versions.snapshot(live, "condense")
    assert a != b and a.exists() and b.exists()


def test_mt_is_a_reserved_producer(tmp_path):
    """`...v000.mt.json` would end in the live suffix."""
    live = _live(tmp_path)
    assert mt_versions.snapshot(live, "mt") is None


@pytest.mark.parametrize("bad", ["", "has space", "UPPER", "a" * 40, "dots.x"])
def test_a_malformed_producer_is_refused(tmp_path, bad):
    assert mt_versions.snapshot(_live(tmp_path), bad) is None


# --- content ---------------------------------------------------------------

def test_the_snapshot_is_the_document(tmp_path):
    live = _live(tmp_path, n=3)
    dest = mt_versions.snapshot(live, "original")
    assert json.loads(dest.read_text()) == json.loads(live.read_text())


def test_in_memory_data_wins_over_whatever_is_on_disk(tmp_path):
    """A caller that already has the document snapshots exactly what it
    wrote, not whatever a concurrent stage left on the path."""
    live = _live(tmp_path)
    dest = mt_versions.snapshot(live, "condense",
                                data={"segments": [{"id": "SEG_0000",
                                                    "mt_text": "shorter"}]})
    assert json.loads(dest.read_text())["segments"][0]["mt_text"] == "shorter"


def test_the_index_lists_every_version(tmp_path):
    live = _live(tmp_path, n=4)
    mt_versions.snapshot(live, "original")
    mt_versions.snapshot(live, "condense")
    rows = json.loads(
        (mt_versions.versions_dir(live) / mt_versions.INDEX_NAME).read_text())
    assert [r["producer"] for r in rows] == ["original", "condense"]
    assert all(r["segments"] == 4 for r in rows)


def test_a_corrupt_index_does_not_stop_the_next_snapshot(tmp_path):
    """The version FILES are the record; the index is a convenience."""
    live = _live(tmp_path)
    mt_versions.snapshot(live, "original")
    (mt_versions.versions_dir(live) / mt_versions.INDEX_NAME).write_text("{{{")
    assert mt_versions.snapshot(live, "condense") is not None


# --- it must never break a dub ---------------------------------------------

def test_an_unwritable_destination_is_not_an_error(tmp_path):
    live = _live(tmp_path)
    assert mt_versions.snapshot(Path("/nonexistent/x.mt.json"), "original") is None


def test_every_mutating_stage_snapshots():
    """Each stage that rewrites the merged JSON in place must record it.

    A stage added later that writes the document and does NOT snapshot
    silently loses a version, and nothing fails — which is why this is
    grepped rather than left to review.
    """
    expected = {
        "backend/pipeline.py": "original",
        "backend/codemix.py": "codemix",
        "backend/condense.py": "condense",
        "backend/emotion.py": "emotion",
        "backend/tts.py": "tts",
        "backend/edit_agent.py": "edit_",
        # The `warped` producer writes `tts_warped_audio` onto every segment.
        "backend/warp_plan.py": "warp",
    }
    for rel, producer in expected.items():
        src = (REPO / rel).read_text()
        assert "mt_versions" in src, f"{rel} does not snapshot"
        assert producer in src, f"{rel} does not snapshot as {producer!r}"
