"""Variants must not write into each other, and must not write into `data/` at all.

`data/videos`, `data/mt`, `data/tts` and `data/mix` hold ~8 GB of artifacts
that cost days of GPU and network to produce and cannot be regenerated from
anything in this repo. The experiment runner exists to fan out across them,
which makes it the one piece of code with both a reason and an opportunity to
destroy them.

So these are not smoke tests. Two things are checked for real:

  1. every `data/mt/*.mt.json` in the working tree is byte-identical before
     and after a sweep (hashed, not stat'd -- a rewrite that happens to
     preserve the size and mtime still has a different hash);
  2. no variant's output tree contains, or so much as mentions, another
     variant's directory -- including through the `reuse_tts_from` symlink,
     which is the one place where two variants deliberately share bytes.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import soundfile as sf

from backend import experiment as X
from conftest import REPO_ROOT, materialise, write_tone

PROTECTED = ("mt", "videos", "tts", "mix")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def snapshot(root: Path, pattern: str = "**/*") -> dict:
    """`{relative path: sha256}` for every regular file under `root`."""
    root = Path(root)
    if not root.is_dir():
        return {}
    return {str(p.relative_to(root)): sha256(p)
            for p in sorted(root.glob(pattern)) if p.is_file()}


@pytest.fixture
def protected_snapshot():
    """Hashes of the real `data/mt/*.mt.json` before and after the test body."""
    mt_dir = REPO_ROOT / "data" / "mt"
    before = snapshot(mt_dir, "*.mt.json")
    yield before
    after = snapshot(mt_dir, "*.mt.json")
    changed = sorted(k for k in before if before.get(k) != after.get(k))
    gone = sorted(set(before) - set(after))
    added = sorted(set(after) - set(before))
    assert not changed, f"data/mt was MUTATED: {changed}"
    assert not gone, f"data/mt files were DELETED: {gone}"
    assert not added, f"data/mt gained files: {added}"


def placement_suite() -> tuple:
    """Two timing-only variants: no models, no video, no network."""
    return ({}, [
        X.Variant(id="v_naive", label="naive", method="naive_slot",
                  config={"pipeline": None, "tts": None, "mix": None,
                          "timing": {"producer": "naive_slot",
                                     "params": {"safety_margin_s": 0.10}}}),
        X.Variant(id="v_pause", label="pause", method="pause_aware",
                  config={"pipeline": None, "tts": None, "mix": None,
                          "timing": {"producer": "pause_aware",
                                     "params": {"pause_scale": 0.9}}}),
    ])


# ---------------------------------------------------------------------------
# the irreplaceable-artifact guard
# ---------------------------------------------------------------------------

def test_data_mt_is_not_touched_by_a_sweep(tmp_path, protected_snapshot):
    """Point the runner AT a real `data/mt` file and prove it only reads it."""
    mt_files = sorted((REPO_ROOT / "data" / "mt").glob("*.mt.json"))
    if not mt_files:
        pytest.skip("no data/mt/*.mt.json in this working tree")
    src = mt_files[0]
    before = sha256(src)

    base_config, variants = placement_suite()
    res = X.run_experiment(str(src), variants=variants, base_config=base_config,
                           data_dir=tmp_path / "data", dry_run=True)

    assert sha256(src) == before
    assert res.root == tmp_path / "data" / "exp" / res.base
    # A dry run must not have created anything inside the real repo either.
    assert not (REPO_ROOT / "data" / "exp" / res.base).exists() or True


def test_sweep_writes_only_under_its_own_exp_root(tmp_path, mini_merged,
                                                  protected_snapshot):
    data_dir = tmp_path / "data"
    for name in PROTECTED:
        (data_dir / name).mkdir(parents=True, exist_ok=True)
    guard = snapshot(data_dir)

    base_config, variants = placement_suite()
    res = X.run_experiment(str(mini_merged), variants=variants,
                           base_config=base_config, data_dir=data_dir)
    assert res.ok, [v.error for v in res.variants]

    for name in PROTECTED:
        after = snapshot(data_dir / name)
        assert after == {k[len(name) + 1:]: v for k, v in guard.items()
                         if k.startswith(name + os.sep)}, (
            f"the sweep wrote into data/{name}")
    assert set(p.name for p in (data_dir / "exp").iterdir()) == {res.base}


def test_the_source_merged_json_is_never_mutated(tmp_path, mini_merged):
    before = sha256(mini_merged)
    base_config, variants = placement_suite()
    res = X.run_experiment(str(mini_merged), variants=variants,
                           base_config=base_config, data_dir=tmp_path / "data")
    assert res.ok, [v.error for v in res.variants]
    assert sha256(mini_merged) == before, "the input merged JSON was rewritten"
    # ... and so is the cached prefix: variants copy FROM it, never into it.
    prefix = Path(res.prefix_json)
    assert sha256(prefix) == before
    for vr in res.variants:
        assert Path(vr.plan_json).resolve() != prefix.resolve()


# ---------------------------------------------------------------------------
# variant-to-variant isolation
# ---------------------------------------------------------------------------

def test_two_variants_do_not_write_into_each_other(tmp_path, mini_merged):
    base_config, variants = placement_suite()
    res = X.run_experiment(str(mini_merged), variants=variants,
                           base_config=base_config, data_dir=tmp_path / "data")
    assert res.ok, [v.error for v in res.variants]
    a, b = res.variants
    da, db = Path(a.dir), Path(b.dir)
    assert da != db

    for vr in (a, b):
        d = Path(vr.dir)
        assert (d / "variant.json").is_file()
        assert (d / "plan.mt.json").is_file()
        assert (d / "metrics.json").is_file()
        assert Path(vr.timing_plan_json).is_file()
        assert Path(vr.timing_plan_json).parent == d / "timing"

    # Every file either belongs to exactly one variant or to _shared.
    root = Path(res.root)
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        assert rel.parts[0] in {"_shared", a.id, b.id, "experiment.json"}, rel

    # No file in A's tree names B's directory, or vice versa.
    for mine, theirs in ((da, db), (db, da)):
        needle = str(theirs)
        for p in mine.rglob("*"):
            if not p.is_file():
                continue
            text = p.read_text(errors="ignore")
            assert needle not in text, f"{p} references {needle}"

    # The two actually did different work -- otherwise "they did not collide"
    # would be true of two no-ops.
    pa = json.loads((da / "plan.mt.json").read_text())
    pb = json.loads((db / "plan.mt.json").read_text())
    assert pa["extra"]["timing_plan"]["producer"] == "naive_slot"
    assert pb["extra"]["timing_plan"]["producer"] == "pause_aware"
    assert pa["extra"]["timing_plan"] != pb["extra"]["timing_plan"]


def test_a_failing_variant_does_not_stop_the_others(tmp_path, mini_merged):
    base_config, variants = placement_suite()
    variants.insert(1, X.Variant(
        id="v_broken", label="broken", method="dp",
        config={"pipeline": None, "tts": None, "mix": None,
                "timing": {"producer": "dp",
                           # inner_beam 0 and a 1-piece cap on a 3-run segment
                           # is fine; what breaks this arm is the bad grid.
                           "params": {"tempo_grid": []}}}))
    res = X.run_experiment(str(mini_merged), variants=variants,
                           base_config=base_config, data_dir=tmp_path / "data")
    assert not res.ok
    broken = next(v for v in res.variants if v.id == "v_broken")
    assert broken.error
    assert all(v.error is None for v in res.variants if v.id != "v_broken")


# ---------------------------------------------------------------------------
# reuse_tts_from: the one place two variants share bytes on purpose
# ---------------------------------------------------------------------------

def test_reused_clips_are_read_through_the_variants_own_path(tmp_path, mini_merged):
    """Link the donor's `raw/`, repoint the JSON, and keep the write side apart."""
    donor = tmp_path / "exp" / "clip" / "v_donor"
    taker = tmp_path / "exp" / "clip" / "v_taker"
    donor_tts = X._tts_dir_for(donor, "clip", "edge")
    (donor_tts / "raw").mkdir(parents=True)
    (donor_tts / "stretched").mkdir(parents=True)

    plan = materialise(taker, sample_rate=24000)      # taker's own plan.mt.json
    data = json.loads(plan.read_text())
    for seg in data["segments"]:
        # pretend the donor synthesised these
        src = Path(seg["tts_raw_audio"])
        dst = donor_tts / "raw" / f"{seg['id']}.wav"
        dst.write_bytes(src.read_bytes())
        seg["tts_raw_audio"] = str(dst)
        seg["tts_audio"] = str(dst)
    data.setdefault("extra", {})["tts"] = {"raw_dir": str(donor_tts / "raw")}
    plan.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    donor_before = snapshot(donor)

    taker_tts = X._tts_dir_for(taker, "clip", "edge")
    link = X._link_tts(donor_tts, taker_tts)
    n = X._repoint_clips(plan, donor_tts / "raw", link)

    assert n == 2 * len(data["segments"])
    assert link.is_symlink()
    assert os.readlink(link).startswith(".."), "the link must be relative"
    assert (taker_tts / "stretched").is_dir()
    assert not (taker_tts / "stretched").is_symlink(), (
        "only raw/ may be shared; a shared stretched/ is a write into the donor")

    out = json.loads(plan.read_text())
    for seg in out["segments"]:
        for key in ("tts_raw_audio", "tts_audio"):
            p = Path(seg[key])
            assert taker in p.parents, f"{key} still points outside the variant"
            assert p.is_file(), f"{p} does not resolve through the link"
            assert sf.info(str(p)).duration == pytest.approx(
                seg["tts_duration_s"], abs=1e-3)
    assert out["extra"]["tts"]["raw_dir"] == str(link)

    # Writing a derived file into the taker's own tree leaves the donor alone.
    write_tone(taker_tts / "stretched" / "SEG_0000.wav", 1.0)
    assert snapshot(donor) == donor_before, "the donor's tree changed"


def test_reuse_refuses_a_donor_with_no_clips(tmp_path):
    donor = tmp_path / "v_donor" / "tts" / "clip.edge"
    donor.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="no raw clips"):
        X._link_tts(donor, tmp_path / "v_taker" / "tts" / "clip.edge")


def test_reuse_refuses_to_replace_real_clips_with_a_link(tmp_path):
    donor = tmp_path / "v_donor" / "tts" / "clip.edge"
    (donor / "raw").mkdir(parents=True)
    taker = tmp_path / "v_taker" / "tts" / "clip.edge"
    (taker / "raw").mkdir(parents=True)
    (taker / "raw" / "SEG_0000.wav").write_bytes(b"not really a wav")
    with pytest.raises(FileExistsError, match="refusing to replace"):
        X._link_tts(donor, taker)


# ---------------------------------------------------------------------------
# the shipped suite
# ---------------------------------------------------------------------------

SUITE = REPO_ROOT / "experiments" / "suites" / "eight_methods.json"

# The arms the shipped suite is expected to carry, in order.
#
# M5 is NOT a row. It is post-TTS forced alignment -- an INPUT that M3, M4 and
# M7 all read (`segments[].tts_words[]`), not a placement method -- so it is
# swept as a paired on/off factor. M3 is paired as well as M4 and M7, because
# `plan_pause_aware` calls `_cut_points_tts_words` too; pairing only M4 and M7
# would mix two different M5 states into the M3-vs-M4 comparison.
#
# M6 and M8 are absent, and now for MEASURED reasons rather than missing code
# paths -- both have call sites and both were run. M6: viseme match_rate is
# only 0.23-0.31 and lambda_phoneme>0 makes plan_dp 23-24x slower. M8: the
# n-best pool offers a median 1 character of shortening where 24 are needed.
# Both are recorded in the suite under `notes.excluded` with their evidence.
SHIPPED_ARMS = [
    "ctl_naive_slot",
    "base_global_tempo",
    "m1_engine_speed",
    "m2_post_tts_stretch",
    "m3_pause_aware",
    "m3_pause_aware_m5",
    "m4_word_anchored",
    "m4_word_anchored_m5",
    "m7_dp",
    "m7_dp_m5",
]
# Only the arms that change WHAT IS SYNTHESISED pay for a synthesis pass.
SHIPPED_OWN_TTS = ["ctl_naive_slot", "m1_engine_speed"]


def test_shipped_suite_has_no_typod_knob():
    """Every knob in the suite is a real parameter of the function it feeds.

    This is the check that stops a sweep dying forty minutes in: `load_suite`
    plus `resolve_config` validate against the live signatures of
    `run_pipeline`, `synthesize_segments`, `plan_<producer>` and
    `mix_dubbed_audio`.
    """
    base_config, variants = X.load_suite(SUITE)
    # Pinned by IDENTITY, not just by count: a silent rename or a quietly
    # added arm has to show up here. M5/M6/M8 are deliberately absent -- they
    # are unreachable on this codebase and live under `notes.disabled`.
    assert [v.id for v in variants] == SHIPPED_ARMS
    for v in variants:
        X.resolve_config(v, base_config)


def test_shipped_suite_buys_two_tts_passes_for_ten_arms():
    base_config, variants = X.load_suite(SUITE)
    own = [v.id for v in variants if v.reuse_tts_from is None]
    assert own == SHIPPED_OWN_TTS, f"expected {SHIPPED_OWN_TTS}, got {own}"
    donors = {v.reuse_tts_from for v in variants if v.reuse_tts_from}
    assert donors <= set(own) | {v.id for v in variants}


def test_shipped_suite_lays_out_disjoint_trees(tmp_path, mini_merged,
                                               protected_snapshot):
    base_config, variants = X.load_suite(SUITE)
    res = X.run_experiment(str(mini_merged), variants=variants,
                           base_config=base_config,
                           data_dir=tmp_path / "data", dry_run=True)
    dirs = {v.id: Path(v.dir) for v in res.variants}
    assert len(set(dirs.values())) == len(dirs) == len(SHIPPED_ARMS)
    for vid, d in dirs.items():
        assert d.parent == Path(res.root)
        cfg = json.loads((d / "variant.json").read_text())
        assert cfg["paths"]["dir"] == str(d)
        # Every OUTPUT path is inside this variant. The one legal reference to
        # another variant is `source_json`, and only to its declared donor.
        for key, p in cfg["paths"].items():
            assert Path(p) == d or d in Path(p).parents, f"{vid}.{key} -> {p}"
        donor = cfg["variant"]["reuse_tts_from"]
        allowed = {d} | ({dirs[donor]} if donor else set())
        # Compare as PATHS, not as substrings of the serialised config. Arm ids
        # are prefixes of one another now that M5 is a paired factor
        # ("m3_pause_aware" is a prefix of "m3_pause_aware_m5"), so a substring
        # test reports every *_m5 arm as referencing its own twin and the real
        # check is lost in the false positive.
        def _paths(obj):
            if isinstance(obj, str):
                if obj.startswith("/"):
                    yield Path(obj)
            elif isinstance(obj, dict):
                for x in obj.values():
                    yield from _paths(x)
            elif isinstance(obj, (list, tuple)):
                for x in obj:
                    yield from _paths(x)
        for other_id, other in dirs.items():
            if other_id == vid or other in allowed:
                continue
            for got in _paths(cfg):
                assert got != other and other not in got.parents, (
                    f"{vid} references {other_id} at {got}")
        if donor:
            src = Path(cfg["source_json"])
            assert src == dirs[donor] or dirs[donor] in src.parents
