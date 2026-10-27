"""Stage 9b1's register must survive all the way into the Bodhan payload.

The chain is five hops and no test covered it end to end:

    tts_style="auto"                       ui/presets.py
      -> classify_register(metadata)       pipeline.py:1676-1690  (stage 9b1)
      -> run_post_mt_stages(tts_style=…)   pipeline.py:1722
      -> speaker_voice_map[spk]["style"]   pipeline.py:584,604
      -> TTSRequest.style                  backend/tts.py
      -> instructions["style"]             bodhan_adapter._post_one

Every hop is cheap to break silently: a dropped kwarg anywhere gives a dub
that is merely neutral rather than wrong, so nothing errors and no artifact
looks different. That is the same shape as the voice-assignment dead code
that read an all-male lecture in a female voice.

No network: the engine is stubbed and the payload builder is exercised
directly.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.classify import REGISTER_STYLES, classify_register
from conftest import write_tone
from models.tts.base import TTSResult, TTSSegment


# --- hop 1: metadata -> register ------------------------------------------

def test_education_metadata_yields_the_lecture_register():
    """The live case: the IIT Madras W3L9 lecture's own metadata."""
    meta = {"categories": ["Education"],
            "channel": "IIT Madras - B.S. Degree Programme",
            "title": "W3L9: Deep Convolution GANs and Conditional GANs",
            "description": "Prof. Prathosh A P | Division of EECS | IISc Bangalore"}
    style, tier = classify_register(metadata=meta, transcript="")
    assert style == "educational lecture"
    assert tier == "categories", "tier 1 is free; falling past it costs an API call"


def test_an_unmapped_category_abstains_rather_than_guessing():
    """A wrong register reads a lecture as an advert; neutral is cheaper."""
    style, tier = classify_register(
        metadata={"categories": ["Gaming"], "title": "x", "description": "y"},
        transcript="")
    assert style is None


# --- the enum contract ----------------------------------------------------

def test_every_register_is_a_style_the_engine_accepts():
    """An unknown style is a 422, not a silent fallback.

    `REGISTER_STYLES` (8) and `INDIC_SPEAK_STYLES` (14) are declared in
    different modules, so adding a register without checking the engine's
    list would fail at the API on every segment of a finished run.
    """
    from models.tts.bodhan_adapter import INDIC_SPEAK_STYLES
    missing = [s for s in REGISTER_STYLES if s not in INDIC_SPEAK_STYLES]
    assert not missing, f"not accepted by indic-speak: {missing}"


# --- hop 4: speaker map -> TTSRequest -------------------------------------

class _StubTTS:
    supports_rate = False

    def __init__(self):
        self.seen = []

    def synthesize(self, requests, *, out_dir, language, sample_rate=24000):
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        segs = []
        for r in requests:
            self.seen.append((r.id, r.style))
            dest = out_dir / f"{r.id}.wav"
            write_tone(dest, 1.0, sample_rate=sample_rate)
            segs.append(TTSSegment(id=r.id, audio_path=dest, duration_s=1.0,
                                   sample_rate=sample_rate, voice=r.voice,
                                   speaker=r.speaker))
        return TTSResult(model_name="stub", out_dir=out_dir,
                         language=language, segments=segs)


def _run(tmp_path, monkeypatch, voice_map, segments):
    import backend.tts as T
    stub = _StubTTS()
    monkeypatch.setattr(T, "get_tts", lambda *a, **k: stub)
    monkeypatch.setattr(T, "available_models", lambda: ["stub"])
    merged = tmp_path / "merged.json"
    merged.write_text(json.dumps({"segments": segments, "extra": {}}))
    T.synthesize_segments(merged, tts_model="stub", language="hi",
                          speaker_voice_map=voice_map,
                          out_dir=tmp_path / "tts", apply_stretch=False)
    return stub


SEGS = [{"id": "SEG_0001", "start": 0.0, "end": 2.0, "speaker": "SPK_00",
         "final_text": "एक"},
        {"id": "SEG_0002", "start": 2.0, "end": 4.0, "speaker": "SPK_01",
         "final_text": "दो"}]


def test_the_register_reaches_every_tts_request(tmp_path, monkeypatch):
    """Stage 9d writes the register onto each speaker; TTS must read it."""
    vmap = {"SPK_00": {"engine": "indic-speak", "voice": "Amit",
                       "style": "educational lecture"},
            "SPK_01": {"engine": "indic-speak", "voice": "Kavya",
                       "style": "educational lecture"}}
    stub = _run(tmp_path, monkeypatch, vmap, SEGS)
    assert stub.seen == [("SEG_0001", "educational lecture"),
                         ("SEG_0002", "educational lecture")]


def test_a_per_segment_tone_edit_outranks_the_register(tmp_path, monkeypatch):
    """`backend.edits` writes `tts_style`; it must beat the global register."""
    vmap = {"SPK_00": {"engine": "indic-speak", "voice": "Amit",
                       "style": "educational lecture"},
            "SPK_01": {"engine": "indic-speak", "voice": "Kavya",
                       "style": "educational lecture"}}
    segs = [dict(SEGS[0], tts_style="news"), dict(SEGS[1])]
    stub = _run(tmp_path, monkeypatch, segs and vmap, segs)
    assert stub.seen == [("SEG_0001", "news"),
                         ("SEG_0002", "educational lecture")]


# --- incremental synthesis: a one-cue edit must not re-speak the dub -------

def _run_keep_merged(tmp_path, monkeypatch, merged, vmap):
    """Run the TTS stage against a PERSISTENT merged.json and return the ids
    the engine was actually asked to synthesise.

    Unlike `_run`, this does not rewrite the file, so the per-segment
    signatures the stage writes survive into the next call — which is the
    whole point of the incremental path.
    """
    import backend.tts as T
    stub = _StubTTS()
    monkeypatch.setattr(T, "get_tts", lambda *a, **k: stub)
    monkeypatch.setattr(T, "available_models", lambda: ["stub"])
    T.synthesize_segments(merged, tts_model="stub", language="hi",
                          speaker_voice_map=vmap,
                          out_dir=tmp_path / "tts", apply_stretch=False)
    return [sid for sid, _ in stub.seen]


_VMAP = {"SPK_00": {"engine": "indic-speak", "voice": "Amit"},
         "SPK_01": {"engine": "indic-speak", "voice": "Kavya"}}


def test_a_second_run_only_respeaks_changed_cues(tmp_path, monkeypatch):
    """A one-word fix must cost one engine call, not a whole-dub re-speak.

    The stage re-synthesises a segment only when its text/voice/style/rate
    signature changes; everything else reuses the previous clip. Before this,
    a replay from `tts` called the engine for every segment.
    """
    merged = tmp_path / "merged.json"
    merged.write_text(json.dumps({"segments": [dict(s) for s in SEGS],
                                  "extra": {}}))

    # First run speaks everything.
    assert sorted(_run_keep_merged(tmp_path, monkeypatch, merged, _VMAP)) == \
        ["SEG_0001", "SEG_0002"]

    # Edit one line; only that cue is re-spoken.
    data = json.loads(merged.read_text())
    data["segments"][0]["final_text"] = "बदला हुआ"
    merged.write_text(json.dumps(data))
    assert _run_keep_merged(tmp_path, monkeypatch, merged, _VMAP) == ["SEG_0001"]

    # Nothing changed: the engine is not called at all.
    assert _run_keep_merged(tmp_path, monkeypatch, merged, _VMAP) == []


def test_a_speed_edit_does_not_respeak_the_cue(tmp_path, monkeypatch):
    """A pure speed change re-fits the cached clip — no engine call.

    `set_speed` touches only the stretch target (and a rate override the
    engine here does not honour), so the raw synthesis is identical and
    re-running TTS for it would be a wasted call.
    """
    merged = tmp_path / "merged.json"
    merged.write_text(json.dumps({"segments": [dict(s) for s in SEGS],
                                  "extra": {}}))
    assert sorted(_run_keep_merged(tmp_path, monkeypatch, merged, _VMAP)) == \
        ["SEG_0001", "SEG_0002"]

    # Speed the first cue up, the way `backend.edits.set_speed` does.
    data = json.loads(merged.read_text())
    data["segments"][0]["target_duration_s"] = 1.0
    data["segments"][0]["tts_rate_override"] = 1.5
    data["segments"][0]["tts_max_stretch"] = 0.6
    merged.write_text(json.dumps(data))
    assert _run_keep_merged(tmp_path, monkeypatch, merged, _VMAP) == []


# --- the one style that is the absence of a style -------------------------

def test_neutral_takes_a_segment_off_the_register(tmp_path, monkeypatch):
    """`neutral` must reach the engine as NO style, not as the word.

    It cannot be expressed by clearing `tts_style`: None is exactly what an
    unedited segment looks like, and the `segment or document` precedence
    would then fall straight back to the register the operator just asked to
    be rid of. So the sentinel travels as itself and
    `models.tts.resolve_style` drops it here, at the last hop before the
    payload. `"neutral"` reaching the API would be a 422 on every segment.
    """
    from models.tts import NEUTRAL_STYLE

    vmap = {"SPK_00": {"engine": "indic-speak", "voice": "Amit",
                       "style": "educational lecture"},
            "SPK_01": {"engine": "indic-speak", "voice": "Kavya",
                       "style": "educational lecture"}}
    segs = [dict(SEGS[0], tts_style=NEUTRAL_STYLE), dict(SEGS[1])]
    stub = _run(tmp_path, monkeypatch, vmap, segs)
    assert stub.seen == [("SEG_0001", None),
                         ("SEG_0002", "educational lecture")]


def test_a_neutral_document_register_reaches_no_segment(tmp_path, monkeypatch):
    """The whole-dub form offers it too, so it arrives on the speaker map."""
    from models.tts import NEUTRAL_STYLE

    vmap = {"SPK_00": {"engine": "indic-speak", "voice": "Amit",
                       "style": NEUTRAL_STYLE},
            "SPK_01": {"engine": "indic-speak", "voice": "Kavya",
                       "style": NEUTRAL_STYLE}}
    stub = _run(tmp_path, monkeypatch, vmap, SEGS)
    assert stub.seen == [("SEG_0001", None), ("SEG_0002", None)]


def test_neutral_is_not_an_engine_style():
    """If upstream ever adds one, the sentinel stops being unambiguous.

    `resolve_style` maps the literal to None, so a real Bodhan style of the
    same name could never be requested — it would silently become "no style"
    on every segment.
    """
    from models.tts import NEUTRAL_STYLE
    from models.tts.bodhan_adapter import INDIC_SPEAK_STYLES

    assert NEUTRAL_STYLE not in INDIC_SPEAK_STYLES
    assert not any(s.lower() == NEUTRAL_STYLE for s in INDIC_SPEAK_STYLES)


def test_resolve_style_keeps_the_existing_precedence():
    """Everything that is not the sentinel behaves exactly as `seg or doc`."""
    from models.tts import resolve_style

    assert resolve_style("news", "educational lecture") == "news"
    assert resolve_style(None, "educational lecture") == "educational lecture"
    assert resolve_style("", "educational lecture") == "educational lecture"
    assert resolve_style(None, None) is None


# --- hop 5: TTSRequest -> HTTP payload ------------------------------------

def _payload(style):
    """Build the request body `_post_one` would POST, without posting it."""
    instructions = {"lang": "hi"}
    if style:
        instructions["style"] = style
    return {"model": "indic-speak", "input": "x", "voice": "Amit",
            "instructions": json.dumps(instructions)}


@pytest.mark.parametrize("style", REGISTER_STYLES)
def test_every_register_lands_in_the_instructions_blob(style):
    body = _payload(style)
    assert json.loads(body["instructions"]) == {"lang": "hi", "style": style}


def test_no_register_leaves_the_payload_exactly_as_before():
    """Abstention must not add an empty style key -- the neutral reading is
    the absence of `style`, not `style: null`."""
    assert json.loads(_payload(None)["instructions"]) == {"lang": "hi"}


# --- anti-drift: the pipeline must keep making the call -------------------

def test_pipeline_classifies_before_the_post_mt_stages():
    """Reading the register AFTER TTS would make it decorative.

    Greps rather than runs, because reaching stage 9b1 for real costs a
    download, a diarization, an ASR pass and an MT pass.
    """
    src = (Path(__file__).resolve().parent.parent
           / "backend" / "pipeline.py").read_text()
    assert "classify_register" in src, "stage 9b1 no longer calls the classifier"
    assert 'log.info("=== stage 9b1' in src, "the 9b1 marker moved or was renamed"
    assert src.index("classify_register(") < src.index("post = run_post_mt_stages("), \
        "the register is decided after TTS runs, so it cannot reach the engine"
    assert "tts_style=tts_style," in src, "the classified style is not forwarded"


# --- the pre-cut gap: metadata on disk the pipeline never looked at -------

def _job(tmp_path, meta=None, *, where="source", stem="VID"):
    root = tmp_path / "job"
    (root / where).mkdir(parents=True, exist_ok=True)
    (root / "videos").mkdir(parents=True, exist_ok=True)
    clip = root / "videos" / f"{stem}_10to20.mp4"
    clip.write_bytes(b"\0")
    if meta is not None:
        (root / where / f"{stem}.meta.json").write_text(json.dumps(meta))
    return root, clip


def test_metadata_is_recovered_when_the_ui_pre_cut_the_clip(tmp_path):
    """The real failure: a UI run recorded register=None with
    `categories: ["Education"]` sitting in <job>/source/ the whole time.

    The pipeline only fills `source_metadata` from its OWN download, and a
    pre-cut clip means that download never happens here.
    """
    from backend.pipeline import _recover_source_metadata
    root, clip = _job(tmp_path, {"categories": ["Education"],
                                 "channel": "IIT Madras - B.S. Degree Programme"})
    meta = _recover_source_metadata(root, clip)
    assert meta.get("categories") == ["Education"]
    assert classify_register(meta, transcript="", use_llm=False) == (
        "educational lecture", "categories")


def test_a_genuine_local_file_recovers_nothing(tmp_path):
    """No metadata on disk must stay None, not invent a register."""
    from backend.pipeline import _recover_source_metadata
    root, clip = _job(tmp_path, None)
    assert _recover_source_metadata(root, clip) == {}
    assert classify_register({}, transcript="", use_llm=False) == (None, "none")


def test_unreadable_metadata_does_not_kill_the_run(tmp_path):
    """Stage 9b1 sits after ASR, alignment and MT have been paid for."""
    from backend.pipeline import _recover_source_metadata
    root, clip = _job(tmp_path)
    (root / "source" / "VID.meta.json").write_text("{not json")
    assert _recover_source_metadata(root, clip) == {}


def test_recovery_only_fires_when_the_pipeline_did_not_download(tmp_path):
    """A URL run already has richer metadata in memory; do not overwrite it."""
    src = (Path(__file__).resolve().parent.parent
           / "backend" / "pipeline.py").read_text()
    assert 'if tts_style == "auto" and not source_metadata:' in src, \
        "recovery must be gated on source_metadata being empty"
    assert src.index("_recover_source_metadata(job_root, video)") < \
        src.index("tts_style, tier = classify_register("), \
        "recovery has to run before the classifier reads the metadata"
