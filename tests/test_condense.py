"""Stage 9c2: shortening a translation that will not fit its slot.

The case it exists for, measured 2026-10-21 on svwrTOm55y4: 50.9s of Hindi
for 30.6s of slots -- 1.66x against a 1.5x tempo cap, 3 of 5 segments
clamped, dub ~8s long. Bodhan's TTS has no speed parameter (their API docs
list model/input/voice/instructions and nothing else), so the audio cannot
be shortened after it exists. The sentence has to be.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from backend import condense  # noqa: E402


class _Client:
    """Stand-in LLM. `replies` is consumed one per call.

    A reply may be the batch shape the prompt asks for
    (`{"segments": [...]}`), a bare dict (the single-segment shape a model
    drops back to), or an Exception to raise.
    """

    def __init__(self, replies, model="test/model"):
        self.replies = list(replies)
        self.model = model
        self.calls = 0
        #: Every message list this was called with, so a test can assert
        #: WHAT was asked -- which prompt, and whether the source line was
        #: actually sent -- not merely that a call happened.
        self.seen = []
        #: Every `max_tokens` it was given. The batch reply has to fit, and
        #: a truncated one is unparseable JSON that loses the whole chunk.
        self.max_tokens = []

    def available(self):
        return True

    def chat_json(self, messages, **kw):
        self.calls += 1
        self.seen.append(messages)
        self.max_tokens.append(kw.get("max_tokens"))
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _sent(client, call=0):
    """The segment rows of one call's user payload."""
    return json.loads(client.seen[call][1]["content"])["segments"]


def _batch(*rows):
    """A reply in the shape the prompt declares."""
    return {"segments": list(rows)}


def _job(tmp_path: Path, segs) -> Path:
    p = tmp_path / "x.mt.json"
    p.write_text(json.dumps({"segments": segs, "extra": {}}))
    return p


def _seg(sid, text, target, est):
    return {"id": sid, "mt_text": text, "final_text": text,
            "target_duration_s": target, "est_duration_s": est}


# --- scope ----------------------------------------------------------------

def test_only_overlong_segments_are_touched(tmp_path):
    """A line already inside its budget is left alone.

    Re-asking an LLM to rewrite text that fits spends money to risk a worse
    translation, and the risk is one-directional: the current text is known
    to work.
    """
    mt = _job(tmp_path, [
        _seg("SEG_0000", "fits fine", target=10.0, est=9.0),     # inside
        _seg("SEG_0001", "way too long a line here", 10.0, 20.0),  # over
    ])
    c = _Client([{"text": "short"}])
    out = condense.condense_overlong(mt, language="hi", client=c,
                                     cache_dir=tmp_path)
    assert out["considered"] == 1, "only the overrunning segment counts"
    assert c.calls == 1
    segs = json.loads(mt.read_text())["segments"]
    assert segs[0]["mt_text"] == "fits fine", "a fitting line was rewritten"


def test_nothing_to_do_makes_no_calls(tmp_path):
    mt = _job(tmp_path, [_seg("SEG_0000", "ok", 10.0, 9.0)])
    c = _Client([])
    out = condense.condense_overlong(mt, language="hi", client=c,
                                     cache_dir=tmp_path)
    assert out["considered"] == 0 and c.calls == 0


# --- the verification that makes it safe ----------------------------------

def test_a_rewrite_that_is_not_shorter_is_discarded(tmp_path):
    """An LLM asked for brevity will happily return something longer.

    Only measuring knows. The duration model that flagged the segment
    re-scores the rewrite, and a rewrite that does not come back shorter is
    thrown away -- otherwise "shorten this" could silently make the overrun
    worse while reporting success.
    """
    long_original = "x" * 40
    mt = _job(tmp_path, [_seg("SEG_0000", long_original, 1.0, 20.0)])
    c = _Client([{"text": "y" * 400}])          # much longer
    out = condense.condense_overlong(mt, language="hi", client=c,
                                     cache_dir=tmp_path)
    assert out["rewritten"] == 0
    assert out["rejected"] == 1
    seg = json.loads(mt.read_text())["segments"][0]
    assert seg["mt_text"] == long_original, "a longer rewrite was accepted"


def test_a_shorter_rewrite_is_kept_and_backed_up(tmp_path):
    original = "a very long winded way of saying something quite simple"
    mt = _job(tmp_path, [_seg("SEG_0000", original, 1.0, 20.0)])
    c = _Client([{"text": "briefly"}])
    out = condense.condense_overlong(mt, language="hi", client=c,
                                     cache_dir=tmp_path)
    assert out["rewritten"] == 1
    seg = json.loads(mt.read_text())["segments"][0]
    assert seg["mt_text"] == "briefly"
    # Revertible, the same way backend/edits.py keeps `edit_backup`.
    assert seg["condense_backup"]["text"] == original
    assert seg["qc"]["condensed"] is True


def test_both_text_keys_are_updated(tmp_path):
    """`tts.py` prefers `final_text` and falls back to `mt_text`. Updating
    one leaves the other stale, and whichever is read first wins -- which
    is a silent half-applied edit."""
    mt = _job(tmp_path, [_seg("SEG_0000", "long original text here", 1.0, 20.0)])
    condense.condense_overlong(mt, language="hi",
                               client=_Client([{"text": "brief"}]),
                               cache_dir=tmp_path)
    seg = json.loads(mt.read_text())["segments"][0]
    assert seg["mt_text"] == "brief" and seg["final_text"] == "brief"


# --- every failure keeps the dub ------------------------------------------

@pytest.mark.parametrize("reply", [
    RuntimeError("model is down"),
    {"nope": "wrong shape"},
    {"text": ""},
])
def test_failures_keep_the_original_text(tmp_path, reply):
    """A shortening pass is an optimisation. It must never be the reason a
    dub fails, so every way the call can go wrong falls back."""
    original = "the original wording"
    mt = _job(tmp_path, [_seg("SEG_0000", original, 1.0, 20.0)])
    out = condense.condense_overlong(mt, language="hi",
                                     client=_Client([reply]),
                                     cache_dir=tmp_path)
    assert out["rewritten"] == 0
    assert json.loads(mt.read_text())["segments"][0]["mt_text"] == original


def test_no_api_key_is_not_an_error(tmp_path):
    class _NoKey(_Client):
        def available(self):
            return False
    mt = _job(tmp_path, [_seg("SEG_0000", "long", 1.0, 20.0)])
    out = condense.condense_overlong(mt, language="hi", client=_NoKey([]),
                                     cache_dir=tmp_path)
    assert out["error"] == "no api key" and out["rewritten"] == 0


# --- the wiring -----------------------------------------------------------

def test_the_stage_marker_is_one_line_and_registered():
    """`ui/progress.py` scrapes these markers and the anti-drift test greps
    the literal, so a marker split across two source lines is captured
    without its closing `===` and silently stops matching -- which freezes
    the progress bar."""
    src = (REPO / "backend" / "pipeline.py").read_text()
    assert '"=== stage 9c2: shortening over-long lines ==="' in src, (
        "the 9c2 marker is missing or wrapped across lines")
    prog = (REPO / "ui" / "progress.py").read_text()
    assert '"9c2"' in prog, "9c2 has no user-facing label / phase group"


def test_9c2_is_planned_only_with_duration_adjust():
    """9c2 runs inside the `duration_adjust` branch, acting on the
    estimates 9c writes, so it cannot be planned without it."""
    from ui.progress import plan_stages

    # `plan_stages` returns ROWS, not keys. `"9c2" in [{...}]` is False for
    # every config, so a membership test on the list passes by accident in
    # both directions and says nothing.
    def planned(cfg):
        return any(s.get("key") == "9c2" for s in plan_stages(cfg))

    base = {"mt": "bodhan", "tts": "bodhan", "duration_adjust": True}
    assert not planned(base)
    assert planned({**base, "condense_overlong_text": True})
    assert not planned({"mt": "bodhan", "tts": "bodhan",
                        "condense_overlong_text": True})


def test_the_manifest_records_the_switch():
    """The manifest is the only durable record of how a run was configured;
    a stage that ran without being recorded is invisible to every reader."""
    src = (REPO / "backend" / "pipeline.py").read_text()
    assert '"condense_overlong_text": bool(condense_overlong_text),' in src


def test_the_cli_refuses_it_without_duration_adjust():
    """Fail at parse time, not at 9c2 after MT has been paid for."""
    src = (REPO / "backend" / "cli.py").read_text()
    assert "--condense-overlong requires --duration-adjust" in src


# --- the speed-cap gate (2026-10-22) --------------------------------------

@pytest.mark.parametrize("est,expect_call", [
    (10.0, False),   # 1.00x -- fits outright
    (14.0, False),   # 1.40x -- the stretch absorbs it
    (15.0, False),   # 1.50x -- exactly at the cap, still reachable
    (15.5, True),    # 1.55x -- past it
    (20.0, True),    # 2.00x
])
def test_only_lines_past_the_speed_cap_are_rewritten(tmp_path, est, expect_call):
    """The gate is the speed-up the MIX can apply, not a flat overrun.

    It used to fire at 1.10x, which spent a call rewriting lines `atempo`
    would have fitted perfectly well -- on the svwrTOm55y4 run that is 4 of 5
    segments rewritten where only 3 ever clamped. Rewriting text that already
    renders correctly is a translation risk taken for nothing.
    """
    mt = _job(tmp_path, [_seg("SEG_0000", "a line", target=10.0, est=est)])
    c = _Client([{"text": "short"}])
    out = condense.condense_overlong(mt, language="hi", client=c,
                                     cache_dir=tmp_path)
    assert bool(c.calls) is expect_call
    assert bool(out["considered"]) is expect_call


def test_the_cap_is_configurable_and_reported(tmp_path):
    """1.5 is a default, not a constant -- and the summary records which
    number a finished run actually used."""
    segs = [_seg("SEG_0000", "a line", target=10.0, est=17.0)]   # 1.7x
    mt = _job(tmp_path, segs)
    c = _Client([])
    out = condense.condense_overlong(mt, language="hi", client=c,
                                     cache_dir=tmp_path, speed_cap=1.8)
    assert c.calls == 0, "1.7x is inside a 1.8x cap"
    assert out["speed_cap"] == 1.8


# --- retranslate vs shorten ------------------------------------------------

def test_the_prompt_is_the_authors_reference_file(tmp_path):
    """`translate_agetn.prompt.txt` IS the prompt, not an inspiration for one.

    Two copies of a prompt in a repo drift, and the drift is invisible: the
    stage keeps answering, just not the way the file says it should.
    """
    seg = _seg("SEG_0000", "bahut lamba vaakya", target=10.0, est=20.0)
    seg["text"] = "a very long sentence"
    mt = _job(tmp_path, [seg])
    c = _Client([{"text": "chhota"}])
    condense.condense_overlong(mt, language="Hindi", client=c,
                               cache_dir=tmp_path)
    sys_prompt = c.seen[0][0]["content"]
    ref = (REPO / "translate_agetn.prompt.txt").read_text()
    assert "{{target_language}}" not in sys_prompt, "language not substituted"
    assert "Hindi" in sys_prompt
    # A distinctive line from each half of the file, so a truncated or
    # re-summarised prompt fails rather than passing on a keyword.
    assert "Rewrite `mt_text`, not `english_asr`" in sys_prompt
    assert "Never pad the output to fill the available duration." in sys_prompt
    assert len(sys_prompt) >= len(ref) - 200


def test_the_segment_is_sent_as_the_prompts_own_input_contract(tmp_path):
    """The prompt names its fields and does arithmetic on them, so the
    payload has to use those names."""
    seg = _seg("SEG_0000", "bahut lamba vaakya", target=10.0, est=20.0)
    seg["text"] = "a very long sentence"
    seg["retain_spans"] = [{"text": "GAN", "start": 0, "end": 3}]
    mt = _job(tmp_path, [seg])
    c = _Client([{"text": "chhota"}])
    condense.condense_overlong(mt, language="Hindi", client=c,
                               cache_dir=tmp_path)
    rows = _sent(c)
    # Always an ARRAY, even for one segment: a request whose shape changes
    # with the batch size is two contracts, and the rare one breaks unseen.
    assert len(rows) == 1
    payload = rows[0]
    assert payload["mt_text"] == "bahut lamba vaakya"
    assert payload["english_asr"] == "a very long sentence"
    assert payload["target_duration_s"] == 10.0
    assert payload["est_duration_s"] == 20.0
    assert payload["protected_terms"] == ["GAN"], "retain_spans -> protected"
    # 1.5x cap, NOT the prompt's default of 10. `fits` is computed from this,
    # so leaving it at 10 makes the model cut far harder than necessary.
    assert payload["stretch_cap_pct"] == 50


def test_the_models_infeasible_verdict_ends_the_rounds(tmp_path):
    """"required cut exceeds 40%" -- a second ask cannot make an
    incompressible line compressible, and the prompt says the remainder is
    the pipeline's problem."""
    mt = _job(tmp_path, [_seg("SEG_0000", "long line", target=10.0, est=30.0)])
    c = _Client([{"text": "shorter", "infeasible": True}])
    out = condense.condense_overlong(mt, language="Hindi", client=c,
                                     cache_dir=tmp_path, max_rounds=3)
    assert c.calls == 1, "it must not keep asking"
    assert out["infeasible"] == 1


def test_qc_fields_are_recorded_but_do_not_decide(tmp_path):
    """The prompt calls its own duration a proxy. `duration.predict` is the
    arbiter; the model's metrics are logged for review."""
    mt = _job(tmp_path, [_seg("SEG_0000", "long line here", 10.0, 20.0)])
    # Claims it fits and is adequate -- but it MEASURES longer than the
    # 20.0s the segment declares, so the measurement must still reject it.
    # The comparison is against `est_duration_s`, not against the original
    # string: a handful of extra words would not have been enough.
    c = _Client([{"text": "long line here " * 25, "fits": True,
                  "adequacy": "ok", "meaning_loss": "none"}])
    out = condense.condense_overlong(mt, language="Hindi", client=c,
                                     cache_dir=tmp_path)
    assert out["rejected"] == 1 and out["rewritten"] == 0
    segs = json.loads(mt.read_text())["segments"]
    assert segs[0]["mt_text"] == "long line here", "a longer 'fit' was taken"


def test_a_major_issue_is_surfaced(tmp_path):
    """A translation problem the length pass noticed and could not fix is
    the interesting signal, and it would otherwise vanish."""
    mt = _job(tmp_path, [_seg("SEG_0000", "line", target=10.0, est=20.0)])
    c = _Client([{"text": "s", "adequacy": "major_issue",
                  "adequacy_issues": ["term_violation: GAN dropped"]}])
    out = condense.condense_overlong(mt, language="Hindi", client=c,
                                     cache_dir=tmp_path)
    assert out["flagged"][0]["adequacy"] == "major_issue"


def test_the_summary_reports_rate_not_just_seconds(tmp_path):
    """"1.66x against a 1.5x cap" is the readable form of the problem;
    "20.0s into 12.0s" makes the reader do the division."""
    mt = _job(tmp_path, [_seg("SEG_0000", "long", target=10.0, est=20.0)])
    c = _Client([{"text": "s"}])
    out = condense.condense_overlong(mt, language="hi", client=c,
                                     cache_dir=tmp_path)
    row = out["segments"][0]
    assert row["rate_before"] == 2.0
    assert "rate_after" in row


# --- batching (2026-10-23) -------------------------------------------------

def _overlong(n, start=0):
    """`n` segments that all need more than the 1.5x cap."""
    return [_seg(f"SEG_{i:04d}", f"line {i} is far too long to fit",
                 target=10.0, est=20.0) for i in range(start, start + n)]


def test_segments_are_batched_and_the_prompt_is_sent_once_per_call(tmp_path):
    """The system prompt is ~2.9k tokens and used to ride on every segment.

    Five over-long lines at a batch of 2 is three calls, not five -- and
    the saving is the prompt, not the segments.
    """
    mt = _job(tmp_path, _overlong(5))
    c = _Client([
        _batch({"segment_id": "SEG_0000", "text": "a"},
               {"segment_id": "SEG_0001", "text": "b"}),
        _batch({"segment_id": "SEG_0002", "text": "c"},
               {"segment_id": "SEG_0003", "text": "d"}),
        _batch({"segment_id": "SEG_0004", "text": "e"}),
    ])
    out = condense.condense_overlong(mt, language="hi", client=c,
                                     cache_dir=tmp_path, batch_size=2)
    assert out["considered"] == 5
    assert c.calls == 3, "2 + 2 + 1"
    assert out["calls"] == 3 and out["batch_size"] == 2
    assert out["rewritten"] == 5
    assert [len(_sent(c, i)) for i in range(3)] == [2, 2, 1]


def test_a_single_overlong_segment_is_still_one_call(tmp_path):
    """`min(batch_size, overlong)` -- the common case is unchanged."""
    mt = _job(tmp_path, _overlong(1))
    c = _Client([_batch({"segment_id": "SEG_0000", "text": "s"})])
    condense.condense_overlong(mt, language="hi", client=c,
                               cache_dir=tmp_path, batch_size=5)
    assert c.calls == 1 and len(_sent(c)) == 1


def test_batch_size_one_is_the_pre_batch_behaviour(tmp_path):
    mt = _job(tmp_path, _overlong(3))
    c = _Client([_batch({"segment_id": f"SEG_{i:04d}", "text": "s"})
                 for i in range(3)])
    condense.condense_overlong(mt, language="hi", client=c,
                               cache_dir=tmp_path, batch_size=1)
    assert c.calls == 3
    assert all(len(_sent(c, i)) == 1 for i in range(3))


def test_the_token_cap_grows_with_the_batch(tmp_path):
    """The client's 2048 default truncates a multi-segment reply into
    unparseable JSON, which loses the whole chunk. Devanagari costs 2-3
    tokens per akshara, so this is not theoretical."""
    mt = _job(tmp_path, _overlong(4))
    c = _Client([_batch(*({"segment_id": f"SEG_{i:04d}", "text": "s"}
                          for i in range(4)))])
    condense.condense_overlong(mt, language="hi", client=c,
                               cache_dir=tmp_path, batch_size=4)
    assert c.max_tokens[0] >= 2048 * 1.5, "a batch of 4 got the single budget"
    # ...and never below what a single call always had.
    assert condense._max_tokens(1) >= 2048


def test_rewrites_are_matched_by_the_echoed_segment_id(tmp_path):
    """Order is not trusted when ids are present.

    A rewrite pasted onto the wrong segment is short, fluent and wrong, and
    `duration.predict` cannot see it -- the wrong line measures perfectly
    well. So the id decides, even when the model answers out of order.
    """
    mt = _job(tmp_path, _overlong(2))
    c = _Client([_batch({"segment_id": "SEG_0001", "text": "second"},
                        {"segment_id": "SEG_0000", "text": "first"})])
    condense.condense_overlong(mt, language="hi", client=c,
                               cache_dir=tmp_path, batch_size=2)
    segs = json.loads(mt.read_text())["segments"]
    assert segs[0]["mt_text"] == "first"
    assert segs[1]["mt_text"] == "second"


def test_an_unmappable_batch_is_re_asked_one_at_a_time(tmp_path):
    """Batching may cost an extra call; it must never cost a segment."""
    mt = _job(tmp_path, _overlong(2))
    c = _Client([
        # Names a segment that was never asked about -- unmappable.
        _batch({"segment_id": "SEG_9999", "text": "who?"}),
        _batch({"segment_id": "SEG_0000", "text": "a"}),
        _batch({"segment_id": "SEG_0001", "text": "b"}),
    ])
    out = condense.condense_overlong(mt, language="hi", client=c,
                                     cache_dir=tmp_path, batch_size=2)
    assert c.calls == 3, "the batch, then one call per segment"
    assert [len(_sent(c, i)) for i in range(3)] == [2, 1, 1]
    assert out["rewritten"] == 2


def test_a_failed_batch_falls_back_to_single_calls(tmp_path):
    mt = _job(tmp_path, _overlong(2))
    c = _Client([RuntimeError("model is down"),
                 _batch({"segment_id": "SEG_0000", "text": "a"}),
                 _batch({"segment_id": "SEG_0001", "text": "b"})])
    out = condense.condense_overlong(mt, language="hi", client=c,
                                     cache_dir=tmp_path, batch_size=2)
    assert c.calls == 3 and out["rewritten"] == 2


def test_a_segment_the_reply_omits_is_re_asked_alone(tmp_path):
    """A partial answer keeps what it answered and re-asks the rest."""
    mt = _job(tmp_path, _overlong(2))
    c = _Client([_batch({"segment_id": "SEG_0000", "text": "a"}),
                 _batch({"segment_id": "SEG_0001", "text": "b"})])
    out = condense.condense_overlong(mt, language="hi", client=c,
                                     cache_dir=tmp_path, batch_size=2)
    assert c.calls == 2
    assert [len(_sent(c, i)) for i in range(2)] == [2, 1]
    assert out["rewritten"] == 2


def test_half_labelled_replies_are_refused_rather_than_guessed(tmp_path):
    """Positional matching is allowed ONLY when no row carries an id and the
    count is exactly right. A half-labelled reply is where a wrong guess is
    both easy to make and impossible to see afterwards."""
    expected = ["SEG_0000", "SEG_0001"]
    with pytest.raises(condense._BatchMismatch):
        condense._parse_batch(
            {"segments": [{"segment_id": "SEG_0000", "text": "a"},
                          {"text": "b"}]}, expected)
    with pytest.raises(condense._BatchMismatch):     # duplicated id
        condense._parse_batch(
            {"segments": [{"segment_id": "SEG_0000", "text": "a"},
                          {"segment_id": "SEG_0000", "text": "b"}]}, expected)
    with pytest.raises(condense._BatchMismatch):     # wrong count, no ids
        condense._parse_batch({"segments": [{"text": "a"}]}, expected)
    # No ids anywhere and the count is right -- position is all there is.
    got = condense._parse_batch(
        {"segments": [{"text": "a"}, {"text": "b"}]}, expected)
    assert got["SEG_0000"]["text"] == "a"
    assert got["SEG_0001"]["text"] == "b"


def test_a_second_round_re_asks_only_what_still_does_not_fit(tmp_path):
    """Round 2 is a batch of the stragglers, not of everything."""
    # Tight slots: the limit is 1.0 * 1.5 = 1.5s, so a rewrite has to be
    # genuinely short to be done after one round.
    mt = _job(tmp_path, [_seg("SEG_0000", "far too long a line", 1.0, 20.0),
                         _seg("SEG_0001", "far too long a line", 1.0, 20.0)])
    c = _Client([
        # SEG_0000 now fits (0.27s <= 1.5); SEG_0001 is shorter than the
        # 20.0s it declared but still over the cap at 8.1s, so it comes
        # back for a second ask -- alone.
        _batch({"segment_id": "SEG_0000", "text": "ok"},
               {"segment_id": "SEG_0001", "text": "x" * 120}),
        _batch({"segment_id": "SEG_0001", "text": "tiny"}),
    ])
    condense.condense_overlong(mt, language="hi", client=c,
                               cache_dir=tmp_path, batch_size=2, max_rounds=2)
    assert c.calls == 2
    assert [r["segment_id"] for r in _sent(c, 1)] == ["SEG_0001"]


def test_the_prompt_declares_the_batch_contract():
    """The isolation rules are the only defence against the failure the
    duration check cannot see: a line that borrowed its neighbour's
    wording. It is short, fluent, and in the wrong place."""
    ref = (REPO / "translate_agetn.prompt.txt").read_text()
    assert "`segments` array" in ref
    assert "Segment isolation and identity" in ref
    assert "Echo `segment_id` exactly as supplied" in ref
    assert "Never use one segment as context for another" in ref
    assert '"segments": [' in ref, "the output schema is still the old flat object"


def test_the_batch_size_is_an_option_everywhere():
    """A knob reachable from only one entry point is a knob the UI runner,
    the edit agent and `experiment.py` cannot reach."""
    import inspect
    assert "batch_size" in inspect.signature(condense.condense_overlong).parameters
    assert condense.DEFAULT_BATCH_SIZE == 2
    cli = (REPO / "backend" / "cli.py").read_text()
    assert "--condense-batch-size" in cli
    assert "condense_batch_size=args.condense_batch_size," in cli
    pipe = (REPO / "backend" / "pipeline.py").read_text()
    assert "batch_size=condense_batch_size," in pipe
    assert "CONDENSE_BATCH_SIZE" in (REPO / ".env.example").read_text()


# --- the UI turns it on ----------------------------------------------------

def test_the_ui_enables_9c2_by_default():
    """On for every UI run since 2026-10-22, by request. A UI run has nobody
    watching 9c's overrun warnings go past; the first sign of the problem is
    a finished dub drifting off the mouth."""
    from ui import presets
    kw = presets.build_kwargs("standard", target_language="hi")
    assert kw["condense_overlong_text"] is True
    assert kw["duration_adjust"] is True, "9c2 reads 9c's estimates"
    assert kw["condense_speed_cap"] == 1.5


def test_the_ui_plans_the_9c2_node():
    """Enabled but unplanned means the stage runs and arrives as an
    UNPLANNED node -- the "12 of 14 then extra nodes" shape."""
    from ui import presets
    from ui.progress import plan_stages
    cfg = presets.plan_config(
        presets.build_kwargs("standard", target_language="hi"))
    assert cfg.get("condense_overlong_text") is True
    assert any(s.get("key") == "9c2" for s in plan_stages(cfg))
