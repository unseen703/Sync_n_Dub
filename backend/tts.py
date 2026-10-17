"""Backend-facing TTS entry point.

Reads the merged ASR+MT+codemix+duration JSON, synthesises one clip per
segment with the per-speaker assigned voice, applies the WSOLA ±10% stretch
to meet `target_duration_s`, writes a per-segment TTS manifest, and
updates the main JSON in place with:

  segments[].tts_audio       : path to the stretched WAV
  segments[].tts_duration_s  : final length after stretch
  segments[].stretch_factor  : WSOLA stretch applied
  segments[].qc.budget_violation : True when cap was hit
  extra.tts                  : {model, voice counts, violations, …}
"""
from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from models.tts import TTSRequest, available_models, get_tts

from .duration import wsola_stretch_cap
from .sanitize import sanitize_for_tts

if TYPE_CHECKING:                       # heavy-ish import; only needed for the hint
    from .duration import DurationModel

log = logging.getLogger(__name__)

SPEED_CONTROL_MODES = ("off", "oneshot", "closed_loop")
DEFAULT_DURATION_MODEL_DIR = Path("data/duration_model")


def synthesize_segments(
    merged_json: Path,
    *,
    tts_model: str,
    language: str,
    speaker_voice_map: Dict[str, dict],   # spk_id → {engine, voice, reference_audio, reference_text?}
    out_dir: Path,
    stretched_dir: Optional[Path] = None,
    sample_rate: int = 24000,
    max_stretch: float = 0.10,
    tts_kwargs: Optional[dict] = None,
    out_json: Optional[Path] = None,
    stretch_backend: str = "auto",
    apply_stretch: bool = True,
    speed_control: str = "off",            # "off" | "oneshot" | "closed_loop"
    speed_bounds: Tuple[float, float] = (0.85, 1.15),
    speed_tolerance: float = 0.05,
    speed_max_rounds: int = 2,
    duration_model: Optional["DurationModel"] = None,
    duration_model_dir: Optional[Path] = None,
) -> dict:
    """Synthesise every segment and stretch each to its target duration.

    `stretch_backend` selects the time-stretch implementation and is recorded in
    `extra.tts` so a finished run carries evidence of which one ran -- the old
    code silently fell back to a pitch-shifting resampler and left no trace.

    `apply_stretch=False` writes raw clips only and leaves `stretch_factor`
    unset, for variants where a downstream timing plan owns all stretching.
    Without it the pipeline stretches twice: once here, once as a global
    `atempo` in the mix.

    `speed_control` asks the ENGINE to speak at a different rate instead of
    stretching the waveform afterwards -- the method-2 arm of the comparison.
      "off"         : no rate is ever sent (bit-identical to the old path).
      "oneshot"     : predict the clip length from the text with a fitted
                      duration model and set the rate once, before synthesis.
                      Costs nothing extra; only as good as the predictor.
      "closed_loop" : synthesise at 1.0, measure, re-synthesise at
                      measured/target, repeat while outside `speed_tolerance`
                      for at most `speed_max_rounds` extra rounds. Accurate,
                      but each round is a real TTS call -- `extra.tts
                      .extra_synthesis_rounds` records the bill.
    Both no-op (with one warning) on an adapter whose `supports_rate` is False.

    There is no cross-engine fallback. Bodhan is the only TTS engine here, so
    when it cannot speak there is no second voice to borrow -- the segments
    come back with `tts_error` set and the count lands in `extra.tts.errors`.
    The fallback that does exist is one layer up and is about WHICH voice,
    not which engine: `backend.voices.assign_voices` picks by target language
    and speaker gender when it cannot rank by timbre.
    """
    if tts_model not in available_models():
        raise ValueError(
            f"Unknown TTS model {tts_model!r}. "
            f"Available: {', '.join(available_models())}"
        )
    if speed_control not in SPEED_CONTROL_MODES:
        raise ValueError(
            f"Unknown speed_control {speed_control!r}. "
            f"Available: {', '.join(SPEED_CONTROL_MODES)}"
        )
    speed_control_requested = speed_control
    merged_json = Path(merged_json)
    data = json.loads(merged_json.read_text())
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"; raw_dir.mkdir(exist_ok=True)
    stretched_dir = Path(stretched_dir or out_dir / "stretched")
    stretched_dir.mkdir(parents=True, exist_ok=True)

    requests: List[TTSRequest] = []
    for seg in data.get("segments", []):
        spk = seg.get("speaker")
        info = speaker_voice_map.get(spk) or {}
        text = seg.get("final_text") or seg.get("mt_text") or seg.get("text") or ""
        text = sanitize_for_tts(text, seg_id=seg["id"])
        # Per-SEGMENT overrides beat the per-speaker assignment. `backend.edits`
        # writes these for an edit scoped to a window rather than to a whole
        # speaker ("the voice at 1:20 should be female"), and nothing else in
        # the pipeline sets them, so a run without edits is unchanged.
        ref = seg.get("tts_reference_audio") or info.get("reference_audio")
        requests.append(TTSRequest(
            id=seg["id"],
            text=text,
            voice=seg.get("tts_voice") or info.get("voice"),
            language=language,
            reference_audio=(Path(ref) if ref else None),
            reference_text=info.get("reference_text"),
            speaker=spk,
            style=seg.get("tts_style") or info.get("style"),
        ))

    # Resolve up front so the manifest records the backend even if no segment
    # ends up needing a stretch.
    if apply_stretch:
        from .stretch import resolve_backend
        resolved_backend = resolve_backend(stretch_backend)
        log.info("tts: stretch backend %r -> %r", stretch_backend, resolved_backend)
    else:
        resolved_backend = None

    tts = get_tts(tts_model, **(tts_kwargs or {}))

    # ---- synthesis-side speed control (method 2) --------------------------
    targets = {seg["id"]: _segment_target_s(seg) for seg in data.get("segments", [])}
    rate_supported = bool(getattr(tts, "supports_rate", False))
    speed_notes: dict = {}
    if speed_control != "off" and not rate_supported:
        # Faking a rate with a stretch here would make the comparison
        # meaningless, so refuse rather than approximate.
        log.warning("tts: %s does not accept a synthesis rate (supports_rate=False); "
                    "speed_control=%r disabled", tts_model, speed_control)
        speed_notes["disabled_reason"] = f"{tts_model}.supports_rate is False"
        speed_control = "off"

    if speed_control == "oneshot":
        model = duration_model or _resolve_duration_model(
            language=language, speaker_voice_map=speaker_voice_map,
            cache_dir=Path(duration_model_dir or DEFAULT_DURATION_MODEL_DIR),
        )
        if model.n_fit == 0:
            log.warning(
                "tts: ***** speed_control='oneshot' is running on the HAND-GUESSED "
                "default duration model (n_fit=0) -- its rates are a guess, not a "
                "prediction. Fit one with `python -m backend.duration fit`. *****")
            speed_notes["uncalibrated_duration_model"] = True
        speed_notes["duration_model"] = model.to_jsonable()
        for req in requests:
            target = targets.get(req.id, 0.0)
            if target <= 0:
                continue
            # predicted/target > 1 means the text is too long for the slot, so
            # the engine must speak proportionally faster.
            req.rate = _clamp_rate(model.predict(req.text) / target, speed_bounds)
        log.info("tts: oneshot rates set for %d segments (model n_fit=%d, MAE=%.2fs)",
                 len(requests), model.n_fit, model.mae_s)

    # Explicit per-segment rate from `backend.edits.set_speed`. It overrides
    # whatever the predictor chose, and it is still clamped to `speed_bounds`:
    # anything the clamp refuses is picked up by the stretch below, which has
    # its own per-segment cap for exactly this case.
    rate_overrides = {
        seg["id"]: float(seg["tts_rate_override"])
        for seg in data.get("segments", [])
        if seg.get("tts_rate_override") is not None
    }
    if rate_overrides:
        if rate_supported:
            for req in requests:
                if req.id in rate_overrides:
                    req.rate = _clamp_rate(rate_overrides[req.id], speed_bounds)
            log.info("tts: %d segment(s) carry an edited rate override",
                     len(rate_overrides))
        else:
            log.info("tts: %s ignores a synthesis rate; the %d edited "
                     "segment(s) are sped up by the stretch instead",
                     tts_model, len(rate_overrides))

    log.info("tts: synthesising %d segments with %s (lang=%s, speed_control=%s)",
             len(requests), tts_model, language, speed_control)
    result = tts.synthesize(
        requests, out_dir=raw_dir, language=language, sample_rate=sample_rate,
    )

    rates = {r.id: float(r.rate) for r in requests}
    rounds = {r.id: 1 for r in requests}
    extra_synthesis_rounds = 0
    if speed_control == "closed_loop":
        result, rates, rounds, extra_synthesis_rounds = _run_closed_loop(
            tts, requests, result,
            targets=targets, raw_dir=raw_dir, language=language,
            sample_rate=sample_rate, speed_bounds=speed_bounds,
            tolerance=speed_tolerance, max_rounds=speed_max_rounds,
        )

    violations = 0
    errors = 0
    stretch_errors = 0
    by_id = {s.id: s for s in result.segments}
    for seg in data.get("segments", []):
        rec = by_id.get(seg["id"])
        if rec is None:
            continue
        if speed_control != "off" or seg["id"] in rate_overrides:
            # Written even for failed segments: a sweep needs to see which rate
            # was in flight when a clip went missing.
            seg["tts_rate"] = rates.get(seg["id"], 1.0)
            seg["tts_rounds"] = rounds.get(seg["id"], 1)
        if rec.error:
            errors += 1
            seg["tts_audio"] = None
            seg.setdefault("qc", {})["tts_error"] = rec.error
            continue
        target = float(seg.get("target_duration_s") or
                       (seg.get("end", 0.0) - seg.get("start", 0.0)))
        import soundfile as sf
        seg["tts_raw_audio"] = str(rec.audio_path)
        if not apply_stretch:
            # The timing plan will stretch; stretching here too would compound.
            seg["tts_audio"] = str(rec.audio_path)
            seg["tts_duration_s"] = float(sf.info(str(rec.audio_path)).duration)
            seg.pop("stretch_factor", None)
            continue
        stretched_path = stretched_dir / f"{rec.id}.wav"
        # A deliberate speed edit asks for more than the default +-10%; without
        # its own cap the stretch clamps it back and the edit arrives as a
        # budget_violation instead of as the speed-up that was requested.
        seg_cap = seg.get("tts_max_stretch")
        seg_cap = float(seg_cap) if seg_cap is not None else max_stretch
        try:
            factor, violated = wsola_stretch_cap(
                rec.audio_path, stretched_path,
                target_duration_s=target, max_stretch=seg_cap,
                backend=stretch_backend,
            )
        except Exception as e:
            # A clip shorter than the backend's analysis window used to yield a
            # zero-length "success"; surface it instead of mixing in silence.
            stretch_errors += 1
            log.warning("tts: stretch failed for %s (%s); using raw clip",
                        rec.id, e)
            seg["tts_audio"] = str(rec.audio_path)
            seg["tts_duration_s"] = float(sf.info(str(rec.audio_path)).duration)
            seg.setdefault("qc", {})["stretch_error"] = str(e)
            continue
        final_dur = float(sf.info(str(stretched_path)).duration)
        seg["tts_audio"] = str(stretched_path)
        seg["tts_duration_s"] = final_dur
        seg["stretch_factor"] = factor
        if violated:
            violations += 1
            seg.setdefault("qc", {})["budget_violation"] = True

    applied = [rates.get(s.id, 1.0) for s in result.segments if not s.error]
    data.setdefault("extra", {})["tts"] = {
        "model": result.model_name,
        "language": language,
        "sample_rate": sample_rate,
        "max_stretch": max_stretch,
        "stretch_backend": resolved_backend,
        "stretch_backend_requested": stretch_backend,
        "apply_stretch": apply_stretch,
        "segments": len(result.segments),
        "errors": errors,
        "stretch_errors": stretch_errors,
        "budget_violations": violations,
        "raw_dir": str(raw_dir),
        "stretched_dir": str(stretched_dir),
        "speed_control": speed_control,
        "speed_control_requested": speed_control_requested,
        "speed_rate_supported": rate_supported,
        "speed_bounds": list(speed_bounds),
        "speed_tolerance": speed_tolerance,
        "speed_max_rounds": speed_max_rounds,
        "rate_mean": (sum(applied) / len(applied)) if applied else 1.0,
        "rate_min": min(applied) if applied else 1.0,
        "rate_max": max(applied) if applied else 1.0,
        # Every re-synthesis is a billable/rate-limited engine call; the report
        # has to be able to quote the price of the closed loop.
        "extra_synthesis_rounds": extra_synthesis_rounds,
        **({"speed_notes": speed_notes} if speed_notes else {}),
    }

    dest = Path(out_json) if out_json else merged_json
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    log.info("tts: wrote %s (errors=%d, cap_violations=%d)",
             dest, errors, violations)
    return data


# ---------------------------------------------------------------------------
# synthesis-side speed control helpers
# ---------------------------------------------------------------------------

def _segment_target_s(seg: dict) -> float:
    """Slot length this clip has to fit in, in seconds (0.0 when unknown)."""
    t = seg.get("target_duration_s")
    if t is None:
        t = float(seg.get("end", 0.0)) - float(seg.get("start", 0.0))
    try:
        return max(0.0, float(t))
    except (TypeError, ValueError):
        return 0.0


def _clamp_rate(rate: float, bounds: Tuple[float, float]) -> float:
    """Keep the requested rate inside the intelligibility window.

    Past roughly +-15% the Azure voices start to sound rushed or drawled, and
    a rate the engine refuses is worse than a clip we stretch afterwards.
    """
    lo, hi = float(bounds[0]), float(bounds[1])
    if lo > hi:
        lo, hi = hi, lo
    try:
        r = float(rate)
    except (TypeError, ValueError):
        return 1.0
    if r != r or r <= 0:          # NaN / nonsense from a zero-length target
        return 1.0
    return min(hi, max(lo, r))


def _resolve_duration_model(
    *, language: str, speaker_voice_map: Dict[str, dict], cache_dir: Path,
) -> "DurationModel":
    """Pick the fitted `<lang>.<voice>.json` that matches this run's voices.

    Tries each configured voice before the generic "default" slot so a run that
    calibrated Swara does not silently fall back to the hand-guessed constants.
    """
    from .duration import DEFAULT_MODEL, load_or_default_model
    primary = (language or "hi").split("-")[0].lower()
    voices: List[str] = []
    for info in (speaker_voice_map or {}).values():
        v = (info or {}).get("voice")
        if v and v not in voices:
            voices.append(v)
    for voice in [*voices, "default"]:
        model = load_or_default_model(cache_dir, language=primary, voice=voice)
        if model.n_fit:
            log.info("tts: using fitted duration model %s/%s (n=%d, MAE=%.2fs)",
                     primary, voice, model.n_fit, model.mae_s)
            return model
    return DEFAULT_MODEL


def _run_closed_loop(
    tts, requests: List[TTSRequest], result,
    *,
    targets: Dict[str, float],
    raw_dir: Path,
    language: str,
    sample_rate: int,
    speed_bounds: Tuple[float, float],
    tolerance: float,
    max_rounds: int,
):
    """Measure, re-rate, re-synthesise until every clip is within tolerance.

    Returns `(result, rates, rounds, extra_calls)`; `result.segments` is
    rewritten in request order to point at the accepted clip per segment.
    """
    by_id = {s.id: s for s in result.segments}
    by_req = {r.id: r for r in requests}
    rates = {r.id: float(r.rate) for r in requests}
    rounds = {r.id: 1 for r in requests}
    extra_calls = 0

    for rnd in range(1, int(max_rounds) + 1):
        todo: List[Tuple[str, float]] = []
        for rid, rec in by_id.items():
            target = targets.get(rid, 0.0)
            if target <= 0 or rec.error or rec.duration_s <= 0:
                continue
            ratio = rec.duration_s / target
            if abs(ratio - 1.0) <= tolerance:
                continue
            # Multiplicative update: spoken at rate r the clip ran `ratio` too
            # long, so r*ratio is the rate that would have landed on target.
            # On the first round r == 1, i.e. plain measured/target.
            new_rate = _clamp_rate(rates[rid] * ratio, speed_bounds)
            if abs(new_rate - rates[rid]) < 1e-3:
                # Pinned at a bound (or already there): another engine call
                # would return the same clip, so stop spending on it.
                continue
            todo.append((rid, new_rate))
        if not todo:
            break

        round_dir = Path(raw_dir) / f"round{rnd}"
        round_dir.mkdir(parents=True, exist_ok=True)
        reqs = [replace(by_req[rid], rate=new_rate) for rid, new_rate in todo]
        log.info("tts: closed loop round %d -- re-synthesising %d segment(s)",
                 rnd, len(reqs))
        try:
            rr = tts.synthesize(reqs, out_dir=round_dir, language=language,
                                sample_rate=sample_rate)
        except Exception as e:
            # One bad round must not lose the clips we already have.
            log.warning("tts: closed loop round %d failed wholesale (%s); "
                        "keeping the clips from the previous round", rnd, e)
            break
        extra_calls += len(reqs)

        proposed = dict(todo)
        for rec in rr.segments:
            prev = by_id.get(rec.id)
            target = targets.get(rec.id, 0.0)
            if prev is None or target <= 0 or rec.error or rec.duration_s <= 0:
                log.warning("tts: closed loop round %d: %s came back unusable "
                            "(%s); keeping previous clip",
                            rnd, rec.id, rec.error or "zero length")
                continue
            if abs(rec.duration_s / target - 1.0) >= abs(prev.duration_s / target - 1.0):
                # The engine quantises its rate to whole percent and prosody is
                # not linear in it, so a round can overshoot. Never regress.
                log.info("tts: closed loop round %d: %s at rate %.3f is no closer "
                         "(%.2fs vs %.2fs, target %.2fs); keeping previous clip",
                         rnd, rec.id, proposed[rec.id], rec.duration_s,
                         prev.duration_s, target)
                continue
            by_id[rec.id] = rec
            rates[rec.id] = proposed[rec.id]
            rounds[rec.id] = rnd + 1

    result.segments = [by_id[r.id] for r in requests if r.id in by_id]
    return result, rates, rounds, extra_calls
