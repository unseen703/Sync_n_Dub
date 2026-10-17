"""Audio mixing — overlay per-segment TTS/VC clips onto the M&E stem.

Reads the merged JSON produced after the TTS/VC stages. For each segment
whose `tts_audio` (or `vc_audio` when present) exists:
  - places it on the timeline using a shift-forward schedule: the clip
    starts at `max(segment.start, cursor)` where `cursor` is the end of
    the previous clip. This preserves the entire TTS clip (no `atrim`)
    and prevents overlap.
  - sums it with the M&E stem under `amix`
  - optionally re-paces the whole mix with a single `atempo` so the dub
    lands back on the source-video duration (strategy=global_tempo).

Three strategies. `--speed-mode` on the CLI is the two-option front end to
them: `global` -> `global_tempo`, `segment` -> `timing_plan` rendering a
`naive_slot` plan with a zero slot margin, which fits every clip to its own
slot in BOTH directions (a short clip is stretched out to fill, rather than
leaving a pause). `no_tempo` and the research arms stay reachable directly.

  * `global_tempo` (default): natural-pace per clip; one global atempo
    at the end to match source duration. Pitch preserved. Gives a
    uniformly slightly-fast dub whose length == source video.
  * `no_tempo`: pure shift-forward with no stretching anywhere. The dub
    is lossless and natural-paced but can be longer than the video; the
    mux step will trim the tail with `-shortest`.
  * `timing_plan`: render a `backend.timing.TimingPlan`. The plan owns
    placement AND tempo: each `Piece` is a chunk of a raw TTS clip with
    its own `dst_start` and `local_tempo`, so there is no global atempo
    and no shift-forward cursor here at all. Every producer in
    `backend.timing.PRODUCERS` — including `naive_slot`, the control arm
    that reproduces `global_tempo` — renders through this one path, which
    is what keeps a producer comparison about timing rather than about
    two different ffmpeg graphs.

Which clip per segment (`clip_source`): the TTS stage writes TWO files and
records both on the segment — `tts_raw_audio` is the UNSTRETCHED synth
output, while `tts_audio` is the WSOLA-stretched clip and so may already
have been re-timed to fit its slot. Mixing `tts_audio` under
`strategy=global_tempo` therefore stretches a second time, on top of the
first. `clip_source` makes that choice explicit; `vc_audio` (voice
conversion) is derived from the stretched clip and inherits its timing.

Nothing is written outside `out_dir`. Fallbacks:
  * If the M&E stem is missing (user ran `--skip-stems`), we still write
    the dubbed-dialogue-only mix and warn.
  * Segments with no synthesized audio are skipped silently; their slot
    carries whatever is in the M&E bed (silence or source music).
"""
from __future__ import annotations

import json
import logging
import subprocess
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .media_io import _ffmpeg_bin, probe_duration

log = logging.getLogger(__name__)


def _audio_duration(path: Path) -> float:
    """Duration in seconds for a TTS clip.

    Uses Python's stdlib `wave` for PCM WAVs (fast, no ffprobe dependency)
    and falls back to `probe_duration` for anything else.
    """
    p = Path(path)
    if p.suffix.lower() == ".wav":
        try:
            with wave.open(str(p)) as w:
                n = w.getnframes()
                sr = w.getframerate()
                if sr > 0:
                    return n / sr
        except Exception:
            pass
    return probe_duration(p)


@dataclass
class MixResult:
    audio_wav: Path
    video_mp4: Optional[Path]
    num_mixed: int
    num_skipped: int
    sample_rate: int
    extra: dict = field(default_factory=dict)


STRATEGIES = ("global_tempo", "no_tempo", "timing_plan")

CLIP_SOURCES = ("auto", "raw", "stretched", "vc")

# One ffmpeg `-i` per piece, so the argv and the filter graph both grow
# linearly with the piece count. Past a few hundred the failure mode is an
# opaque ffmpeg/OS argument-length error thousands of characters long, which
# is a terrible way to learn that a producer over-segmented.
DEFAULT_MAX_PIECES = 400

# A stretch filter with less input than its analysis window emits nothing and
# still exits 0 (see backend.stretch) — the piece would reach the mix as pure
# silence. Below this we render the piece at its natural pace instead; a few
# milliseconds of tempo error on a sliver is invisible, a dropped piece is not.
_MIN_STRETCH_INPUT_S = 0.05

# Below this a `local_tempo` is not worth a filter instance at all.
_TEMPO_EPS = 1e-3

# Explicit clip_source → the segment field it reads. "auto" is absent on
# purpose: it resolves per segment (see `_resolve_clip`).
_CLIP_FIELDS = {
    "raw": "tts_raw_audio",
    "stretched": "tts_audio",
    "vc": "vc_audio",
}

# Any field that counts as "this segment was synthesised", used to tell a
# genuinely empty segment apart from one that is merely missing the
# requested variant.
_ALL_CLIP_FIELDS = ("vc_audio", "tts_audio", "tts_raw_audio")


def _resolve_clip(seg: dict, clip_source: str) -> tuple[Optional[str], Optional[str]]:
    """Pick the audio field for one segment; returns (field_name, path).

    Both are None when the segment carries no synthesised audio at all.
    An explicit `clip_source` whose field is missing on an otherwise
    synthesised segment raises instead of falling back — a silent
    fallback to `tts_audio` is what hid the double-stretch bug.
    """
    if clip_source == "auto":
        if seg.get("vc_audio"):
            return "vc_audio", seg["vc_audio"]
        if seg.get("tts_audio"):
            return "tts_audio", seg["tts_audio"]
        return None, None

    clip_field = _CLIP_FIELDS[clip_source]
    value = seg.get(clip_field)
    if value:
        return clip_field, value
    if any(seg.get(f) for f in _ALL_CLIP_FIELDS):
        have = [f for f in _ALL_CLIP_FIELDS if seg.get(f)]
        raise RuntimeError(
            f"mix: segment {seg.get('id')!r} has no {clip_field!r} "
            f"(clip_source={clip_source!r}) but does carry {have}. "
            f"Re-run the stage that produces {clip_field!r}, or pick a "
            f"clip_source that exists."
        )
    return None, None


@dataclass
class _PlacedPiece:
    """One `timing.Piece` resolved against a concrete wav on disk.

    `src_start`/`src_end` are seconds INTO the clip (the plan's convention);
    `dst_start` is absolute output-timeline seconds. Keeping both on one
    object is the only place the two time bases meet, so the conversion bug
    the schema warns about has exactly one home.
    """
    sid: str
    wav: Path
    src_start: float
    src_end: float
    dst_start: float
    tempo: float

    @property
    def src_dur(self) -> float:
        return max(0.0, self.src_end - self.src_start)

    @property
    def dst_end(self) -> float:
        return self.dst_start + self.src_dur / max(1e-6, self.tempo)


def _ffmpeg_stretch_backend(name: str) -> str:
    """Resolve `name` to a backend that is an ffmpeg *filter*.

    The mix renders inside one `filter_complex`, so `librosa`/`linear` — which
    are in-process array operations — cannot participate. Rejecting them here
    with a named error beats letting `ffmpeg_stretch_filter` raise a generic
    ValueError from three frames down.
    """
    from . import stretch as stretch_mod

    ffmpeg_backends = ("rubberband", "ffmpeg_atempo")
    if name == "auto":
        for b in stretch_mod.available_backends():
            if b in ffmpeg_backends:
                log.info("mix: stretch backend %s (available: %s)", b,
                         ", ".join(stretch_mod.available_backends()))
                return b
        raise RuntimeError(
            "mix: strategy='timing_plan' needs an ffmpeg stretch filter "
            f"({' or '.join(ffmpeg_backends)}) and this host has neither."
        )
    if name not in ffmpeg_backends:
        raise ValueError(
            f"mix: stretch_backend={name!r} cannot run inside the mix filter "
            f"graph; choose one of {ffmpeg_backends} or 'auto'."
        )
    return stretch_mod.resolve_backend(name)       # raises if unavailable


def _build_plan_pieces(
    data: dict, tplan, clip_source: str,
) -> tuple[List[_PlacedPiece], List[str], List[dict]]:
    """Resolve every `Piece` of `tplan` to a wav + a trim window.

    Returns (pieces, clip_fields, skipped). A segment that cannot be resolved
    is skipped with a recorded reason rather than raising: one bad segment
    must not kill a producer sweep.
    """
    import soundfile as sf

    by_id = {s.get("id"): s for s in data.get("segments", [])}
    pieces: List[_PlacedPiece] = []
    clip_fields: List[str] = []
    skipped: List[dict] = []

    for sp in tplan.segments:
        seg = by_id.get(sp.id)
        if seg is None:
            skipped.append({"id": sp.id, "reason": "not_in_merged_json"})
            continue
        if not sp.pieces:
            # Normal for a `no_audio` segment: the plan says "render nothing
            # here", and the M&E bed covers the slot.
            skipped.append({"id": sp.id, "reason": "no_pieces"})
            continue
        try:
            clip_field, audio = _resolve_clip(seg, clip_source)
        except RuntimeError as e:
            log.warning("mix: %s", e)
            skipped.append({"id": sp.id, "reason": "clip_source_missing"})
            continue
        if not audio:
            skipped.append({"id": sp.id, "reason": "no_audio"})
            continue
        wav = Path(audio)
        if not wav.exists():
            log.warning("mix: segment %s %s=%s does not exist; skipping",
                        sp.id, clip_field, wav)
            skipped.append({"id": sp.id, "reason": "clip_missing_on_disk"})
            continue
        try:
            clip_dur = float(sf.info(str(wav)).duration)
        except Exception as e:
            log.warning("mix: could not probe %s (%s); skipping segment %s",
                        wav, e, sp.id)
            skipped.append({"id": sp.id, "reason": "unprobeable"})
            continue
        if clip_field not in clip_fields:
            clip_fields.append(clip_field)

        for i, pc in enumerate(sp.pieces):
            # Clamp into the clip: a producer that cut past the end would
            # otherwise get a silently short piece and a timeline that drifts.
            s = max(0.0, min(clip_dur, float(pc.src_span.start)))
            e = max(0.0, min(clip_dur, float(pc.src_span.end)))
            if e - s <= 1e-4:
                log.warning("mix: %s piece %d is empty after clamping to the "
                            "%.3fs clip (src_span=%.3f-%.3f); skipping",
                            sp.id, i, clip_dur, pc.src_span.start, pc.src_span.end)
                skipped.append({"id": sp.id, "piece": i, "reason": "empty_span"})
                continue
            if e < float(pc.src_span.end) - 1e-3:
                log.warning("mix: %s piece %d asked for %.3fs of a %.3fs clip; "
                            "truncated", sp.id, i, pc.src_span.end, clip_dur)
            pieces.append(_PlacedPiece(
                sid=sp.id, wav=wav, src_start=s, src_end=e,
                dst_start=max(0.0, float(pc.dst_start)),
                tempo=max(1e-6, float(pc.local_tempo)),
            ))
    return pieces, clip_fields, skipped


def mix_dubbed_audio(
    merged_json: Path,
    *,
    me_stem: Optional[Path],
    source_video: Path,
    out_dir: Path,
    sample_rate: int = 48000,
    tts_gain_db: float = 3.0,
    me_gain_db: float = -3.0,
    mux_video: bool = True,
    strategy: str = "global_tempo",
    tempo_cap: float = 1.5,
    clip_source: str = "auto",
    out_json: Optional[Path] = None,
    timing_plan: Optional[Path] = None,
    stretch_backend: str = "auto",
    max_pieces: int = DEFAULT_MAX_PIECES,
) -> MixResult:
    """Build the final dubbed audio and (optionally) mux it back on video.

    `strategy`:
      * ``global_tempo`` (default) — place every clip at natural pace via
        shift-forward, then apply a single global ``atempo`` so the dub
        matches the source-video duration. Capped at ``tempo_cap``.
      * ``no_tempo`` — pure shift-forward, no stretching. The dub may be
        longer than the source; mux trims the tail with ``-shortest``.
      * ``timing_plan`` — render a `backend.timing.TimingPlan`, read from
        ``timing_plan`` if given and otherwise from ``extra["timing_plan"]``
        of the merged JSON. Placement and tempo both come from the plan;
        ``tempo_cap`` is ignored (the producer already capped) and no global
        ``atempo`` is applied.

    `clip_source` picks which per-segment clip to place:
      * ``auto`` (default) — ``vc_audio`` when present, else ``tts_audio``;
        under ``timing_plan`` it resolves to ``raw`` instead (see below).
        Kept bit-for-bit compatible with pre-``clip_source`` callers.
      * ``raw`` — ``tts_raw_audio``, the unstretched synth output. Use this
        with ``global_tempo`` to avoid stretching twice.
      * ``stretched`` — ``tts_audio`` (already WSOLA-stretched by the TTS
        stage).
      * ``vc`` — ``vc_audio``.

    `stretch_backend` is the ffmpeg filter used for every ``local_tempo``
    under ``timing_plan`` (``rubberband``, ``ffmpeg_atempo`` or ``auto``);
    unused by the other strategies.

    `max_pieces` guards the one-input-per-piece graph (see
    ``DEFAULT_MAX_PIECES``).

    `out_json` receives the updated manifest; without it `merged_json` is
    rewritten in place.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"mix: unknown strategy {strategy!r}; "
                         f"choose one of {STRATEGIES}")
    if clip_source not in CLIP_SOURCES:
        raise ValueError(f"mix: unknown clip_source {clip_source!r}; "
                         f"choose one of {CLIP_SOURCES}")
    merged_json = Path(merged_json)
    data = json.loads(merged_json.read_text())
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    source_video = Path(source_video)

    # Probed up front because both branches report it; only `global_tempo`
    # actually steers on it.
    try:
        source_duration = probe_duration(source_video)
    except Exception:
        source_duration = max(float(s.get("end", s.get("end_s", 0.0)) or 0.0)
                              for s in data.get("segments", []))
        source_duration = max(source_duration, 1.0)

    out_wav = out_dir / f"{source_video.stem}.dubbed.wav"
    plan_extra: dict = {}

    if strategy == "timing_plan":
        from . import timing as timing_mod

        if timing_plan is not None:
            tplan = timing_mod.TimingPlan.read(Path(timing_plan))
            plan_src = str(timing_plan)
        else:
            tplan = timing_mod.load_plan_from_merged(merged_json)
            plan_src = f"{merged_json}:extra.timing_plan"
            if tplan is None:
                raise RuntimeError(
                    f"mix: strategy='timing_plan' needs a plan. Pass "
                    f"timing_plan=<path>, or run a producer and "
                    f"backend.timing.apply_plan_to_merged so {merged_json} "
                    f"carries extra['timing_plan']."
                )

        # The plan owns ALL stretching. Letting "auto" land on `tts_audio`
        # here is what made a real run compress 1.098x in the TTS stage and
        # then 1.29x again in the mix; "raw" is the only default that keeps
        # `local_tempo` meaning what the producer computed.
        effective_clip_source = "raw" if clip_source == "auto" else clip_source
        pieces, clip_fields, skipped = _build_plan_pieces(
            data, tplan, effective_clip_source)
        if not pieces:
            raise RuntimeError(
                f"mix: timing plan {plan_src!r} produced no renderable pieces "
                f"(skipped: {skipped or 'nothing'}). Did TTS run?"
            )
        if len(pieces) > max_pieces:
            raise RuntimeError(
                f"mix: timing plan has {len(pieces)} pieces, over max_pieces="
                f"{max_pieces}. Each piece costs one ffmpeg input and one "
                f"filter chain, so this would fail deep inside ffmpeg instead "
                f"of here. Raise max_pieces deliberately, or use a producer "
                f"that cuts fewer pieces."
            )

        needs_stretch = any(abs(p.tempo - 1.0) >= _TEMPO_EPS for p in pieces)
        resolved_backend = (_ffmpeg_stretch_backend(stretch_backend)
                            if needs_stretch else None)
        global_tempo = 1.0          # per-piece local_tempo is the only tempo
        fit_duration = max(p.dst_end for p in pieces)
        total_duration = fit_duration + 0.5
        num_mixed = len({p.sid for p in pieces})

        log.info("mix: %d pieces over %d segments, strategy=timing_plan, "
                 "producer=%s, clip_source=%s->%s(%s), stretch=%s, me_stem=%s",
                 len(pieces), num_mixed, tplan.producer, clip_source,
                 effective_clip_source, "+".join(clip_fields),
                 resolved_backend or "none needed", me_stem)
        log.info("mix: fit_dur=%.2fs source_dur=%.2fs drift=%+.2fs",
                 fit_duration, source_duration, fit_duration - source_duration)
        for pc in pieces:
            log.debug("  %s src=%.3f-%.3f -> dst=%.3f tempo=%.4f",
                      pc.sid, pc.src_start, pc.src_end, pc.dst_start, pc.tempo)
        if skipped:
            log.warning("mix: %d segment(s)/piece(s) skipped: %s",
                        len(skipped), skipped)

        _ffmpeg_mix_plan(
            pieces=pieces,
            me_stem=me_stem,
            out_wav=out_wav,
            total_duration=total_duration,
            sample_rate=sample_rate,
            tts_gain_db=tts_gain_db,
            me_gain_db=me_gain_db,
            stretch_backend=resolved_backend,
        )
        plan_extra = {
            "timing_producer": tplan.producer,
            "timing_plan_source": plan_src,
            "timing_summary": tplan.summary,
            "n_pieces": len(pieces),
            "stretch_backend": resolved_backend,
            "skipped": skipped,
            "effective_clip_source": effective_clip_source,
        }
    else:
        segs = []
        clip_fields = []
        for seg in data.get("segments", []):
            clip_field, audio = _resolve_clip(seg, clip_source)
            if not audio:
                continue
            p = Path(audio)
            if not p.exists():
                log.warning("mix: segment %s %s=%s does not exist; skipping",
                            seg.get("id"), clip_field, p)
                continue
            if clip_field not in clip_fields:
                clip_fields.append(clip_field)
            start_s = float(seg.get("start", seg.get("start_s", 0.0)))
            try:
                raw_dur = _audio_duration(p)
            except Exception as e:
                log.warning("mix: could not probe duration of %s (%s); "
                            "treating as 0 — overrun shift will be inaccurate",
                            p, e)
                raw_dur = 0.0
            segs.append((seg["id"], start_s, p, raw_dur))
        if not segs:
            raise RuntimeError(
                f"mix: no synthesised segments found in {merged_json}. "
                f"Did TTS run?"
            )
        segs.sort(key=lambda x: x[1])

        # Pure shift-forward placement: no per-clip tempo. Each clip plays its
        # full raw duration; the next clip waits if that overruns.
        shift_plan: List = []
        cursor = 0.0
        for sid, start_s, wav, raw_dur in segs:
            placed = max(start_s, cursor)
            cursor = placed + raw_dur
            shift_plan.append((sid, placed, wav, raw_dur, placed - start_s))
        fit_duration = cursor
        num_mixed = len(segs)

        # Global atempo factor (only for strategy == global_tempo).
        if strategy == "global_tempo" and fit_duration > source_duration + 0.01:
            global_tempo = min(tempo_cap, fit_duration / source_duration)
        else:
            global_tempo = 1.0

        # Output canvas length. With global_tempo the amix output is retimed
        # by `global_tempo`, so the pre-tempo canvas must be `fit_duration`;
        # after `atempo` it collapses to ~`fit_duration / global_tempo`. With
        # no_tempo we need room for the entire shifted timeline.
        total_duration = fit_duration + 0.5

        log.info("mix: %d dubbed segments, strategy=%s, clip_source=%s(%s), me_stem=%s",
                 len(segs), strategy, clip_source, "+".join(clip_fields), me_stem)
        log.info("mix: fit_dur=%.2fs source_dur=%.2fs global_tempo=%.3f",
                 fit_duration, source_duration, global_tempo)
        for sid, placed, _, _, drift in shift_plan:
            if abs(drift) > 0.01:
                log.info("  %s placed=%.2fs drift=%+0.2fs", sid, placed, drift)

        _ffmpeg_mix(
            plan=shift_plan,
            me_stem=me_stem,
            out_wav=out_wav,
            total_duration=total_duration,
            sample_rate=sample_rate,
            tts_gain_db=tts_gain_db,
            me_gain_db=me_gain_db,
            global_tempo=global_tempo,
        )

    out_mp4: Optional[Path] = None
    if mux_video:
        out_mp4 = out_dir / f"{source_video.stem}.dubbed.mp4"
        _ffmpeg_mux(source_video, out_wav, out_mp4)

    data.setdefault("extra", {})["mix"] = {
        "me_stem": str(me_stem) if me_stem else None,
        "out_wav": str(out_wav),
        "out_mp4": str(out_mp4) if out_mp4 else None,
        "num_mixed": num_mixed,
        "sample_rate": sample_rate,
        "tts_gain_db": tts_gain_db,
        "me_gain_db": me_gain_db,
        "strategy": strategy,
        "tempo_cap": tempo_cap,
        "global_tempo": global_tempo,
        "fit_duration_s": fit_duration,
        "source_duration_s": source_duration,
        "clip_source": clip_source,
        "clip_fields": clip_fields,
        **plan_extra,
    }
    dest = Path(out_json) if out_json else merged_json
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    log.info("mix: wrote %s + %s (%s)", dest, out_wav, out_mp4 or "no video mux")
    return MixResult(
        audio_wav=out_wav, video_mp4=out_mp4,
        num_mixed=num_mixed,
        num_skipped=len(plan_extra.get("skipped", [])),
        sample_rate=sample_rate,
        extra={"strategy": strategy, "global_tempo": global_tempo,
               "fit_duration_s": fit_duration,
               "source_duration_s": source_duration,
               "clip_source": clip_source,
               "clip_fields": clip_fields,
               **plan_extra},
    )


# ---------------------------------------------------------------------------
# ffmpeg plumbing
# ---------------------------------------------------------------------------

def _ffmpeg_mix(
    *,
    plan: List,
    me_stem: Optional[Path],
    out_wav: Path,
    total_duration: float,
    sample_rate: int,
    tts_gain_db: float,
    me_gain_db: float,
    global_tempo: float = 1.0,
) -> None:
    ff = _ffmpeg_bin()
    inputs: List[str] = []
    filters: List[str] = []
    mix_labels: List[str] = []

    # Input 0: a silent base at `total_duration`, so even without an M&E
    # bed we have a canvas every segment sits on.
    inputs += ["-f", "lavfi", "-t", f"{total_duration:.3f}",
               "-i", f"anullsrc=channel_layout=stereo:sample_rate={sample_rate}"]
    mix_labels.append("[0:a]")

    idx = 1
    if me_stem and Path(me_stem).exists():
        inputs += ["-i", str(me_stem)]
        filters.append(
            f"[{idx}:a]aresample={sample_rate},"
            f"volume={me_gain_db}dB[me]"
        )
        mix_labels.append("[me]")
        idx += 1
    else:
        log.warning("mix: no M&E stem; dubbing dialogue onto silence")

    for _sid, placed_s, wav, _raw, _drift in plan:
        inputs += ["-i", str(wav)]
        delay_ms = int(max(0.0, placed_s) * 1000)
        parts = [
            f"[{idx}:a]aresample={sample_rate}",
            "aformat=channel_layouts=stereo",
            f"adelay={delay_ms}|{delay_ms}",
            f"volume={tts_gain_db}dB",
        ]
        filters.append(",".join(parts) + f"[s{idx}]")
        mix_labels.append(f"[s{idx}]")
        idx += 1

    mix_out = "[mix]" if global_tempo != 1.0 else "[out]"
    mix_filter = "".join(mix_labels) + (
        f"amix=inputs={len(mix_labels)}:duration=longest:normalize=0{mix_out}"
    )
    tail_filters: List[str] = []
    if global_tempo != 1.0:
        tail_filters.append(f"[mix]atempo={global_tempo:.4f}[out]")
    filter_graph = ";".join(filters + [mix_filter] + tail_filters)

    cmd = [
        ff, "-y", "-hide_banner", "-loglevel", "error",
        *inputs,
        "-filter_complex", filter_graph,
        "-map", "[out]",
        "-ac", "2", "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(out_wav),
    ]
    log.debug("ffmpeg mix: %s", " ".join(cmd[:12]) + " … ")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg mix failed (rc={proc.returncode}): {proc.stderr.strip()[:800]}"
        )


def _ffmpeg_mix_plan(
    *,
    pieces: List[_PlacedPiece],
    me_stem: Optional[Path],
    out_wav: Path,
    total_duration: float,
    sample_rate: int,
    tts_gain_db: float,
    me_gain_db: float,
    stretch_backend: Optional[str],
) -> None:
    """Render a resolved timing plan: one ffmpeg input and chain per piece.

    One `-i` per PIECE rather than per clip. Two pieces cut from the same
    wav would otherwise have to share a stream, which in ffmpeg means an
    `asplit` whose outputs must all be consumed — a graph that grows a new
    failure mode every time a producer changes how it cuts. A duplicate
    `-i` of the same file costs a second demuxer and nothing else.

    NOTE ON PAUSES: `SegmentPlan.inserted_pauses` needs no filter and no
    silence source here. A pause IS the gap between one piece's `dst_end`
    and the next piece's `dst_start`, and each piece is placed absolutely by
    its own `adelay` onto a silent canvas. Nothing renders the gap because
    nothing has to — that is the payoff of planning in `Piece`s instead of
    durations, and the reason there is no pause-handling code to find below.
    """
    from . import stretch as stretch_mod

    # The PATH ffmpeg here has `atempo` but not `rubberband`, so the binary
    # must be chosen by the filter the graph needs, not the other way round.
    if stretch_backend == "rubberband":
        ff = stretch_mod.ffmpeg_with_filter("rubberband")
        if ff is None:
            raise RuntimeError("mix: no ffmpeg on this host carries rubberband")
    elif stretch_backend == "ffmpeg_atempo":
        ff = stretch_mod.ffmpeg_with_filter("atempo") or _ffmpeg_bin()
    else:
        ff = _ffmpeg_bin()

    inputs: List[str] = []
    filters: List[str] = []
    mix_labels: List[str] = []

    # Input 0: the silent canvas every piece is delayed onto.
    inputs += ["-f", "lavfi", "-t", f"{total_duration:.3f}",
               "-i", f"anullsrc=channel_layout=stereo:sample_rate={sample_rate}"]
    mix_labels.append("[0:a]")

    idx = 1
    if me_stem and Path(me_stem).exists():
        inputs += ["-i", str(me_stem)]
        filters.append(f"[{idx}:a]aresample={sample_rate},volume={me_gain_db}dB[me]")
        mix_labels.append("[me]")
        idx += 1
    else:
        log.warning("mix: no M&E stem; dubbing dialogue onto silence")

    for pc in pieces:
        inputs += ["-i", str(pc.wav)]
        delay_ms = int(round(max(0.0, pc.dst_start) * 1000))
        parts = [
            f"[{idx}:a]atrim=start={pc.src_start:.6f}:end={pc.src_end:.6f}",
            "asetpts=PTS-STARTPTS",
            f"aresample={sample_rate}",
            "aformat=channel_layouts=stereo",
        ]
        if abs(pc.tempo - 1.0) >= _TEMPO_EPS:
            if pc.src_dur < _MIN_STRETCH_INPUT_S:
                log.warning("mix: %s piece at %.3fs is only %.3fs; rendering at "
                            "natural pace instead of %.3fx (the filter would "
                            "drop it)", pc.sid, pc.dst_start, pc.src_dur, pc.tempo)
            else:
                parts.append(stretch_mod.ffmpeg_stretch_filter(
                    pc.tempo, backend=stretch_backend))
        parts.append(f"adelay={delay_ms}|{delay_ms}")
        parts.append(f"volume={tts_gain_db}dB")
        filters.append(",".join(parts) + f"[p{idx}]")
        mix_labels.append(f"[p{idx}]")
        idx += 1

    mix_filter = "".join(mix_labels) + (
        f"amix=inputs={len(mix_labels)}:duration=longest:normalize=0[out]"
    )
    filter_graph = ";".join(filters + [mix_filter])

    cmd = [
        ff, "-y", "-hide_banner", "-loglevel", "error",
        *inputs,
        "-filter_complex", filter_graph,
        "-map", "[out]",
        "-ac", "2", "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(out_wav),
    ]
    log.debug("ffmpeg mix_plan: %d inputs, %d filters", len(pieces), len(filters))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg timing-plan mix failed (rc={proc.returncode}): "
            f"{proc.stderr.strip()[:800]}"
        )


def _ffmpeg_mux(video: Path, audio_wav: Path, out_mp4: Path) -> None:
    ff = _ffmpeg_bin()
    cmd = [
        ff, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video), "-i", str(audio_wav),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(out_mp4),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg mux failed (rc={proc.returncode}): {proc.stderr.strip()[:800]}"
        )
