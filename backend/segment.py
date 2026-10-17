"""Logical segmentation: refine raw diarization turns into TTS-friendly segments.

Diarization models detect *speaker changes*, not linguistic boundaries.  A long
monologue becomes one giant turn; rapid crosstalk produces many tiny turns.
This module sits between diarization and ASR and applies three passes:

  1. **Split** long same-speaker turns at internal silence gaps detected by a
     lightweight VAD (silero-vad).
  2. **Merge** short consecutive same-speaker turns separated by a small gap.
  3. **Cap** any remaining segment that still exceeds ``max_duration_s`` by
     splitting it at the deepest energy valley.

The output is a new ``DiarizationResult`` with the same ``Turn`` type so every
downstream stage is unaware that refinement happened.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from models.diarization.base import DiarizationResult, Turn

log = logging.getLogger(__name__)

# ── defaults ─────────────────────────────────────────────────────────────────
SPLIT_MIN_DURATION_S = 8.0     # only split turns longer than this
SPLIT_PAUSE_S = 0.15           # silence gap inside a turn to split on
MERGE_MAX_GAP_S = 0.50         # merge same-speaker turns closer than this
MERGE_MAX_RESULT_S = 15.0      # … but don't let the merged turn exceed this
MAX_DURATION_S = 15.0          # hard cap — force-split anything longer
MIN_DURATION_S = 0.3           # drop turns shorter than this after all passes


# ── VAD helper ───────────────────────────────────────────────────────────────

_vad_model = None
_vad_utils = None


def _ensure_vad():
    global _vad_model, _vad_utils
    if _vad_model is not None:
        return
    _vad_model, _vad_utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        onnx=False,
        trust_repo=True,
    )
    log.info("segment: silero-vad loaded")


def _vad_speech_timestamps(
    audio: np.ndarray, sr: int = 16_000
) -> List[Tuple[float, float]]:
    """Return speech regions as ``[(start_s, end_s), ...]``."""
    _ensure_vad()
    get_speech_timestamps = _vad_utils[0]
    tensor = torch.from_numpy(audio).float()
    if tensor.ndim > 1:
        tensor = tensor.mean(dim=0)
    stamps = get_speech_timestamps(tensor, _vad_model, sampling_rate=sr)
    return [(s["start"] / sr, s["end"] / sr) for s in stamps]


# ── core passes ──────────────────────────────────────────────────────────────

def _load_audio(path: Path, sr: int = 16_000) -> np.ndarray:
    """Load a mono WAV into a float32 numpy array."""
    import soundfile as sf
    audio, file_sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_sr != sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
    return audio


def _split_long_turns(
    turns: List[Turn],
    audio: np.ndarray,
    sr: int = 16_000,
    *,
    min_duration_s: float = SPLIT_MIN_DURATION_S,
    pause_s: float = SPLIT_PAUSE_S,
) -> List[Turn]:
    """Split turns longer than *min_duration_s* at internal silences."""
    out: List[Turn] = []
    speech_regions = _vad_speech_timestamps(audio, sr)

    for turn in turns:
        if turn.duration < min_duration_s:
            out.append(turn)
            continue

        regions_in_turn = [
            (max(r[0], turn.start), min(r[1], turn.end))
            for r in speech_regions
            if r[1] > turn.start and r[0] < turn.end
        ]
        regions_in_turn = [(s, e) for s, e in regions_in_turn if e - s > 0.05]

        if len(regions_in_turn) < 2:
            out.append(turn)
            continue

        # find gaps between consecutive speech regions
        gaps: List[Tuple[float, float]] = []
        for i in range(len(regions_in_turn) - 1):
            gap_start = regions_in_turn[i][1]
            gap_end = regions_in_turn[i + 1][0]
            if gap_end - gap_start >= pause_s:
                gaps.append((gap_start, gap_end))

        if not gaps:
            out.append(turn)
            continue

        # split at each qualifying gap
        prev_start = turn.start
        for gap_start, gap_end in gaps:
            mid = (gap_start + gap_end) / 2
            out.append(Turn(start=prev_start, end=mid, speaker=turn.speaker))
            prev_start = mid
        out.append(Turn(start=prev_start, end=turn.end, speaker=turn.speaker))

    return out


def _merge_short_turns(
    turns: List[Turn],
    *,
    max_gap_s: float = MERGE_MAX_GAP_S,
    max_result_s: float = MERGE_MAX_RESULT_S,
) -> List[Turn]:
    """Merge consecutive same-speaker turns separated by ≤ *max_gap_s*."""
    if not turns:
        return []
    out: List[Turn] = [turns[0]]
    for t in turns[1:]:
        prev = out[-1]
        gap = t.start - prev.end
        same_speaker = t.speaker == prev.speaker
        merged_dur = t.end - prev.start
        if same_speaker and 0 <= gap <= max_gap_s and merged_dur <= max_result_s:
            out[-1] = Turn(start=prev.start, end=t.end, speaker=prev.speaker)
        else:
            out.append(t)
    return out


def _cap_duration(
    turns: List[Turn],
    audio: np.ndarray,
    sr: int = 16_000,
    *,
    max_duration_s: float = MAX_DURATION_S,
) -> List[Turn]:
    """Force-split any turn still exceeding *max_duration_s*.

    Strategy: prefer splitting at the deepest VAD gap inside the turn. If no
    gap is found (pure continuous speech), fall back to the lowest-energy
    50 ms window.
    """
    speech_regions = _vad_speech_timestamps(audio, sr)
    out: List[Turn] = []

    for turn in turns:
        if turn.duration <= max_duration_s:
            out.append(turn)
            continue

        # collect all gaps between VAD speech regions inside this turn
        regions_in = [
            (max(r[0], turn.start), min(r[1], turn.end))
            for r in speech_regions
            if r[1] > turn.start and r[0] < turn.end
        ]
        regions_in = [(s, e) for s, e in regions_in if e - s > 0.02]

        gaps: List[Tuple[float, float]] = []
        for i in range(len(regions_in) - 1):
            gs = regions_in[i][1]
            ge = regions_in[i + 1][0]
            if ge > gs + 0.02:
                gaps.append((gs, ge))

        # greedily split: walk through the turn, and every time the accumulated
        # duration would exceed max_duration_s, cut at the best available gap
        result_turns: List[Turn] = []
        seg_start = turn.start

        while turn.end - seg_start > max_duration_s:
            target = seg_start + max_duration_s
            # find the gap whose midpoint is closest to `target`
            best_gap = None
            best_dist = float("inf")
            for gs, ge in gaps:
                mid = (gs + ge) / 2
                if mid <= seg_start + 1.0:
                    continue
                dist = abs(mid - target)
                if dist < best_dist:
                    best_dist = dist
                    best_gap = (gs, ge)

            if best_gap is not None:
                cut = (best_gap[0] + best_gap[1]) / 2
            else:
                # fallback: lowest-energy 50ms window
                s0 = int(round(seg_start * sr))
                s1 = int(round(min(seg_start + max_duration_s + 2, turn.end) * sr))
                clip = audio[s0:s1]
                win = int(0.05 * sr)
                if len(clip) > win * 2:
                    energy = np.array([
                        np.sqrt(np.mean(clip[i:i + win] ** 2))
                        for i in range(0, len(clip) - win, win)
                    ])
                    idx = int(np.argmin(energy))
                    cut = seg_start + (idx * win / sr)
                else:
                    cut = seg_start + max_duration_s

            if cut <= seg_start + 0.5:
                cut = seg_start + max_duration_s

            result_turns.append(Turn(start=seg_start, end=cut, speaker=turn.speaker))
            seg_start = cut

        result_turns.append(Turn(start=seg_start, end=turn.end, speaker=turn.speaker))
        out.extend(result_turns)

    return out


def _drop_tiny(turns: List[Turn], *, min_s: float = MIN_DURATION_S) -> List[Turn]:
    """Drop turns shorter than *min_s*."""
    return [t for t in turns if t.duration >= min_s]


# ── public API ───────────────────────────────────────────────────────────────

def resegment(
    diar_result: DiarizationResult,
    audio_path: Optional[Path] = None,
    *,
    split_min_duration_s: float = SPLIT_MIN_DURATION_S,
    split_pause_s: float = SPLIT_PAUSE_S,
    merge_max_gap_s: float = MERGE_MAX_GAP_S,
    merge_max_result_s: float = MERGE_MAX_RESULT_S,
    max_duration_s: float = MAX_DURATION_S,
    min_duration_s: float = MIN_DURATION_S,
) -> DiarizationResult:
    """Refine diarization turns into better segments for downstream TTS.

    Parameters
    ----------
    diar_result : DiarizationResult
        Raw diarization output.
    audio_path : Path, optional
        Mono 16 kHz WAV.  Falls back to ``diar_result.audio_path``.
    split_min_duration_s : float
        Only split turns longer than this.
    split_pause_s : float
        Minimum internal silence gap to split on.
    merge_max_gap_s : float
        Merge same-speaker turns closer than this.
    merge_max_result_s : float
        Don't merge if the result would be longer than this.
    max_duration_s : float
        Hard cap — force-split anything longer.
    min_duration_s : float
        Drop turns shorter than this after all passes.

    Returns
    -------
    DiarizationResult
        New result with refined turns. ``extra["resegment"]`` holds stats.
    """
    ap = Path(audio_path) if audio_path else diar_result.audio_path
    sr = 16_000
    audio = _load_audio(ap, sr)
    orig_count = len(diar_result.turns)

    turns = list(diar_result.turns)
    turns.sort(key=lambda t: (t.start, t.end))

    log.info("resegment: input %d turns", len(turns))

    # pass 1: split long turns at internal silences
    turns = _split_long_turns(
        turns, audio, sr,
        min_duration_s=split_min_duration_s,
        pause_s=split_pause_s,
    )
    after_split = len(turns)
    log.info("resegment: after split → %d turns", after_split)

    # pass 2: merge short same-speaker turns
    turns = _merge_short_turns(
        turns,
        max_gap_s=merge_max_gap_s,
        max_result_s=merge_max_result_s,
    )
    after_merge = len(turns)
    log.info("resegment: after merge → %d turns", after_merge)

    # pass 3: hard cap
    turns = _cap_duration(turns, audio, sr, max_duration_s=max_duration_s)
    after_cap = len(turns)
    log.info("resegment: after cap   → %d turns", after_cap)

    # pass 4: drop tiny
    turns = _drop_tiny(turns, min_s=min_duration_s)
    log.info("resegment: after drop  → %d turns (was %d)", len(turns), orig_count)

    stats = {
        "original_turns": orig_count,
        "after_split": after_split,
        "after_merge": after_merge,
        "after_cap": after_cap,
        "final_turns": len(turns),
        "params": {
            "split_min_duration_s": split_min_duration_s,
            "split_pause_s": split_pause_s,
            "merge_max_gap_s": merge_max_gap_s,
            "merge_max_result_s": merge_max_result_s,
            "max_duration_s": max_duration_s,
            "min_duration_s": min_duration_s,
        },
    }

    durations = [t.duration for t in turns]
    if durations:
        stats["duration_stats"] = {
            "min": round(min(durations), 2),
            "max": round(max(durations), 2),
            "mean": round(sum(durations) / len(durations), 2),
            "median": round(float(np.median(durations)), 2),
        }

    extra = dict(diar_result.extra)
    extra["resegment"] = stats

    return DiarizationResult(
        turns=turns,
        audio_path=diar_result.audio_path,
        model_name=diar_result.model_name,
        extra=extra,
    )
