"""Duration-aware text adjustment: rank translation candidates by fit.

Indic translations run 1.1-1.4x longer than English and the MT engine has no
length control, so candidates are ranked by a fitted linear duration model,
the winner is synthesised, and the residual is WSOLA-stretched under a +-10%
cap.

Stops short of synthesis: writes `candidates[]`, `est_duration_s`,
`target_duration_s` and the winning `final_text` per segment for the TTS
stage to pick up.

The model is `duration ~= a*chars + b*words + c` per (language, voice), with
coefficients under `data/duration_model/<lang>.<voice>.json`.
`fit_duration_model` builds one from a real TTS calibration run.

This is the RANKER, not the generator -- upstream fills `candidates[]` with
alternative translations and this scores what the manifest already holds.
"""
from __future__ import annotations

import glob
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# fitted linear duration model
# ---------------------------------------------------------------------------

@dataclass
class DurationModel:
    """`duration_s ≈ a·chars + b·words + c`."""
    a_chars: float
    b_words: float
    c_bias: float
    language: str = ""
    voice: str = ""
    n_fit: int = 0
    mae_s: float = 0.0

    def predict(self, text: str) -> float:
        n_chars, n_words = _text_shape(text)
        return max(0.1, self.a_chars * n_chars + self.b_words * n_words + self.c_bias)

    def to_jsonable(self) -> dict:
        return {
            "a_chars": self.a_chars, "b_words": self.b_words, "c_bias": self.c_bias,
            "language": self.language, "voice": self.voice,
            "n_fit": self.n_fit, "mae_s": self.mae_s,
        }

    @classmethod
    def from_jsonable(cls, d: dict) -> "DurationModel":
        return cls(
            a_chars=float(d["a_chars"]),
            b_words=float(d["b_words"]),
            c_bias=float(d["c_bias"]),
            language=d.get("language", ""),
            voice=d.get("voice", ""),
            n_fit=int(d.get("n_fit", 0)),
            mae_s=float(d.get("mae_s", 0.0)),
        )


# Fallback for an uncalibrated (language, voice): Indic-Speak at ~15 chars/s
# and ~2.5 words/s, plus a 0.1 s pad.
DEFAULT_MODEL = DurationModel(
    a_chars=1.0 / 15.0, b_words=1.0 / 2.5 * 0.1, c_bias=0.10,
    language="default", voice="default", n_fit=0, mae_s=0.0,
)


def _text_shape(text: str) -> Tuple[int, int]:
    t = (text or "").strip()
    n_chars = len(t)
    n_words = len([w for w in re.split(r"\s+", t) if w])
    return n_chars, n_words


def fit_duration_model(
    samples: List[Tuple[str, float]],
    *,
    language: str = "",
    voice: str = "",
) -> DurationModel:
    """Least-squares fit over (text, measured_duration_s) pairs.

    Needs ~200 utterances for a stable fit; works on fewer, and the MAE field
    says how far to trust it.
    """
    import numpy as np
    if len(samples) < 3:
        raise ValueError(f"need at least 3 samples to fit, got {len(samples)}")
    X = np.array([[*_text_shape(t), 1.0] for t, _ in samples], dtype=np.float64)
    y = np.array([d for _, d in samples], dtype=np.float64)
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b, c = (float(v) for v in coeffs.tolist())
    preds = X @ coeffs
    mae = float(np.mean(np.abs(preds - y)))
    model = DurationModel(
        a_chars=a, b_words=b, c_bias=c,
        language=language, voice=voice,
        n_fit=len(samples), mae_s=mae,
    )
    log.info("duration fit: lang=%s voice=%s n=%d → a=%.4f b=%.4f c=%.3f MAE=%.3fs",
             language, voice, len(samples), a, b, c, mae)
    return model


def load_or_default_model(
    cache_dir: Path, *, language: str, voice: str,
) -> DurationModel:
    """Pick `<cache_dir>/<lang>.<voice>.json` if it exists, else default."""
    p = Path(cache_dir) / f"{language}.{voice}.json"
    if p.exists():
        try:
            return DurationModel.from_jsonable(json.loads(p.read_text()))
        except Exception as e:
            log.warning("duration model %s unreadable (%s); using default", p, e)
    log.info("duration model: using DEFAULT for %s/%s (fit one for better MAE)",
             language, voice)
    return DEFAULT_MODEL


# ---------------------------------------------------------------------------
# candidate ranking
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    text: str
    est_duration_s: float
    delta_s: float               # est - target (positive = too long)
    chrf: Optional[float] = None
    term_penalty: float = 0.0
    source: str = "mt"            # "mt" | "path_b_more_retain" | "prompted_short" | …
    score: float = 0.0            # combined — lower is better
    extra: dict = field(default_factory=dict)


# A quality scorer: (candidate_text, reference_text) -> 0..1, higher is better.
QualityFn = Callable[[str, str], float]


def sacrebleu_chrf(cand: str, ref: str) -> float:
    """Real chrF (sacrebleu, chrF++ word order 2), scaled to 0..1.

    Falls back to the `_chrf_quick` stand-in when sacrebleu is absent, so the
    caller never has to branch.
    """
    if not cand or not ref:
        return 0.0
    try:
        from sacrebleu.metrics import CHRF
    except ImportError:
        log.info("sacrebleu not installed; falling back to _chrf_quick")
        return _chrf_quick(cand, ref)
    global _CHRF_METRIC
    if _CHRF_METRIC is None:
        _CHRF_METRIC = CHRF(word_order=2)
    return float(_CHRF_METRIC.sentence_score(cand, [ref]).score) / 100.0


_CHRF_METRIC = None


def _chrf_quick(cand: str, ref: str) -> float:
    """Cheap character n-gram overlap, 0..1.

    Stands in for `sacrebleu_chrf` in a tight inner loop.
    """
    if not cand or not ref:
        return 0.0
    cs = {cand[i:i+3] for i in range(max(0, len(cand) - 2))}
    rs = {ref[i:i+3] for i in range(max(0, len(ref) - 2))}
    if not cs or not rs:
        return 0.0
    inter = len(cs & rs)
    return 2 * inter / (len(cs) + len(rs))


def rank_candidates(
    candidates: List[str],
    *,
    target_duration_s: float,
    model: DurationModel,
    reference_text: Optional[str] = None,
    retain_terms: Optional[List[str]] = None,
    w_duration: float = 1.0,
    w_quality: float = 0.5,
    w_terms: float = 1.5,
    quality_fn: Optional[QualityFn] = None,
) -> List[Candidate]:
    """Rank `candidates` (same source segment, different wordings).

    `score = w_duration * |delta|/target + w_quality * (1 - quality)
           + w_terms * (missing_terms / expected_terms)`

    Lower is better; returns candidates sorted ascending.

    `quality_fn(candidate, reference_text) -> 0..1` overrides `_chrf_quick`.
    What `reference_text` is matters: against the beam-1 output the quality
    term is a pure DEVIATION penalty, punishing every alternative for not
    being beam-1. Passing the SOURCE text with `quality_fn=sacrebleu_chrf`
    scores adequacy instead. `plan_segment_durations` defaults to beam-1;
    this hook is how a caller opts out.
    """
    out: List[Candidate] = []
    retain_lc = [t.lower() for t in (retain_terms or []) if t]
    expected = max(1, len(retain_lc))
    for raw in candidates:
        text = (raw or "").strip()
        if not text:
            continue
        est = model.predict(text)
        delta = est - max(target_duration_s, 0.01)
        scorer = quality_fn or _chrf_quick
        chrf = scorer(text, reference_text) if reference_text else 1.0
        if retain_lc:
            low = text.lower()
            missed = sum(1 for t in retain_lc if t not in low)
            term_pen = missed / expected
        else:
            term_pen = 0.0
        score = (
            w_duration * abs(delta) / max(target_duration_s, 0.5)
            + w_quality * (1.0 - chrf)
            + w_terms * term_pen
        )
        out.append(Candidate(
            text=text, est_duration_s=est, delta_s=delta,
            chrf=chrf, term_penalty=term_pen, score=score,
        ))
    out.sort(key=lambda c: c.score)
    return out


# ---------------------------------------------------------------------------
# segment-level driver
# ---------------------------------------------------------------------------

def plan_segment_durations(
    merged_json: Path,
    *,
    language: str,
    voice: str = "default",
    cache_dir: Optional[Path] = None,
    safety_margin_s: float = 0.10,
    out_json: Optional[Path] = None,
) -> dict:
    """Fill `target_duration_s`, `est_duration_s` and `candidates[]` on every
    segment of the merged ASR+MT+codemix JSON.

    A segment that already carries `candidates[]` has them ranked; otherwise
    `final_text` alone is scored and recorded as the winner.
    """
    merged_json = Path(merged_json)
    data = json.loads(merged_json.read_text())

    model = load_or_default_model(
        cache_dir or (merged_json.parent.parent / "duration_model"),
        language=language, voice=voice,
    )

    over = 0
    for seg in data.get("segments", []):
        s = float(seg.get("start", seg.get("start_s", 0.0)))
        e = float(seg.get("end", seg.get("end_s", 0.0)))
        target = max(0.01, (e - s) - safety_margin_s)
        base = seg.get("final_text") or seg.get("mt_text") or seg.get("text") or ""
        alternates: List[str] = [c for c in seg.get("candidates_text", []) if c]
        pool = [base] + [a for a in alternates if a != base]
        retain = [sp["text"] for sp in seg.get("retain_spans", []) if sp.get("text")]
        ranked = rank_candidates(
            pool,
            target_duration_s=target,
            model=model,
            reference_text=base,
            retain_terms=retain,
        )
        if not ranked:
            seg["target_duration_s"] = target
            continue
        winner = ranked[0]
        seg["target_duration_s"] = target
        seg["est_duration_s"] = winner.est_duration_s
        seg["final_text"] = winner.text
        seg["candidates"] = [
            {
                "text": c.text, "est_duration_s": c.est_duration_s,
                "delta_s": c.delta_s, "chrf": c.chrf,
                "term_penalty": c.term_penalty, "score": c.score,
            }
            for c in ranked
        ]
        residual = abs(winner.est_duration_s - target) / max(target, 0.5)
        if residual > 0.10:
            over += 1
            seg.setdefault("qc", {})["budget_violation_estimated"] = True

    data.setdefault("extra", {})["duration"] = {
        "model": model.to_jsonable(),
        "language": language, "voice": voice,
        "safety_margin_s": safety_margin_s,
        "estimated_violations": over,
        "segments": len(data.get("segments", [])),
    }

    dest = Path(out_json) if out_json else merged_json
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    log.info("duration: wrote %s (estimated >10%% violations: %d/%d)",
             dest, over, len(data.get("segments", [])))
    return data


# ---------------------------------------------------------------------------
# post-synthesis stretch (called by TTS stage after audio exists)
# ---------------------------------------------------------------------------

def wsola_stretch_cap(
    wav_in: Path, wav_out: Path, *,
    target_duration_s: float,
    max_stretch: float = 0.10,
    backend: str = "auto",
) -> Tuple[float, bool]:
    """Stretch `wav_in` to `target_duration_s`, capped at +-`max_stretch`.

    Returns (applied_factor, budget_violation). A shim: `backend.tts` unpacks
    this exact tuple, while the real work -- and the record of which backend
    ran, which this signature cannot carry -- is in `backend.stretch`.
    """
    from .stretch import stretch_to_duration
    res = stretch_to_duration(
        wav_in, wav_out,
        target_duration_s=target_duration_s,
        max_stretch=max_stretch,
        backend=backend,
    )
    return res.applied_factor, res.capped


# ---------------------------------------------------------------------------
# calibration from a completed TTS run
# ---------------------------------------------------------------------------

def _iter_merged_paths(merged_json) -> List[Path]:
    items = merged_json if isinstance(merged_json, (list, tuple)) else [merged_json]
    out: List[Path] = []
    for item in items:
        s = str(item)
        # Quoted globs arrive unexpanded by design, so expand them here.
        if any(ch in s for ch in "*?["):
            out.extend(sorted(Path(p) for p in glob.glob(s)))
        else:
            out.append(Path(s))
    return out


def collect_duration_samples(
    merged_json,
    *,
    language: str,
    text_field: str = "final_text",
    audio_field: str = "tts_raw_audio",
) -> List[Tuple[str, float]]:
    """Pair each segment's text with the measured length of its raw TTS wav.

    Raw, not stretched: a stretched copy has been forced toward its target,
    so fitting on it teaches the model the cap, not the voice's pace.
    """
    import soundfile as sf
    samples: List[Tuple[str, float]] = []
    for path in _iter_merged_paths(merged_json):
        if not path.exists():
            log.warning("calibrate: %s does not exist, skipping", path)
            continue
        data = json.loads(path.read_text())
        file_lang = (data.get("extra", {}) or {}).get("mt_target_language")
        if file_lang and file_lang != language:
            log.info("calibrate: %s is %s, not %s — skipping", path.name, file_lang, language)
            continue
        n_before = len(samples)
        for seg in data.get("segments", []):
            wav = seg.get(audio_field)
            if not wav or not Path(wav).exists():
                continue
            # Manifests predating the duration stage carry no `final_text`,
            # so fall back the same way tts.py does to find what was spoken.
            text = seg.get(text_field) or seg.get("mt_text") or seg.get("text") or ""
            if not text.strip():
                continue
            samples.append((text, float(sf.info(str(wav)).duration)))
        log.info("calibrate: %s → %d samples", path.name, len(samples) - n_before)
    return samples


def calibrate_duration_model(
    merged_json,
    *,
    language: str,
    voice: str = "default",
    cache_dir: Path,
    text_field: str = "final_text",
    audio_field: str = "tts_raw_audio",
    min_samples: int = 30,
    out_json: Optional[Path] = None,
) -> DurationModel:
    """Fit and persist `<cache_dir>/<language>.<voice>.json` from real TTS output.

    `merged_json` may be one path, a list of paths, or a glob string.
    """
    samples = collect_duration_samples(
        merged_json, language=language,
        text_field=text_field, audio_field=audio_field,
    )
    if len(samples) < min_samples:
        raise ValueError(
            f"only {len(samples)} (text, duration) pairs found for "
            f"{language}/{voice}; need {min_samples}. Synthesise more segments "
            f"or lower --min-samples (the fit will be correspondingly noisy)."
        )
    model = fit_duration_model(samples, language=language, voice=voice)
    dest = Path(out_json) if out_json else Path(cache_dir) / f"{language}.{voice}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(model.to_jsonable(), indent=2))
    log.info("calibrate: wrote %s", dest)
    return model


def _mae(model: DurationModel, samples: List[Tuple[str, float]]) -> float:
    return sum(abs(model.predict(t) - d) for t, d in samples) / max(1, len(samples))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    """Fit a per-(language, voice) duration model from a finished TTS run.

    Example:
        python -m backend.duration fit \
            --merged 'data/mt/*.mt.json' --language hi --voice Kavya \
            --cache-dir data/duration_model
    """
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m backend.duration",
        description="Duration model utilities.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="Fit a duration model from merged manifests "
                                   "that already carry TTS audio.")
    f.add_argument("--merged", nargs="+", required=True,
                   help="Merged MT JSON path(s); globs are expanded here, so "
                        "quote them to stop the shell eating them.")
    f.add_argument("--language", required=True, help="Target language, e.g. hi.")
    f.add_argument("--voice", default="default", help="Voice/model id label.")
    f.add_argument("--cache-dir", type=Path, default=Path("data/duration_model"),
                   help="Writes <cache-dir>/<language>.<voice>.json.")
    f.add_argument("--text-field", default="final_text")
    f.add_argument("--audio-field", default="tts_raw_audio")
    f.add_argument("--min-samples", type=int, default=30)
    f.add_argument("--out-json", type=Path, default=None,
                   help="Override the output path entirely.")

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

    if args.cmd != "fit":
        ap.error(f"unknown command {args.cmd}")

    samples = collect_duration_samples(
        args.merged, language=args.language,
        text_field=args.text_field, audio_field=args.audio_field,
    )
    if len(samples) < args.min_samples:
        print(f"only {len(samples)} (text, duration) pairs found for "
              f"{args.language}/{args.voice}; need {args.min_samples}. "
              f"No model written.")
        return 1

    model = calibrate_duration_model(
        args.merged, language=args.language, voice=args.voice,
        cache_dir=args.cache_dir, text_field=args.text_field,
        audio_field=args.audio_field, min_samples=args.min_samples,
        out_json=args.out_json,
    )
    dest = args.out_json or (args.cache_dir / f"{args.language}.{args.voice}.json")
    print(f"n_fit         : {model.n_fit}")
    print(f"a_chars       : {model.a_chars:.6f} s/char")
    print(f"b_words       : {model.b_words:.6f} s/word")
    print(f"c_bias        : {model.c_bias:.6f} s")
    print(f"MAE (fitted)  : {model.mae_s:.3f} s")
    print(f"MAE (DEFAULT) : {_mae(DEFAULT_MODEL, samples):.3f} s  "
          f"(same {len(samples)} samples)")
    print(f"wrote         : {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
