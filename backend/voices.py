"""Per-speaker TTS voice assignment: gender filter, then ECAPA timbre match.

Stage 9d calls `assign_voices` and writes `SpeakerVoice[]` to
`<job>/voices/<stem>.<voice_engine>.voices.json` only when `voice_engine` is
set; the default Edge/IndicF5 path builds `speaker_voice_map` in memory and
persists nothing. Either way the map reaches
`backend.tts.synthesize_segments`, and the voice that spoke is recorded per
segment as `tts_voice`.

`extract_speaker_references` writes reference clips to
`<job>/voices/<stem>.refs/<SPEAKER_ID>.wav`.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# voice catalog
# ---------------------------------------------------------------------------

@dataclass
class Voice:
    """One row from a TTS engine's voice catalog."""
    engine: str                                   # "indic-speak" | "indicf5"
    voice: str                                    # engine-specific voice id
    gender: Optional[str] = None                  # "male" | "female" | "neutral"
    age: Optional[str] = None                     # "child" | "young" | "adult" | "elder"
    language: Optional[str] = None                # ISO-639 code
    reference_audio: Optional[Path] = None        # for embedding similarity
    tier: str = "support"                         # "lead" | "support"
    extra: dict = field(default_factory=dict)


# Bodhan indic-speak / ai4bharat indic-parler-tts voice catalog.
# 44 voices: 1 female + 1 male per each of the 22 scheduled Indian languages.
# Source: bodhan-ai/indic-speak voices.md (trained on RASA + IndicVoices-R).
# The JSON file `data/voices/indic-speak.json` overrides this if present.
DEFAULT_INDIC_SPEAK = [
    # Assamese (as)
    Voice(engine="indic-speak", voice="Prastuti",  gender="female", age="adult", tier="lead", language="as"),
    Voice(engine="indic-speak", voice="Ankur",     gender="male",   age="adult", tier="lead", language="as"),
    # Bengali (bn)
    Voice(engine="indic-speak", voice="Ishita",    gender="female", age="adult", tier="lead", language="bn"),
    Voice(engine="indic-speak", voice="Sourav",    gender="male",   age="adult", tier="lead", language="bn"),
    # Bodo (brx)
    Voice(engine="indic-speak", voice="Gwrbw",     gender="female", age="adult", tier="lead", language="brx"),
    Voice(engine="indic-speak", voice="Sansuma",   gender="male",   age="adult", tier="lead", language="brx"),
    # Dogri (doi)
    Voice(engine="indic-speak", voice="Preeti",    gender="female", age="adult", tier="lead", language="doi"),
    Voice(engine="indic-speak", voice="Sham",      gender="male",   age="adult", tier="lead", language="doi"),
    # Gujarati (gu)
    Voice(engine="indic-speak", voice="Dhara",     gender="female", age="adult", tier="lead", language="gu"),
    Voice(engine="indic-speak", voice="Parth",     gender="male",   age="adult", tier="lead", language="gu"),
    # Hindi (hi)
    Voice(engine="indic-speak", voice="Kavya",     gender="female", age="adult", tier="lead", language="hi"),
    Voice(engine="indic-speak", voice="Suhani",    gender="female", age="adult", tier="support", language="hi"),
    Voice(engine="indic-speak", voice="Amit",      gender="male",   age="adult", tier="lead", language="hi"),
    # Kannada (kn)
    Voice(engine="indic-speak", voice="Deepika",   gender="female", age="adult", tier="lead", language="kn"),
    Voice(engine="indic-speak", voice="Adarsh",    gender="male",   age="adult", tier="lead", language="kn"),
    # Kashmiri (ks)
    Voice(engine="indic-speak", voice="Zoon",      gender="female", age="adult", tier="lead", language="ks"),
    Voice(engine="indic-speak", voice="Ishfaq",    gender="male",   age="adult", tier="lead", language="ks"),
    # Konkani (kok)
    Voice(engine="indic-speak", voice="Anjali",    gender="female", age="adult", tier="lead", language="kok"),
    Voice(engine="indic-speak", voice="Sandeep",   gender="male",   age="adult", tier="lead", language="kok"),
    # Maithili (mai)
    Voice(engine="indic-speak", voice="Vaidehi",   gender="female", age="adult", tier="lead", language="mai"),
    Voice(engine="indic-speak", voice="Madhukar",  gender="male",   age="adult", tier="lead", language="mai"),
    # Malayalam (ml)
    Voice(engine="indic-speak", voice="Lakshmi",   gender="female", age="adult", tier="lead", language="ml"),
    Voice(engine="indic-speak", voice="Kiran",     gender="male",   age="adult", tier="lead", language="ml"),
    # Manipuri (mni)
    Voice(engine="indic-speak", voice="Thoibi",    gender="female", age="adult", tier="lead", language="mni"),
    Voice(engine="indic-speak", voice="Chaoba",    gender="male",   age="adult", tier="lead", language="mni"),
    # Marathi (mr)
    Voice(engine="indic-speak", voice="Anagha",    gender="female", age="adult", tier="lead", language="mr"),
    Voice(engine="indic-speak", voice="Chinmay",   gender="male",   age="adult", tier="lead", language="mr"),
    # Nepali (ne)
    Voice(engine="indic-speak", voice="Srijana",   gender="female", age="adult", tier="lead", language="ne"),
    Voice(engine="indic-speak", voice="Sagar",     gender="male",   age="adult", tier="lead", language="ne"),
    # Odia (or)
    Voice(engine="indic-speak", voice="Itishree",  gender="female", age="adult", tier="lead", language="or"),
    Voice(engine="indic-speak", voice="Akash",     gender="male",   age="adult", tier="lead", language="or"),
    # Punjabi (pa)
    Voice(engine="indic-speak", voice="Kaur",      gender="female", age="adult", tier="lead", language="pa"),
    Voice(engine="indic-speak", voice="Manpreet",  gender="male",   age="adult", tier="lead", language="pa"),
    # Sanskrit (sa)
    Voice(engine="indic-speak", voice="Bharati",   gender="female", age="adult", tier="lead", language="sa"),
    Voice(engine="indic-speak", voice="Aryaman",   gender="male",   age="adult", tier="lead", language="sa"),
    # Santali (sat)
    Voice(engine="indic-speak", voice="Phulmani",  gender="female", age="adult", tier="lead", language="sat"),
    Voice(engine="indic-speak", voice="Sibu",      gender="male",   age="adult", tier="lead", language="sat"),
    # Sindhi (sd)
    Voice(engine="indic-speak", voice="Moomal",    gender="female", age="adult", tier="lead", language="sd"),
    Voice(engine="indic-speak", voice="Rano",      gender="male",   age="adult", tier="lead", language="sd"),
    # Tamil (ta)
    Voice(engine="indic-speak", voice="Anitha",    gender="female", age="adult", tier="lead", language="ta"),
    Voice(engine="indic-speak", voice="Arun",      gender="male",   age="adult", tier="lead", language="ta"),
    # Telugu (te)
    Voice(engine="indic-speak", voice="Sravani",   gender="female", age="adult", tier="lead", language="te"),
    Voice(engine="indic-speak", voice="Vamsi",     gender="male",   age="adult", tier="lead", language="te"),
    # Urdu (ur)
    Voice(engine="indic-speak", voice="Saba",      gender="female", age="adult", tier="lead", language="ur"),
    Voice(engine="indic-speak", voice="Zaid",      gender="male",   age="adult", tier="lead", language="ur"),
]


def load_voice_catalog(
    engine: str, *,
    language: Optional[str] = None,
    catalog_json: Optional[Path] = None,
) -> List[Voice]:
    """Load the voice catalog for `engine`, same-language voices first.

    Any indic-speak voice can read any language, so cross-language voices
    follow rather than being dropped. Falls back to the hard-coded defaults
    when `catalog_json` is absent.
    """
    if catalog_json and Path(catalog_json).exists():
        rows = json.loads(Path(catalog_json).read_text())
        voices = [
            Voice(
                engine=r.get("engine", engine),
                voice=r["voice"],
                gender=r.get("gender"), age=r.get("age"),
                language=r.get("language"),
                reference_audio=Path(r["reference_audio"]) if r.get("reference_audio") else None,
                tier=r.get("tier", "support"),
                extra=r.get("extra", {}),
            )
            for r in rows
            if r.get("engine", engine).lower() == engine.lower()
        ]
    else:
        if engine.lower() in ("indic-speak", "bodhan"):
            voices = list(DEFAULT_INDIC_SPEAK)
        else:
            voices = []
    attach_voice_references(voices)
    if language:
        prim = language.split("-")[0].lower()
        native = [v for v in voices if (v.language or "").lower() == prim]
        cross = [v for v in voices if (v.language or "").lower() != prim]
        voices = native + cross
    return voices


#: Where `scripts/build_voice_refs.py` writes one sample clip per voice.
VOICE_REF_DIR = Path(__file__).resolve().parent.parent / "data" / "voice_refs"


def attach_voice_references(
    voices: List[Voice], *, ref_dir: Optional[Path] = None,
) -> int:
    """Point each voice at its reference clip and return how many were attached.

    The built-in catalog ships names only, so without this the ECAPA
    similarity term in `assign_voices` scores nothing and the assignment
    falls back to gender + language. Clips are found at
    `<ref_dir>/<engine>/<voice>.wav`; render them with
    `scripts/build_voice_refs.py`.
    """
    base = Path(ref_dir) if ref_dir is not None else VOICE_REF_DIR
    found = 0
    for v in voices:
        if v.reference_audio is not None:      # explicit catalog_json wins
            continue
        cand = base / v.engine / f"{v.voice}.wav"
        if cand.exists():
            v.reference_audio = cand
            found += 1
    # Count voices that HAVE a clip, not ones this call attached: the
    # built-in catalog is a module-level list of shared `Voice` objects, so
    # only the first `load_voice_catalog` attaches anything.
    have = sum(1 for v in voices if v.reference_audio is not None)
    if voices:
        log.info("voices: %d/%d catalog voices have reference clips "
                 "(%d attached this call, from %s)",
                 have, len(voices), found, base)
        if have == 0:
            log.info("voices: no voice reference clips found; the ECAPA "
                     "similarity term is INACTIVE and assignment will fall "
                     "back to target language + gender. Render them with "
                     "scripts/build_voice_refs.py to enable it.")
    return found


# ---------------------------------------------------------------------------
# per-speaker references
# ---------------------------------------------------------------------------

def extract_speaker_references(
    audio_wav: Path,
    turns: List,                       # list[Turn]
    *,
    out_dir: Path,
    min_s: float = 3.0,
    max_s: float = 15.0,
) -> Dict[str, Path]:
    """Cut 3-15 s of each speaker's longest single turn into a WAV.

    Longest-turn is the proxy for "cleanest"; no VAD cleanup.
    """
    import soundfile as sf
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    audio, sr = sf.read(str(audio_wav), always_2d=False)
    by_spk: Dict[str, List] = {}
    for t in turns:
        by_spk.setdefault(t.speaker, []).append(t)

    refs: Dict[str, Path] = {}
    for spk, ts in by_spk.items():
        ts.sort(key=lambda t: (t.end - t.start), reverse=True)
        for t in ts:
            dur = t.end - t.start
            if dur < min_s:
                continue
            start = int(t.start * sr)
            end = int(min(t.end, t.start + max_s) * sr)
            clip = audio[start:end]
            if len(clip) == 0:
                continue
            dest = out_dir / f"{spk}.wav"
            sf.write(str(dest), clip, sr)
            refs[spk] = dest
            break
        else:
            # No turn long enough: concatenate shortest-first until `min_s`.
            buf = []
            got = 0.0
            for t in sorted(ts, key=lambda t: (t.end - t.start), reverse=True):
                start = int(t.start * sr); end = int(t.end * sr)
                buf.append(audio[start:end])
                got += t.end - t.start
                if got >= min_s:
                    break
            if buf:
                import numpy as np
                clip = np.concatenate(buf, axis=0)
                dest = out_dir / f"{spk}.wav"
                sf.write(str(dest), clip, sr)
                refs[spk] = dest

    log.info("voices: extracted references for %d/%d speakers → %s",
             len(refs), len(by_spk), out_dir)
    return refs


# ---------------------------------------------------------------------------
# speaker-embedding similarity
# ---------------------------------------------------------------------------

def _ecapa_embed(wav_paths: List[Path], *, device: Optional[str] = None):
    """Return (N, 192) ECAPA-TDNN embeddings, or None if speechbrain is missing."""
    try:
        import torch
        from speechbrain.inference import EncoderClassifier  # type: ignore
    except ImportError:
        log.info("speechbrain not installed; skipping embedding-similarity tiebreaker")
        return None, None
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device or "cpu"},
    )
    import soundfile as sf
    embs = []
    for p in wav_paths:
        if p is None or not Path(p).exists():
            embs.append(None)
            continue
        audio, sr = sf.read(str(p), always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            import numpy as np
            idx = np.linspace(0, len(audio) - 1,
                              int(len(audio) * 16000 / sr)).astype(np.float64)
            audio = np.interp(idx, np.arange(len(audio)), audio).astype("float32")
        wav = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        emb = classifier.encode_batch(wav).squeeze(0).squeeze(0).cpu().numpy()
        embs.append(emb)
    return embs, "speechbrain/ecapa"


def _cosine(a, b) -> float:
    import numpy as np
    if a is None or b is None:
        return 0.0
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float((a @ b) / (na * nb))


# ---------------------------------------------------------------------------
# assignment
# ---------------------------------------------------------------------------

@dataclass
class SpeakerVoice:
    speaker: str
    engine: str
    voice: str
    score: float
    #: Numeric terms (`embed_cos`, `gender_match`, `language_match`) plus
    #: `basis`, the string naming which decided. `score` 0.0 means either no
    #: timbre match was available or the voice is orthogonal; `basis`
    #: distinguishes them.
    rationale: Dict[str, object]
    reference_audio: Optional[Path] = None


def assign_voices(
    speakers: List[dict],
    *,
    engine: str = "indic-speak",
    target_language: str,
    catalog_json: Optional[Path] = None,
    reference_audio: Optional[Dict[str, Path]] = None,
    device: Optional[str] = None,
    allow_voice_reuse: bool = False,
) -> List[SpeakerVoice]:
    """Pick one unique TTS voice per speaker: same gender, nearest timbre.

    Speakers are ranked by screen time and pick in that order. For each, the
    same-gender pool is ranked by ECAPA cosine against that speaker's
    reference clip, so `SpeakerVoice.score` is the cosine itself in [-1, 1],
    not a composite.

    Gender is a hard filter; language is a preference, since indic-speak
    voices read any of the 22 languages. An opposite-gender voice is reached
    only when the same-gender pool is exhausted and `allow_voice_reuse` is
    False, and is logged as a warning.

    The cosine needs a speaker clip, a catalog clip and `speechbrain`. With
    any of them missing every candidate scores 0.0, and the choice falls back
    to a voice recorded in the target language of the speaker's gender.
    `rationale.basis` records which path ran (`"embedding"` or
    `"language+gender"`). English always lands on the fallback: the engine
    accepts it as a `lang` but no voice is recorded in it.
    """
    catalog = load_voice_catalog(
        engine, language=target_language, catalog_json=catalog_json,
    )
    if not catalog:
        raise RuntimeError(
            f"No voices available for engine={engine!r} lang={target_language!r}"
        )

    male_pool = [v for v in catalog if v.gender == "male"]
    female_pool = [v for v in catalog if v.gender == "female"]
    neutral_pool = [v for v in catalog if v.gender not in ("male", "female")]

    ranked = sorted(speakers, key=lambda s: -float(s.get("total_speech_s", 0.0)))

    speaker_embs: Dict[str, object] = {}
    voice_embs: Dict[str, object] = {}
    if reference_audio:
        spk_ids = [s["id"] for s in ranked]
        spk_refs = [reference_audio.get(s) for s in spk_ids]
        vemb_refs = [v.reference_audio for v in catalog]
        embs, _ = _ecapa_embed(spk_refs + vemb_refs, device=device)
        if embs is not None:
            for sid, e in zip(spk_ids, embs[:len(spk_ids)]):
                if e is not None:
                    speaker_embs[sid] = e
            for v, e in zip(catalog, embs[len(spk_ids):]):
                if e is not None:
                    voice_embs[v.voice] = e

    target_primary = (target_language or "").split("-")[0].lower()
    fallback_used = False
    taken: set = set()
    out: List[SpeakerVoice] = []

    for spk in ranked:
        gender = spk.get("predicted_gender")

        # Gender is a hard filter, not a weight. The opposite pool is a
        # last resort when the same-gender one is exhausted.
        if gender == "male":
            primary, fallback = male_pool, neutral_pool + female_pool
        elif gender == "female":
            primary, fallback = female_pool, neutral_pool + male_pool
        else:
            primary, fallback = list(catalog), []

        pool = primary or list(catalog)
        if not allow_voice_reuse:
            fresh = [v for v in primary if v.voice not in taken]
            if not fresh:
                fresh = [v for v in fallback if v.voice not in taken]
                if fresh:
                    log.warning(
                        "voices: same-gender pool exhausted for %s "
                        "(gender=%s, %d voices); crossing gender",
                        spk["id"], gender, len(primary),
                    )
            if fresh:
                pool = fresh
            else:
                log.warning("voices: every voice already assigned; reusing "
                            "within the same-gender pool for %s", spk["id"])

        # Rank by cosine when there is one; the pool already guarantees
        # gender. Otherwise fall back to the target language explicitly,
        # rather than relying on the catalog's incidental order.
        can_embed = bool(voice_embs) and spk["id"] in speaker_embs
        # Gender is unknown, so `gender` above is `FALLBACK_GENDER`, a
        # placeholder rather than a prediction. Rank by target language
        # instead of dressing a guess up as a timbre match. Set by
        # `pipeline.py` from `GenderVote.decided_by`.
        no_gender_evidence = spk.get("gender_basis") == "fallback"
        if no_gender_evidence:
            log.warning("voices: %s has no gender evidence; casting a "
                        "target-language (%s) %s voice rather than ranking "
                        "by timbre", spk["id"], target_primary or "?", gender)
        scored: List[Tuple[float, float, Voice, Dict[str, object]]] = []
        for v in pool:
            why: Dict[str, object] = {}
            score = 0.0
            if v.gender and gender:
                why["gender_match"] = 1.0 if v.gender == gender else 0.0
            native = 1.0 if (v.language or "").lower() == target_primary else 0.0
            why["language_match"] = native
            if can_embed and v.voice in voice_embs:
                sim = _cosine(speaker_embs[spk["id"]], voice_embs[v.voice])
                why["embed_cos"] = sim
                score = sim
            # Language is the tiebreaker only: it cannot outvote a timbre
            # match and decides anything only when `score` is tied.
            scored.append((score, native, v, why))

        # Language first without gender evidence, similarity first with it;
        # either way the other term is the tiebreaker.
        scored.sort(key=(lambda row: (-row[1], -row[0])) if no_gender_evidence
                    else (lambda row: (-row[0], -row[1])))
        best_score, _native, best_voice, why = scored[0]
        if no_gender_evidence:
            why["basis"] = "language+gender (no gender evidence)"
        else:
            why["basis"] = "embedding" if (can_embed and "embed_cos" in why) \
                else "language+gender"
        if str(why["basis"]).startswith("language+gender"):
            fallback_used = True
        taken.add(best_voice.voice)
        out.append(SpeakerVoice(
            speaker=spk["id"],
            engine=best_voice.engine, voice=best_voice.voice,
            score=best_score, rationale=why,
            reference_audio=reference_audio.get(spk["id"]) if reference_audio else None,
        ))
        log.info("voices: %s (gender=%s) → %s (%.2f, %s)",
                 spk["id"], gender, best_voice.voice, best_score, why)

    if fallback_used:
        # The per-speaker lines above look identical whichever path ran, so
        # name the cause once here: `score: 0.00` means no match was
        # attempted, not a bad match.
        missing = []
        if not reference_audio:
            missing.append("no speaker reference clips")
        elif not speaker_embs:
            missing.append("speaker clips could not be embedded "
                           "(is speechbrain installed?)")
        if not voice_embs:
            missing.append(f"no catalog reference clips in {VOICE_REF_DIR} "
                           f"(render them with scripts/build_voice_refs.py)")
        log.warning(
            "voices: assigned by target language (%s) + gender instead of "
            "timbre for at least one speaker: %s",
            target_primary or "unset", "; ".join(missing) or "see rationale",
        )
        if target_primary and not any(
                (v.language or "").lower() == target_primary for v in catalog):
            log.warning(
                "voices: no catalog voice is RECORDED in %r either, so the "
                "language half of that fallback had nothing to prefer; "
                "gender alone decided. indic-speak reads %s fine, it just "
                "has no voice recorded in it.",
                target_primary, target_primary,
            )

    return out


def assignments_to_jsonable(voices: List[SpeakerVoice]) -> list:
    return [
        {
            "speaker": v.speaker,
            "engine": v.engine, "voice": v.voice,
            "score": v.score, "rationale": v.rationale,
            "reference_audio": str(v.reference_audio) if v.reference_audio else None,
        }
        for v in voices
    ]
