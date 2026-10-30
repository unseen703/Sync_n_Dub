"""Speaker-label refinement (stage 4c).

pyannote's labels are globally consistent, but a label is not guaranteed to
hold exactly one person: on noisy material it under-clusters, several
characters collapse into one label, and the dub reads them all in one voice
without erroring.

This splits impure labels by ECAPA cosine, then merges labels that turn out to
be one person. Every threshold here is set against SHORT-WINDOW cosine, which
tops out around 0.45-0.55 for a single speaker — not the ~0.9 a full-clip
comparison gives. Splitting is the pass that matters; merging guards the rarer
opposite failure.

Runs on audio alone, after resegmentation and before face detection, so the
stage order stays acyclic — ASD's face tracks only exist at stage 6.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from models.diarization.base import DiarizationResult, Turn

log = logging.getLogger(__name__)

#: Turns shorter than this give an unstable embedding; they are carried along
#: by their nearest embedded neighbour instead of voting.
MIN_TURN_S = 1.0

#: A label whose mean within-cosine falls below this is judged impure. Set at
#: the measured single-speaker floor so a noisy-but-single speaker is left
#: alone.
PURITY_FLOOR = 0.35

#: Average-linkage cosine DISTANCE at which a label is cut. 1 - 0.3 = 0.7:
#: same-speaker pairs sit near distance 0.5, different-speaker pairs near 0.9.
SPLIT_DISTANCE = 0.70

#: Two final labels whose centroids exceed this are one person split in two.
#: Above the single-speaker mean, so merging is rare.
MERGE_CEILING = 0.55

#: A label needs this many embeddable turns before its purity is judged. Below
#: it the mean is too noisy to act on.
MIN_TURNS_TO_JUDGE = 4

#: Below this much speech a sub-cluster is folded back into its nearest
#: sibling: stage 9d needs a reference clip long enough to predict gender and
#: rank timbre. Without the floor a split run produces singletons of a couple
#: of seconds each.
MIN_SPEAKER_S = 4.0

#: A fragment is only folded back when it plausibly belongs to the host.
#: "Nearest sibling" always finds someone, so an unconditional absorb undoes
#: the split it just made, pushing labels back below the purity floor. A
#: bit-part left standalone costs one catalog voice, not a wrong one.
ABSORB_MIN_COSINE = 0.35


def _embed_turns(audio_path: Path, turns: Sequence[Turn], *,
                 device: Optional[str] = None,
                 min_turn_s: float = MIN_TURN_S):
    """ECAPA embedding per turn, as (index -> vector), or None.

    Optional like `backend.voices._ecapa_embed`: without speechbrain the caller
    leaves the labels untouched rather than guessing.
    """
    try:
        import numpy as np
        import soundfile as sf
        import torch
        from speechbrain.inference import EncoderClassifier  # type: ignore
    except ImportError:
        log.info("speaker_id: speechbrain/soundfile missing; labels left as-is")
        return None

    audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device or "cpu"},
    )
    out: Dict[int, "np.ndarray"] = {}
    for i, t in enumerate(turns):
        if t.duration < min_turn_s:
            continue
        chunk = audio[int(t.start * sr):int(t.end * sr)]
        if chunk.size < int(min_turn_s * sr):
            continue
        with torch.no_grad():
            wav = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0)
            if device and device != "cpu":
                wav = wav.to(device)
            out[i] = enc.encode_batch(wav).squeeze().cpu().numpy()
    return out


def _cos(a, b) -> float:
    import numpy as np
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def label_purity(embs: Sequence) -> Optional[float]:
    """Mean pairwise cosine within one label. None when under-sampled."""
    import itertools
    import numpy as np
    if len(embs) < 2:
        return None
    pairs = [_cos(a, b) for a, b in itertools.combinations(embs, 2)]
    return float(np.mean(pairs))


def _split_label(embs: List, *, distance: float = SPLIT_DISTANCE) -> List[int]:
    """Agglomerative cut of one label's turns. Returns a sub-cluster id each."""
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    X = np.vstack(embs).astype(np.float64)
    X /= np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)
    model = AgglomerativeClustering(
        n_clusters=None, distance_threshold=distance,
        metric="cosine", linkage="average",
    )
    return list(model.fit_predict(X))


def refine_speakers(
    result: DiarizationResult,
    *,
    audio_path: Optional[Path] = None,
    device: Optional[str] = None,
    min_turn_s: float = MIN_TURN_S,
    purity_floor: float = PURITY_FLOOR,
    split_distance: float = SPLIT_DISTANCE,
    merge_ceiling: float = MERGE_CEILING,
    min_speaker_s: float = MIN_SPEAKER_S,
    absorb_min_cosine: float = ABSORB_MIN_COSINE,
    max_speakers: Optional[int] = None,
) -> DiarizationResult:
    """Split impure labels, merge duplicated ones, relabel by speech time.

    Returns a new result; the input is not mutated. `extra["speaker_id"]`
    records per-label purity and every split/merge, so a finished run can be
    audited without re-running the embeddings.
    """
    turns = list(result.turns)
    audio_path = Path(audio_path or result.audio_path)
    if not turns or not audio_path.exists():
        return result

    embs = _embed_turns(audio_path, turns, device=device, min_turn_s=min_turn_s)
    if not embs:
        return replace(result, extra={**result.extra,
                                      "speaker_id": {"ran": False,
                                                     "reason": "no embeddings"}})

    by_label: Dict[str, List[int]] = {}
    for i, t in enumerate(turns):
        by_label.setdefault(t.speaker, []).append(i)

    purity: Dict[str, Optional[float]] = {}
    new_label: Dict[int, str] = {i: t.speaker for i, t in enumerate(turns)}
    splits: Dict[str, int] = {}

    for label, idxs in by_label.items():
        scored = [i for i in idxs if i in embs]
        p = label_purity([embs[i] for i in scored]) if len(scored) >= MIN_TURNS_TO_JUDGE else None
        purity[label] = p
        if p is None or p >= purity_floor:
            continue
        sub = _split_label([embs[i] for i in scored], distance=split_distance)
        if len(set(sub)) < 2:
            continue
        splits[label] = len(set(sub))
        for i, s in zip(scored, sub):
            new_label[i] = f"{label}#{s}"
        # Short turns never voted; give each the sub-label of the nearest
        # scored turn of its own original label so no turn is left unattached.
        for i in idxs:
            if i in embs:
                continue
            nearest = min(scored, key=lambda j: abs(turns[j].start - turns[i].start))
            new_label[i] = new_label[nearest]
        log.info("speaker_id: %s purity %.3f < %.2f -> split into %d",
                 label, p, purity_floor, len(set(sub)))

    # Absorb sub-clusters too small to cast, by centroid cosine — a fragment
    # rejoins the voice it resembles, not the one nearest in time.
    absorbed: List[Tuple[str, str]] = []
    stranded: set = set()
    if splits and min_speaker_s > 0:
        while True:
            speech: Dict[str, float] = {}
            for i, t in enumerate(turns):
                speech[new_label[i]] = speech.get(new_label[i], 0.0) + t.duration
            cents = _centroids(new_label, embs)
            small = [l for l, d in speech.items()
                     if d < min_speaker_s and len(speech) > 1
                     and l not in stranded]
            if not small:
                break
            victim = min(small, key=lambda l: speech[l])
            others = [l for l in speech if l != victim]
            ranked = [l for l in others if l in cents]
            if victim in cents and ranked:
                host = max(ranked, key=lambda l: _cos(cents[victim], cents[l]))
                if _cos(cents[victim], cents[host]) < absorb_min_cosine:
                    # Resembles nobody here: a real bit-part, kept standalone.
                    stranded.add(victim)
                    continue
            else:
                host = max(others, key=lambda l: speech[l])
            for i, lab in new_label.items():
                if lab == victim:
                    new_label[i] = host
            absorbed.append((victim, host))
        if absorbed:
            log.info("speaker_id: absorbed %d sub-cluster(s) under %.1fs",
                     len(absorbed), min_speaker_s)
        if stranded:
            log.info("speaker_id: %d short label(s) kept standalone "
                     "(resembled no other speaker above %.2f)",
                     len(stranded), absorb_min_cosine)

    # Merge pass: the opposite failure, one person under two labels.
    merges: List[Tuple[str, str]] = []
    cents = _centroids(new_label, embs)
    for a, b in _pairs(sorted(cents)):
        if a not in cents or b not in cents:
            continue
        if _cos(cents[a], cents[b]) > merge_ceiling:
            for i, lab in new_label.items():
                if lab == b:
                    new_label[i] = a
            merges.append((b, a))
            cents = _centroids(new_label, embs)
            log.info("speaker_id: merged %s into %s (centroid cosine > %.2f)",
                     b, a, merge_ceiling)

    final = _canonicalise(turns, new_label, max_speakers=max_speakers)
    out_turns = [Turn(start=t.start, end=t.end, speaker=final[i])
                 for i, t in enumerate(turns)]
    before, after = result.num_speakers, len({t.speaker for t in out_turns})
    if after != before:
        log.info("speaker_id: %d label(s) -> %d after refinement", before, after)
    return replace(
        result,
        turns=out_turns,
        extra={**result.extra, "speaker_id": {
            "ran": True,
            "speakers_before": before,
            "speakers_after": after,
            "purity": {k: (round(v, 4) if v is not None else None)
                       for k, v in purity.items()},
            "splits": splits,
            "absorbed": absorbed,
            "stranded": sorted(stranded),
            "merges": merges,
            "min_speaker_s": min_speaker_s,
            "purity_floor": purity_floor,
            "split_distance": split_distance,
            "merge_ceiling": merge_ceiling,
            "turns_embedded": len(embs),
            "turns_total": len(turns),
        }},
    )


def _pairs(xs):
    import itertools
    return list(itertools.combinations(xs, 2))


def _centroids(new_label: Dict[int, str], embs) -> Dict[str, "object"]:
    import numpy as np
    acc: Dict[str, List] = {}
    for i, e in embs.items():
        acc.setdefault(new_label[i], []).append(e)
    return {k: np.mean(v, axis=0) for k, v in acc.items() if v}


def _canonicalise(turns: Sequence[Turn], new_label: Dict[int, str], *,
                  max_speakers: Optional[int] = None) -> Dict[int, str]:
    """Rename to SPK_00.. by descending speech time, stable and gap-free.

    Ordering by speech time matches `backend.voices.assign_voices`, so the lead
    gets SPK_00 in both places.
    """
    total: Dict[str, float] = {}
    for i, t in enumerate(turns):
        total[new_label[i]] = total.get(new_label[i], 0.0) + t.duration
    order = sorted(total, key=lambda k: (-total[k], k))
    if max_speakers and len(order) > max_speakers:
        # Fold the smallest labels into the nearest survivor by speech time.
        # The cap is an operator instruction, so honour it exactly.
        keep, drop = order[:max_speakers], order[max_speakers:]
        log.info("speaker_id: capping %d labels at max_speakers=%d",
                 len(order), max_speakers)
        for i, lab in list(new_label.items()):
            if lab in drop:
                new_label[i] = keep[-1]
        total = {}
        for i, t in enumerate(turns):
            total[new_label[i]] = total.get(new_label[i], 0.0) + t.duration
        order = sorted(total, key=lambda k: (-total[k], k))
    canon = {lab: f"SPK_{n:02d}" for n, lab in enumerate(order)}
    return {i: canon[lab] for i, lab in new_label.items()}
