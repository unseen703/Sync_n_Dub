"""Run the same clip through several dubbing variants, holding everything else fixed.

Every arm must see the same upstream input, which two mechanisms provide:

  `reuse_prefix`      download -> ASR -> align -> MT runs ONCE and is cached
                      at `data/exp/<base>/_shared/prefix.mt.json`. Bodhan's
                      ASR and MT are network calls with real variance, so
                      re-running them per arm would put that variance inside
                      the measurement. The cached prefix is COPIED into each
                      variant and never mutated.

  `reuse_tts_from`    a variant may take another variant's `tts/raw/` by
                      symlink. An arm that only changes PLACEMENT -- the
                      timing producer, the render-time stretch filter, the mix
                      strategy -- runs on the same synthesised waveform. Only
                      an arm that changes what is said, or how fast it is
                      spoken, pays for its own TTS pass.

Output tree (one directory per variant, nothing shared but the prefix)::

    data/exp/<base>/_shared/prefix.mt.json      copied, never mutated
    data/exp/<base>/<variant_id>/
        variant.json          fully-resolved config, every knob incl. defaults
        plan.mt.json          the merged JSON this variant mutates
        timing/<producer>.plan.json
        tts/{raw,stretched}/SEG_*.wav
        mix/<base>.dubbed.wav
        metrics.json

How it drives the pipeline
--------------------------
Nothing here re-implements a stage. The bootstrap and the post-MT chain go
through `backend.pipeline.run_pipeline` (via `variant_tag` / `exp_root` /
`post_mt_only_from`); `backend.timing.plan` and `backend.mix.mix_dubbed_audio`
are called directly where the pipeline cannot interleave them.

Two places the pipeline API does not reach, both reported rather than worked
around silently:

  * the timing plan must be built AFTER the TTS clips exist and BEFORE the mix
    reads them, while `run_post_mt_stages` runs TTS and mix in one call. So
    the mix is invoked directly, with the source video / M&E stem taken off
    the `PipelineResult` the bootstrap returned.
  * `tts_kwargs` is the one `synthesize_segments` knob with no `run_pipeline`
    counterpart, so a variant that sets it pays for a second synthesis pass,
    logged as `resynthesis_reason`. Everything else in the `tts` section,
    `speed_control` and its `speed_*` companions included, is forwarded onto
    the pipeline's own stage-9f pass by `_forward_tts_to_pipeline`.

Config shape. A variant's `config` has four optional sections, each validated
against the real signature of the function it feeds, so a typo'd knob fails at
load time instead of forty minutes into a sweep::

    {"pipeline": {...run_pipeline knobs...},
     "tts":      {...synthesize_segments knobs...} | null,
     "timing":   {"producer": "dp", "params": {...plan_dp knobs...}} | null,
     "mix":      {...mix_dubbed_audio knobs...} | null}

`null` skips that stage. `variant.json` records the resolved value of every
knob in every live section, defaults included.
"""
from __future__ import annotations

import inspect
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .pipeline import DEFAULT_DATA_DIR, PipelineResult, run_pipeline

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

#: Sections a variant's `config` may carry. Order is execution order.
SECTIONS = ("pipeline", "tts", "timing", "mix")

#: The `run_pipeline` knobs a variant is allowed to move: the post-MT subset.
#: Everything else belongs to stages 1..9a, which the prefix froze, so this
#: list makes an upstream knob change impossible rather than discouraged.
#: Every name is checked against the real signature at import time.
PIPELINE_KNOBS = (
    "translit_model",
    "duration_adjust",
    "duration_voice",
    "duration_safety_margin_s",
    "voice_engine",
    "allow_voice_reuse",
    "tts_model",
    "tts_sample_rate",
    "tts_max_stretch",
    "stretch_backend",
    "tts_apply_stretch",
    # Engine-side speed control (M1). `run_pipeline` forwards these straight
    # to `synthesize_segments`, so moving them needs no second TTS pass.
    "tts_speed_control",
    "tts_speed_bounds",
    "tts_speed_tolerance",
    "tts_speed_max_rounds",
    # Post-TTS forced alignment (method M5, stage 9f1). Fills
    # `segments[].tts_words[]`, which M3, M4 and M7 all read. Post-MT, so a
    # variant may move it freely and it is NOT a prefix knob.
    "tts_force_align",
    "tts_align_model",
    # Viseme extraction (method M6, stage 9f1b). Fills `src_visemes` /
    # `tts_visemes`, which `plan_dp`'s D_phoneme term reads. Also post-MT.
    # `lambda_phoneme` itself is NOT here -- it reaches the planner through
    # the `timing.params` section.
    "attach_visemes_stage",
    "viseme_backend",
    "viseme_fallback_backend",
    "viseme_source_audio",
    # Duration-aware translation (method M8, stage 9a). See the PREFIX_KNOBS
    # note: these act in the prefix, so they are settable only in
    # `base_config`, never per variant.
    "mt_n_best",
    "mt_length_penalties",
    "mt_diversity_penalty",
    # Upstream model selection, driving stages 7/8/9a. These are PREFIX
    # knobs: `ensure_prefix` forwards them and `_run_one` does not, since
    # `post_mt_only_from` skips 1..9a and they would be silent no-ops.
    # Without them a suite's `--input <video>` only works for a clip whose
    # `data/mt/<base>.*.mt.json` already exists.
    "asr_model",
    "asr_language",
    "align_model",
    "mt_model",
    "mt_target_language",
    "mt_target_script",
    "diarization_model",
    "stem_model",
    "skip_stems",
)

#: The subset of `PIPELINE_KNOBS` belonging to stages 1..9a, handed to
#: `ensure_prefix` rather than to a variant's post-MT replay.
PREFIX_KNOBS = (
    "asr_model",
    "asr_language",
    "align_model",
    "mt_model",
    "mt_n_best",
    "mt_length_penalties",
    "mt_diversity_penalty",
    "mt_target_language",
    "mt_target_script",
    "diarization_model",
    "stem_model",
    "skip_stems",
)

#: Knobs that only act in stages 1..9a. `run_pipeline` accepts them in the
#: post-MT replay but ignores them, so passing them per-variant would read as
#: "this variant re-ran ASR" and mean nothing. Stripped from `run_kwargs` in
#: `_run_one`. `mt_model` is NOT here: `_bootstrap_post_mt_only` takes it and
#: records it on the result.
_PREFIX_ONLY_KNOBS = ("asr_model", "asr_language", "align_model",
                      "mt_n_best", "mt_length_penalties",
                      "mt_diversity_penalty")

#: `synthesize_segments` parameters a variant may set. The rest are wiring
#: (`merged_json`, `out_dir`, `speaker_voice_map`, ...) owned by this module.
TTS_KNOBS = (
    "tts_model",
    "sample_rate",
    "max_stretch",
    "stretch_backend",
    "apply_stretch",
    "speed_control",
    "speed_bounds",
    "speed_tolerance",
    "speed_max_rounds",
    "tts_kwargs",
)

#: `mix_dubbed_audio` parameters a variant may set.
MIX_KNOBS = (
    "sample_rate",
    "tts_gain_db",
    "me_gain_db",
    "mux_video",
    "strategy",
    "tempo_cap",
    "clip_source",
    "stretch_backend",
    "max_pieces",
)

#: `tts` section knob -> the `run_pipeline` knob that carries it into
#: `synthesize_segments`, so a variant naming one is served by the pipeline's
#: own synthesis pass. `tts_kwargs` is absent on purpose: it has no pipeline
#: counterpart, so a variant that sets it needs a second pass, reported as
#: `resynthesis_reason`.
_TTS_KNOB_TO_PIPELINE = {
    "tts_model": "tts_model",
    "sample_rate": "tts_sample_rate",
    "max_stretch": "tts_max_stretch",
    "stretch_backend": "stretch_backend",
    "apply_stretch": "tts_apply_stretch",
    "speed_control": "tts_speed_control",
    "speed_bounds": "tts_speed_bounds",
    "speed_tolerance": "tts_speed_tolerance",
    "speed_max_rounds": "tts_speed_max_rounds",
}


# ---------------------------------------------------------------------------
# startup: the environment this runner's subprocesses need. Both are
# delegated to `backend.cli` / `backend.stem_separation`, so there is one .env
# rule and one ffmpeg shim, not two of each.
# ---------------------------------------------------------------------------

def _ensure_ffmpeg_on_path() -> None:
    """Put an `ffmpeg` on PATH for this process and everything it spawns.

    Needed because demucs shells out to a bare `ffmpeg`, so stage 3 raises
    FileNotFoundError in a shell with no system ffmpeg. (`backend.media_io`
    calls the imageio binary by absolute path and does not need this.)

    Delegates to `backend.stem_separation._ensure_ffmpeg_on_path`, which
    builds and caches the shim symlink for demucs. A no-op when
    `shutil.which("ffmpeg")` already resolves.
    """
    from .stem_separation import _ensure_ffmpeg_on_path as _ensure
    _ensure()


def _load_env(allow_insecure_env: bool = False) -> None:
    """Load `.env`, keeping `backend.cli`'s file-permission check.

    This is a second entry point into the same pipeline and needs the same
    secrets -- without `HF_TOKEN` the diarizer fails. The check and the load
    are imported from `backend.cli`, so the mode-0600 rule has one definition.
    """
    from dotenv import load_dotenv
    from .cli import _check_env_permissions
    if not allow_insecure_env:
        _check_env_permissions()
    load_dotenv()


# ---------------------------------------------------------------------------
# knob tables, resolved off the real signatures
# ---------------------------------------------------------------------------

def _defaults_for(fn, names: Sequence[str]) -> Dict[str, Any]:
    """`{name: default}` for `names`, raising if one is not a real parameter.

    This is the typo guard. A suite file is written by hand, read once, and
    then runs for hours; a key that silently does nothing is strictly worse
    than a crash on line one.
    """
    sig = inspect.signature(fn)
    missing = [n for n in names if n not in sig.parameters]
    if missing:
        raise ValueError(
            f"{fn.__module__}.{fn.__qualname__} has no parameter(s) "
            f"{', '.join(missing)}; known: {', '.join(sig.parameters)}"
        )
    out: Dict[str, Any] = {}
    for n in names:
        d = sig.parameters[n].default
        out[n] = None if d is inspect.Parameter.empty else d
    return out


def _timing_defaults(producer: str) -> Dict[str, Any]:
    """Every knob of `plan_<producer>`, defaults included.

    Resolved through `backend.timing`, not a hard-coded table, so the two
    producers `backend.timing_advanced` attaches at import time are found the
    same way the dispatcher finds them.
    """
    from . import timing as _t
    from . import timing_advanced as _adv   # noqa: F401  (registers dp/word_anchored)

    if producer not in _t.PRODUCERS:
        raise ValueError(f"unknown timing producer {producer!r}; "
                         f"choose one of {_t.PRODUCERS}")
    fn = getattr(_t, f"plan_{producer}", None)
    if fn is None:
        raise NotImplementedError(f"timing producer {producer!r} is not implemented")
    names = [n for n in inspect.signature(fn).parameters
             if n not in ("merged_json", "out_json")]
    return _defaults_for(fn, names)


def _jsonable(v: Any) -> Any:
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, tuple):
        return [_jsonable(x) for x in v]
    if isinstance(v, list):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    return v


def _merge(base: Optional[dict], over: Optional[dict]) -> dict:
    out = dict(base or {})
    out.update(over or {})
    return out


# ---------------------------------------------------------------------------
# the variant
# ---------------------------------------------------------------------------

@dataclass
class Variant:
    """One arm of the sweep."""

    id: str
    label: str
    method: str
    config: dict
    reuse_tts_from: Optional[str] = None

    @classmethod
    def from_jsonable(cls, d: dict) -> "Variant":
        missing = [k for k in ("id", "label", "method") if not d.get(k)]
        if missing:
            raise ValueError(f"variant is missing {', '.join(missing)}: {d!r}")
        unknown = set(d) - {"id", "label", "method", "config", "reuse_tts_from"}
        if unknown:
            raise ValueError(f"variant {d['id']!r}: unknown key(s) "
                             f"{', '.join(sorted(unknown))}")
        return cls(id=str(d["id"]), label=str(d["label"]), method=str(d["method"]),
                   config=dict(d.get("config") or {}),
                   reuse_tts_from=d.get("reuse_tts_from"))

    def to_jsonable(self) -> dict:
        return {"id": self.id, "label": self.label, "method": self.method,
                "reuse_tts_from": self.reuse_tts_from,
                "config": _jsonable(self.config)}


@dataclass
class VariantResult:
    id: str
    label: str
    method: str
    dir: Optional[Path] = None
    plan_json: Optional[Path] = None
    timing_plan_json: Optional[Path] = None
    tts_dir: Optional[Path] = None
    reused_tts_from: Optional[str] = None
    mix_wav: Optional[Path] = None
    mix_mp4: Optional[Path] = None
    metrics_json: Optional[Path] = None
    wall_clock_s: float = 0.0
    skipped: Optional[str] = None
    error: Optional[str] = None
    summary: dict = field(default_factory=dict)

    def to_jsonable(self) -> dict:
        return {k: (str(v) if isinstance(v, Path) else v)
                for k, v in self.__dict__.items()}


@dataclass
class ExperimentResult:
    exp_id: str
    base: str
    root: Optional[Path] = None
    prefix_json: Optional[Path] = None
    prefix_reused: bool = False
    variants: List[VariantResult] = field(default_factory=list)
    dry_run: bool = False
    wall_clock_s: float = 0.0

    @property
    def ok(self) -> bool:
        return all(v.error is None for v in self.variants)

    def to_jsonable(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "exp_id": self.exp_id,
            "base": self.base,
            "root": str(self.root) if self.root else None,
            "prefix_json": str(self.prefix_json) if self.prefix_json else None,
            "prefix_reused": self.prefix_reused,
            "dry_run": self.dry_run,
            "wall_clock_s": round(self.wall_clock_s, 3),
            "variants": [v.to_jsonable() for v in self.variants],
        }

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_jsonable(), indent=2, ensure_ascii=False))
        return path


# ---------------------------------------------------------------------------
# suite files
# ---------------------------------------------------------------------------

def load_suite(path: Path) -> Tuple[dict, List[Variant]]:
    """Read a suite file. JSON only -- this repo has no YAML dependency.

    Returns `(base_config, variants)`; `base_config` holds the per-section
    defaults every variant inherits. Duplicate ids, unknown keys and a
    `reuse_tts_from` naming a variant that does not exist or is defined later
    are all rejected here rather than mid-sweep.
    """
    path = Path(path)
    doc = json.loads(path.read_text())
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: suite must be a JSON object")
    unknown = set(doc) - {"name", "description", "base_config", "variants",
                          "schema_version", "notes"}
    if unknown:
        raise ValueError(f"{path}: unknown top-level key(s) "
                         f"{', '.join(sorted(unknown))}")
    base_config = dict(doc.get("base_config") or {})
    bad = set(base_config) - set(SECTIONS)
    if bad:
        raise ValueError(f"{path}: base_config has unknown section(s) "
                         f"{', '.join(sorted(bad))}; known: {', '.join(SECTIONS)}")

    raw = doc.get("variants")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: 'variants' must be a non-empty list")
    variants = [Variant.from_jsonable(v) for v in raw]

    seen: set = set()
    for v in variants:
        if v.id in seen:
            raise ValueError(f"{path}: duplicate variant id {v.id!r}")
        seen.add(v.id)
        if v.reuse_tts_from is not None and v.reuse_tts_from not in seen:
            # `not in seen`, not `not in all ids`: the donor must be EARLIER
            # in the list, which is the order they run in.
            raise ValueError(
                f"{path}: variant {v.id!r} reuses TTS from "
                f"{v.reuse_tts_from!r}, which is not defined before it")
        if v.reuse_tts_from == v.id:
            raise ValueError(f"{path}: variant {v.id!r} reuses TTS from itself")
    log.info("suite %s: %d variants (%d own a TTS pass)", path.name, len(variants),
             sum(1 for v in variants if v.reuse_tts_from is None))
    return base_config, variants


def resolve_config(variant: Variant, base_config: Optional[dict] = None) -> dict:
    """Expand a variant's config into every knob of every live section.

    Validation happens here, so `load_suite` + `resolve_config` is a complete
    dry check of a suite file: an unknown knob raises, naming the function
    whose signature it failed against.
    """
    from . import mix as _mix
    from . import tts as _tts

    base_config = base_config or {}
    bad = set(variant.config) - set(SECTIONS)
    if bad:
        raise ValueError(f"variant {variant.id!r}: unknown section(s) "
                         f"{', '.join(sorted(bad))}; known: {', '.join(SECTIONS)}")

    out: Dict[str, Any] = {}
    #: What the suite actually WROTE per section, before defaults were filled
    #: in. `_forward_tts_to_pipeline` uses it to tell an explicit 1.15x from a
    #: 1.15 that is merely the default.
    explicit: Dict[str, Dict[str, Any]] = {}

    for name, fn, knobs in (("pipeline", run_pipeline, PIPELINE_KNOBS),
                            ("tts", _tts.synthesize_segments, TTS_KNOBS),
                            ("mix", _mix.mix_dubbed_audio, MIX_KNOBS)):
        over = (variant.config[name] if name in variant.config
                else base_config.get(name, {} if name == "pipeline" else None))
        if over is None:
            out[name] = None
            continue
        if not isinstance(over, dict):
            raise ValueError(f"variant {variant.id!r}: section {name!r} must be "
                             f"an object or null")
        merged = _merge(base_config.get(name) or {}, over)
        unknown = set(merged) - set(knobs)
        if unknown:
            raise ValueError(
                f"variant {variant.id!r}: section {name!r} has unknown knob(s) "
                f"{', '.join(sorted(unknown))}; settable: {', '.join(knobs)}")
        # Reject a PREFIX knob set by a VARIANT. Stages 1..9a run once per
        # sweep (`ensure_prefix`), driven only by `base_config.pipeline`;
        # `_run_one` always passes `post_mt_only_from` and strips
        # `_PREFIX_ONLY_KNOBS` from `run_kwargs`. So a variant that says
        # `mt_n_best: 4` would run against the cached n_best=1 prefix and
        # report the baseline's numbers under its own name, with no error
        # anywhere. Fail here instead, and say where the knob belongs.
        if name == "pipeline" and name in variant.config:
            leaked = sorted(set(variant.config[name]) & set(PREFIX_KNOBS))
            if leaked:
                raise ValueError(
                    f"variant {variant.id!r}: section 'pipeline' sets prefix "
                    f"knob(s) {', '.join(leaked)}. Stages 1..9a run once per "
                    f"sweep and are driven only by base_config.pipeline, so a "
                    f"per-variant value here would be silently ignored and the "
                    f"arm would report the baseline's numbers under its own "
                    f"name. Move it to base_config.pipeline (applies to every "
                    f"arm), or give it its own sweep with its own --exp-id.")
        resolved = _defaults_for(fn, knobs)
        resolved.update(merged)
        out[name] = _jsonable(resolved)
        explicit[name] = dict(merged)

    tim = (variant.config["timing"] if "timing" in variant.config
           else base_config.get("timing"))
    if tim is None:
        out["timing"] = None
    else:
        if not isinstance(tim, dict):
            raise ValueError(f"variant {variant.id!r}: section 'timing' must be "
                             f"an object or null")
        unknown = set(tim) - {"producer", "params"}
        if unknown:
            raise ValueError(f"variant {variant.id!r}: timing section has unknown "
                             f"key(s) {', '.join(sorted(unknown))}")
        producer = tim.get("producer") or variant.method
        defaults = _timing_defaults(producer)
        params = _merge((base_config.get("timing") or {}).get("params"),
                        tim.get("params"))
        bad_p = set(params) - set(defaults)
        if bad_p:
            raise ValueError(
                f"variant {variant.id!r}: timing producer {producer!r} has no "
                f"knob(s) {', '.join(sorted(bad_p))}; "
                f"settable: {', '.join(sorted(defaults))}")
        defaults.update(params)
        out["timing"] = {"producer": producer, "params": _jsonable(defaults)}

    out["resynthesis_reason"] = _forward_tts_to_pipeline(variant, out, explicit)
    return out


def _forward_tts_to_pipeline(
    variant: "Variant", out: Dict[str, Any], explicit: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    """Push the variant's `tts` section onto its `pipeline` section in place.

    Every `synthesize_segments` knob a suite can name except `tts_kwargs` has
    a `run_pipeline` counterpart (`_TTS_KNOB_TO_PIPELINE`). Copying the
    explicitly-set ones across lets the pipeline's own stage-9f pass
    synthesise at the requested rate, so `_run_one` makes no second
    `synthesize_segments` call.

    Returns the `resynthesis_reason`: None when the pipeline can carry
    everything, otherwise the sentence `_run_one` logs before buying the extra
    pass. A knob that cannot be forwarded is REPORTED, never dropped.
    """
    tts_cfg, pipe_cfg = out.get("tts"), out.get("pipeline")
    if not tts_cfg:
        return None

    set_in_tts = explicit.get("tts") or {}
    unforwardable = sorted(k for k in set_in_tts
                           if k not in _TTS_KNOB_TO_PIPELINE)
    if unforwardable:
        # Only `tts_kwargs` lands here today; named rather than hard-coded so
        # a future TTS_KNOBS entry without a counterpart says so.
        return (f"run_pipeline has no counterpart for tts knob(s) "
                f"{', '.join(unforwardable)}")

    forwardable = dict(set_in_tts)
    if not forwardable:
        return None

    if pipe_cfg is None:
        return ("this variant has no 'pipeline' section, so no pipeline TTS "
                "pass exists to carry " + ", ".join(sorted(forwardable)))
    if not (forwardable.get("tts_model") or pipe_cfg.get("tts_model")):
        return ("no tts_model is set on either section, so the pipeline runs "
                "no TTS stage to carry " + ", ".join(sorted(forwardable)))

    set_in_pipe = explicit.get("pipeline") or {}
    for tts_name, value in sorted(forwardable.items()):
        pipe_name = _TTS_KNOB_TO_PIPELINE[tts_name]
        if pipe_name in set_in_pipe and _jsonable(set_in_pipe[pipe_name]) != _jsonable(value):
            # Two sections of one variant asking for different things is a
            # suite bug; preferring one would make `variant.json` lie.
            raise ValueError(
                f"variant {variant.id!r}: tts.{tts_name}={value!r} conflicts "
                f"with pipeline.{pipe_name}={set_in_pipe[pipe_name]!r}; these "
                f"feed the same synthesize_segments argument -- set one")
        pipe_cfg[pipe_name] = _jsonable(value)
    log.debug("variant %s: forwarded %s onto the pipeline TTS pass",
              variant.id, ", ".join(sorted(forwardable)))
    return None


# ---------------------------------------------------------------------------
# the shared prefix
# ---------------------------------------------------------------------------

VIDEO_SUFFIXES = (".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi")


def _is_merged_json(p: str) -> bool:
    return str(p).endswith(".json") and Path(p).expanduser().exists()


def _base_name(input_: str, prefix_json: Optional[Path]) -> str:
    """The clip's short name: what every output path is keyed on."""
    if prefix_json is not None:
        stem = Path(prefix_json).name
        for suffix in (".mt.json", ".json"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        # `<clip>.<mt_model>.<lang>` -> `<clip>`
        return stem.split(".")[0]
    p = Path(input_)
    return p.stem if p.suffix else p.name


def _find_merged_json(base: str, data_dir: Path) -> Optional[Path]:
    """The newest `data/mt/<base>.*.mt.json`, or None.

    Looked up rather than rebuilt: an ASR+MT pass costs minutes and network
    calls, and every arm of the sweep shares one.
    """
    hits = sorted((Path(data_dir) / "mt").glob(f"{base}.*.mt.json"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0] if hits else None


def _resolve_video(input_: str, base: str, data_dir: Path) -> Optional[Path]:
    """The source video behind `input_`, which may itself be a merged JSON.

    `run_pipeline(post_mt_only_from=...)` wants the video, not the JSON --
    handed a `.mt.json` it tries to download it -- so a JSON input is mapped
    back to `data/videos/<base>.<ext>` here.
    """
    p = Path(input_).expanduser()
    if p.suffix.lower() in VIDEO_SUFFIXES and p.exists():
        return p
    for ext in VIDEO_SUFFIXES:
        cand = Path(data_dir) / "videos" / f"{base}{ext}"
        if cand.exists():
            return cand
    return None


def ensure_prefix(
    input_: str,
    *,
    root: Path,
    data_dir: Path = DEFAULT_DATA_DIR,
    reuse_prefix: bool = True,
    pipeline_kwargs: Optional[dict] = None,
    dry_run: bool = False,
) -> Tuple[Optional[Path], bool]:
    """Produce `<root>/_shared/prefix.mt.json`; return `(path, reused)`.

    Three ways in, cheapest first:
      1. the cache is already there and `reuse_prefix` is on;
      2. `input_` IS a merged JSON -- it is copied in, never opened for write;
      3. otherwise stages 1..9a run once through `run_pipeline` with every
         post-MT stage off, and its merged JSON is copied in.

    Always a COPY: every variant mutates its own `plan.mt.json`, and the file
    they descend from must still be byte-identical at the end of the sweep.
    `data/mt/*.mt.json` is not a scratch directory.
    """
    shared = Path(root) / "_shared"
    dest = shared / "prefix.mt.json"
    if reuse_prefix and dest.exists():
        log.info("prefix: reusing %s (%.1f kB)", dest, dest.stat().st_size / 1024)
        return dest, True

    src = (Path(input_).expanduser().resolve() if _is_merged_json(input_)
           else (_find_merged_json(_base_name(input_, None), data_dir)
                 if reuse_prefix else None))
    if src is not None:
        if dry_run:
            log.info("prefix[dry-run]: would copy %s -> %s", src, dest)
            return src, False
        shared.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        log.info("prefix: copied %s -> %s", src, dest)
        return dest, False

    if dry_run:
        log.info("prefix[dry-run]: would run stages 1..9a for %s", input_)
        return None, False

    log.info("=== prefix: stages 1..9a (runs ONCE for %d-variant sweep) ===", 0)
    # The only call here that reaches demucs, which shells out to a bare
    # `ffmpeg`.
    _ensure_ffmpeg_on_path()
    res = run_pipeline(input_, data_dir=Path(data_dir), **(pipeline_kwargs or {}))
    if res.mt_json is None:
        raise RuntimeError(
            "prefix: the upstream pass produced no merged JSON, so stage 9a "
            "(MT) never ran. Set the upstream models in the suite's "
            "`base_config.pipeline` -- asr_model, asr_language, align_model, "
            "mt_model, mt_target_language. (There are no --mt-model / "
            "--asr-model flags; prefix knobs are named in the suite, not on "
            "the command line.)")
    shared.mkdir(parents=True, exist_ok=True)
    shutil.copy2(res.mt_json, dest)
    log.info("prefix: copied %s -> %s", res.mt_json, dest)
    return dest, False


# ---------------------------------------------------------------------------
# per-variant plumbing
# ---------------------------------------------------------------------------

def _tts_dir_for(variant_dir: Path, base: str, tts_model: Optional[str]) -> Path:
    """Where `run_post_mt_stages` puts this variant's clips: `tts/<stem>.<model>`."""
    return Path(variant_dir) / "tts" / f"{base}.{tts_model}"


def _link_tts(donor_tts: Path, dest_tts: Path) -> Path:
    """Symlink the donor's `raw/` under this variant, give it its own `stretched/`.

    Only `raw/` is shared, and only by link, so every variant reads its clips
    through a path INSIDE its own tree and a cross-variant write is visible
    from the paths alone.
    """
    dest_tts.mkdir(parents=True, exist_ok=True)
    raw_link = dest_tts / "raw"
    donor_raw = Path(donor_tts) / "raw"
    if not donor_raw.is_dir():
        raise FileNotFoundError(
            f"reuse_tts_from: donor has no raw clips at {donor_raw}")
    if raw_link.is_symlink() or raw_link.exists():
        if raw_link.is_symlink():
            raw_link.unlink()
        else:
            raise FileExistsError(
                f"{raw_link} already exists as a real directory; refusing to "
                f"replace a variant's own clips with a link")
    os.symlink(os.path.relpath(donor_raw, dest_tts), raw_link)
    (dest_tts / "stretched").mkdir(exist_ok=True)
    return raw_link


def _repoint_clips(plan_json: Path, donor_raw: Path, own_raw: Path) -> int:
    """Rewrite `tts_raw_audio` / `tts_audio` from the donor's path onto ours.

    Without it a reusing variant's merged JSON points at the donor's
    directory, and any stage writing a derived file beside its input writes
    into another variant's tree.
    """
    data = json.loads(Path(plan_json).read_text())
    donor_raw = Path(donor_raw).resolve()
    n = 0
    for seg in data.get("segments", []):
        for key in ("tts_raw_audio", "tts_audio"):
            val = seg.get(key)
            if not val:
                continue
            p = Path(val)
            try:
                rel = p.resolve().relative_to(donor_raw)
            except ValueError:
                continue
            seg[key] = str(Path(own_raw) / rel)
            n += 1
    extra_tts = (data.get("extra") or {}).get("tts")
    if isinstance(extra_tts, dict):
        extra_tts["raw_dir"] = str(own_raw)
        extra_tts["reused_from"] = str(donor_raw)
    Path(plan_json).write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return n


def _has_clips(plan_json: Path) -> bool:
    """True when every segment of `plan_json` already points at a real clip."""
    data = json.loads(Path(plan_json).read_text())
    segs = data.get("segments") or []
    return bool(segs) and all(
        seg.get("tts_raw_audio") and Path(seg["tts_raw_audio"]).exists()
        for seg in segs)


# ---------------------------------------------------------------------------
# the runner
# ---------------------------------------------------------------------------

def run_experiment(
    input_,
    *,
    variants: Sequence[Variant],
    base_config: Optional[dict] = None,
    data_dir: Path = DEFAULT_DATA_DIR,
    exp_id: Optional[str] = None,
    reuse_prefix: bool = True,
    only: Optional[Sequence[str]] = None,
    dry_run: bool = False,
) -> ExperimentResult:
    """Run every variant of `input_` into its own directory under `data/exp`.

    `input_` is a video path, a URL, or a merged `*.mt.json` from an earlier
    run (the usual case: the prefix is already on disk and costs nothing).

    `only` restricts the run to those variant ids. A restricted run still
    needs its donors, so a variant naming `reuse_tts_from` is refused when the
    donor is neither selected nor already on disk. `dry_run` resolves and
    validates every config and lays out the tree without running a stage.
    """
    t_start = time.perf_counter()
    data_dir = Path(data_dir)
    variants = list(variants)
    if not variants:
        raise ValueError("run_experiment: no variants")

    # Resolve EVERY config before touching the disk, so a typo in variant 8
    # surfaces before variants 1..7 burn an hour of GPU.
    resolved: Dict[str, dict] = {}
    for v in variants:
        resolved[v.id] = resolve_config(v, base_config)

    # Upstream model choice is a property of the SHARED prefix, so only
    # `base_config` may carry it. A variant naming one would be resolved,
    # written into `variant.json` and then do nothing -- refuse instead.
    _model_knobs = ("asr_model", "asr_language", "align_model", "mt_model")
    for v in variants:
        stray = sorted(set((v.config.get("pipeline") or {})) & set(_model_knobs))
        if stray:
            raise ValueError(
                f"variant {v.id!r}: {', '.join(stray)} "
                f"{'selects' if len(stray) == 1 else 'select'} the upstream "
                f"pass, which every variant shares; move them to the suite's "
                f"base_config.pipeline (a per-variant value cannot take "
                f"effect -- stages 1..9a run once, before any variant does)")

    prefix_pipeline = dict((base_config or {}).get("pipeline") or {})
    base = _base_name(str(input_),
                      Path(input_) if _is_merged_json(str(input_)) else None)
    exp_id = exp_id or base
    root = data_dir / "exp" / exp_id
    result = ExperimentResult(exp_id=exp_id, base=base, root=root, dry_run=dry_run)

    prefix_json, reused = ensure_prefix(
        str(input_), root=root, data_dir=data_dir,
        reuse_prefix=reuse_prefix,
        pipeline_kwargs={k: v for k, v in prefix_pipeline.items()
                         if k in PREFIX_KNOBS},
        dry_run=dry_run,
    )
    result.prefix_json = prefix_json
    result.prefix_reused = reused

    selected = [v for v in variants if only is None or v.id in set(only)]
    if only is not None:
        unknown = set(only) - {v.id for v in variants}
        if unknown:
            raise ValueError(f"--only names unknown variant(s): "
                             f"{', '.join(sorted(unknown))}")
    log.info("experiment %s: %d/%d variants, root=%s%s",
             exp_id, len(selected), len(variants), root,
             " [DRY RUN]" if dry_run else "")

    done: Dict[str, VariantResult] = {}
    for v in selected:
        vr = VariantResult(id=v.id, label=v.label, method=v.method)
        t0 = time.perf_counter()
        try:
            _run_one(v, resolved[v.id], vr,
                     input_=str(input_), base=base, root=root,
                     prefix_json=prefix_json, data_dir=data_dir,
                     done=done, dry_run=dry_run)
        except Exception as exc:
            # One dead arm must not cost the others: report and carry on.
            # `ExperimentResult.ok` is how a caller notices.
            log.exception("variant %s failed: %s", v.id, exc)
            vr.error = f"{type(exc).__name__}: {exc}"
        vr.wall_clock_s = round(time.perf_counter() - t0, 3)
        done[v.id] = vr
        result.variants.append(vr)

    result.wall_clock_s = time.perf_counter() - t_start
    if not dry_run:
        result.write(root / "experiment.json")
    n_bad = sum(1 for v in result.variants if v.error)
    log.info("experiment %s: %d variant(s) in %.1fs, %d failed",
             exp_id, len(result.variants), result.wall_clock_s, n_bad)
    return result


def _run_one(
    v: Variant,
    cfg: dict,
    vr: VariantResult,
    *,
    input_: str,
    base: str,
    root: Path,
    prefix_json: Optional[Path],
    data_dir: Path,
    done: Dict[str, VariantResult],
    dry_run: bool,
) -> None:
    variant_dir = Path(root) / v.id
    variant_dir.mkdir(parents=True, exist_ok=True)
    vr.dir = variant_dir
    plan_json = variant_dir / "plan.mt.json"
    vr.plan_json = plan_json

    # --- the source this variant's merged JSON descends from ---------------
    donor_vr: Optional[VariantResult] = None
    if v.reuse_tts_from:
        donor_vr = done.get(v.reuse_tts_from)
        if donor_vr is None:
            donor_dir = Path(root) / v.reuse_tts_from
            if not (donor_dir / "plan.mt.json").exists():
                raise RuntimeError(
                    f"variant {v.id!r} reuses TTS from {v.reuse_tts_from!r}, "
                    f"which has not run and left nothing at {donor_dir}")
            donor_vr = VariantResult(id=v.reuse_tts_from, label="", method="")
            donor_vr.dir = donor_dir
            donor_vr.plan_json = donor_dir / "plan.mt.json"
        if donor_vr.error:
            raise RuntimeError(f"variant {v.id!r} reuses TTS from "
                               f"{v.reuse_tts_from!r}, which failed")
        vr.reused_tts_from = v.reuse_tts_from
    source_json = donor_vr.plan_json if donor_vr is not None else prefix_json

    cfg_out = dict(cfg)
    cfg_out_doc = {
        "schema_version": SCHEMA_VERSION,
        "exp_id": root.name,
        "base": base,
        "variant": v.to_jsonable(),
        "resolved": cfg_out,
        "source_json": str(source_json) if source_json else None,
        "paths": {
            "dir": str(variant_dir),
            "plan_json": str(plan_json),
            "timing_dir": str(variant_dir / "timing"),
            "tts_dir": str(_tts_dir_for(variant_dir, base,
                                        (cfg.get("pipeline") or {}).get("tts_model"))),
            "mix_dir": str(variant_dir / "mix"),
            "metrics_json": str(variant_dir / "metrics.json"),
        },
    }
    (variant_dir / "variant.json").write_text(
        json.dumps(cfg_out_doc, indent=2, ensure_ascii=False))

    if dry_run:
        vr.skipped = "dry_run"
        log.info("variant %s [dry-run]: config resolved, tree laid out", v.id)
        return
    if source_json is None:
        raise RuntimeError(f"variant {v.id!r}: no merged JSON to start from")

    pipe_cfg = cfg.get("pipeline") or {}
    tts_cfg = cfg.get("tts")
    timing_cfg = cfg.get("timing")
    mix_cfg = cfg.get("mix")
    tts_model = pipe_cfg.get("tts_model") or (tts_cfg or {}).get("tts_model")

    # --- 1. bootstrap + post-MT chain --------------------------------------
    # `variant_tag` copies `source_json` to `<exp_root>/<base>/<id>/
    # plan.mt.json` and redirects stages 9b..9h into that directory.
    boot: Optional[PipelineResult] = None
    stage_keys = ("translit_model", "duration_adjust", "voice_engine",
                  "tts_model")
    live_stages = ([] if donor_vr is not None
                   else [k for k in stage_keys if pipe_cfg.get(k)])
    # The bootstrap runs for either of two reasons: to run a post-MT stage,
    # or just to learn where the source video and M&E stem are so the mix can
    # be driven. A placement-only variant needs neither.
    needs_pipeline = bool(live_stages) or mix_cfg is not None
    if not needs_pipeline:
        if Path(source_json).resolve() != plan_json.resolve():
            shutil.copy2(source_json, plan_json)
        log.info("variant %s: placement only, no pipeline stage to run", v.id)
    else:
        run_kwargs = dict(pipe_cfg)
        # Stages 7/8 live in `run_pipeline`'s non-replay branch and this call
        # always passes `post_mt_only_from`, so ASR/align selection forwarded
        # here is accepted and ignored. The prefix owns them; `ensure_prefix`
        # is where they were used.
        for k in _PREFIX_ONLY_KNOBS:
            run_kwargs.pop(k, None)
        if donor_vr is not None:
            # The donor already ran 9b..9e and this variant inherited its
            # merged JSON. Re-running them would cost a second TTS pass and
            # overwrite the clips being reused.
            run_kwargs.update(translit_model=None, duration_adjust=False,
                              voice_engine=None, tts_model=None)
        run_kwargs["mix"] = False          # the timing plan has to land first
        run_kwargs["lipsync_model"] = None
        if tts_cfg is not None and cfg.get("resynthesis_reason"):
            # The knob that forced this is not forwarded by the pipeline, so
            # let the pipeline do voices (9d) and pay for the extra pass.
            log.warning("variant %s: %s; one extra TTS pass",
                        v.id, cfg["resynthesis_reason"])
        video = _resolve_video(input_, base, data_dir)
        # `run_pipeline` builds its variant directory as
        # `<exp_root>/<video.stem>/<variant_tag>`, so its tree and ours only
        # coincide when the experiment directory is named after the clip.
        # Checked, not assumed: otherwise every artifact is silently written
        # twice, one directory apart.
        if video is not None and video.stem != Path(root).name:
            raise ValueError(
                f"exp_id {Path(root).name!r} does not match the clip name "
                f"{video.stem!r}; stages driven through run_pipeline would "
                f"write to {Path(root).parent / video.stem / v.id} instead of "
                f"{variant_dir}. Drop --exp-id, or name it after the clip.")
        boot = run_pipeline(
            str(video) if video is not None else input_,
            data_dir=data_dir,
            post_mt_only_from=Path(source_json),
            variant_tag=v.id,
            exp_root=Path(root).parent,
            **run_kwargs,
        )
        if boot.mt_json is not None:
            plan_json = Path(boot.mt_json)
            vr.plan_json = plan_json

    # --- 2. TTS reuse, or the extra synthesis pass -------------------------
    if donor_vr is not None:
        donor_tts = _tts_dir_for(donor_vr.dir, base, tts_model)
        if (donor_tts / "raw").is_dir():
            own_tts = _tts_dir_for(variant_dir, base, tts_model)
            raw_link = _link_tts(donor_tts, own_tts)
            n = _repoint_clips(plan_json, donor_tts / "raw", raw_link)
            vr.tts_dir = own_tts
            log.info("variant %s: reused %d clip path(s) from %s via %s",
                     v.id, n, v.reuse_tts_from, raw_link)
        elif _has_clips(plan_json):
            # The donor ran no TTS pass of its own: the clips it is sharing
            # came in with the prefix and both variants inherited the same
            # read-only paths. Nothing to link, and neither writes there.
            log.info("variant %s: donor %s has no TTS pass; using the clips "
                     "inherited with the prefix", v.id, v.reuse_tts_from)
            vr.reused_tts_from = f"{v.reuse_tts_from} (inherited prefix clips)"
        else:
            raise FileNotFoundError(
                f"reuse_tts_from: donor {v.reuse_tts_from!r} has no raw clips "
                f"at {donor_tts / 'raw'}, and {plan_json.name} carries none "
                f"either; the donor's TTS stage did not run")
    elif tts_cfg is not None and cfg.get("resynthesis_reason"):
        from .tts import synthesize_segments
        own_tts = _tts_dir_for(variant_dir, base, tts_model)
        voice_map = _speaker_voice_map(plan_json)
        synthesize_segments(
            plan_json,
            tts_model=tts_model,
            language=pipe_cfg.get("mt_target_language"),
            speaker_voice_map=voice_map,
            out_dir=own_tts,
            **{k: val for k, val in tts_cfg.items()
               if k not in ("tts_model",)},
        )
        vr.tts_dir = own_tts
    elif tts_model:
        vr.tts_dir = _tts_dir_for(variant_dir, base, tts_model)

    # --- 3. timing plan ----------------------------------------------------
    tplan = None
    if timing_cfg is not None:
        from . import timing as _t
        from . import timing_advanced as _adv   # noqa: F401  (registration)
        producer = timing_cfg["producer"]
        timing_dir = variant_dir / "timing"
        timing_dir.mkdir(parents=True, exist_ok=True)
        out_plan = timing_dir / f"{producer}.plan.json"
        tplan = _t.plan(producer, plan_json, **timing_cfg["params"])
        tplan.write(out_plan)
        _t.apply_plan_to_merged(tplan, plan_json)
        vr.timing_plan_json = out_plan
        vr.summary["timing"] = tplan.summary
        log.info("variant %s: timing[%s] -> %s", v.id, producer, out_plan.name)

    # --- 4. mix ------------------------------------------------------------
    # Called directly: `run_post_mt_stages` runs TTS and mix in one pass, and
    # the timing plan is built between them.
    if mix_cfg is not None:
        from .mix import mix_dubbed_audio
        if boot is None or boot.video is None:
            raise RuntimeError(
                f"variant {v.id!r}: mix needs the source video, which only the "
                f"pipeline bootstrap can resolve; give this variant a "
                f"'pipeline' section or drop its 'mix' section")
        mr = mix_dubbed_audio(
            plan_json,
            me_stem=boot.me_stem,
            source_video=boot.video,
            out_dir=variant_dir / "mix",
            timing_plan=vr.timing_plan_json,
            **mix_cfg,
        )
        vr.mix_wav = mr.audio_wav
        vr.mix_mp4 = mr.video_mp4

    # --- 5. metrics --------------------------------------------------------
    vr.metrics_json = _write_metrics(vr, cfg, variant_dir=variant_dir, base=base)


def _speaker_voice_map(plan_json: Path) -> Dict[str, dict]:
    """Rebuild stage 9d's map from what the previous TTS pass left on disk.

    Only used for the extra synthesis pass; the voices directory next to the
    plan is the same one `run_post_mt_stages` just wrote.
    """
    data = json.loads(Path(plan_json).read_text())
    refs_dir = Path(plan_json).parent / "voices"
    by_spk: Dict[str, dict] = {}
    for cand in sorted(refs_dir.glob("*.refs/*.wav")) if refs_dir.is_dir() else []:
        by_spk[cand.stem] = {"reference_audio": str(cand)}
    engine = (data.get("extra") or {}).get("tts", {}).get("model")
    speakers = {seg.get("speaker") for seg in data.get("segments", [])}
    out: Dict[str, dict] = {}
    for spk in sorted(s for s in speakers if s):
        info = dict(by_spk.get(spk) or {})
        info.setdefault("engine", engine)
        info.setdefault("voice", None)
        out[spk] = info
    return out


def _write_metrics(
    vr: VariantResult, cfg: dict, *, variant_dir: Path, base: str,
) -> Path:
    """Per-variant `metrics.json`: always the cheap facts, eval scores if possible.

    The eval harness is optional: it needs golden references the clip may not
    have, and a missing reference must not turn a finished dub into a failed
    variant.
    """
    doc: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "variant_id": vr.id,
        "label": vr.label,
        "method": vr.method,
        "reused_tts_from": vr.reused_tts_from,
        "plan_json": str(vr.plan_json) if vr.plan_json else None,
        "timing_plan_json": (str(vr.timing_plan_json)
                             if vr.timing_plan_json else None),
        "mix_wav": str(vr.mix_wav) if vr.mix_wav else None,
        "resolved_config": cfg,
        "timing_summary": vr.summary.get("timing"),
        "eval": None,
        "eval_error": None,
    }
    if vr.plan_json and Path(vr.plan_json).exists():
        data = json.loads(Path(vr.plan_json).read_text())
        extra = data.get("extra") or {}
        doc["tts_summary"] = extra.get("tts")
        doc["mix_summary"] = extra.get("mix")
        doc["n_segments"] = len(data.get("segments") or [])

    if vr.mix_wav and Path(vr.mix_wav).exists():
        try:
            from .eval.harness import VariantInputs, score_variant
            doc["eval"] = score_variant(
                base, vr.id,
                VariantInputs(mt_json=Path(vr.plan_json), mix_wav=Path(vr.mix_wav)),
                write=False,
            )
        except Exception as exc:
            doc["eval_error"] = f"{type(exc).__name__}: {exc}"
            log.info("variant %s: metrics without eval (%s)", vr.id, exc)

    dest = Path(variant_dir) / "metrics.json"
    dest.write_text(json.dumps(doc, indent=2, ensure_ascii=False, default=str))
    return dest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    """python -m backend.experiment run --suite S --input X [--only a,b] [--dry-run]"""
    import argparse

    ap = argparse.ArgumentParser(prog="backend.experiment")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a suite of variants over one clip")
    r.add_argument("--suite", required=True, help="suite JSON (see experiments/suites)")
    r.add_argument("--input", required=True,
                   help="video path, URL, or an existing merged *.mt.json")
    r.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    r.add_argument("--exp-id", default=None,
                   help="output directory name under data/exp (default: clip name)")
    r.add_argument("--only", default=None,
                   help="comma-separated variant ids to run")
    r.add_argument("--no-reuse-prefix", action="store_true",
                   help="rebuild the shared ASR/align/MT prefix instead of "
                        "reusing the cached one")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--allow-insecure-env", action="store_true",
                   help="Skip the .env file-permission check on startup.")
    r.add_argument("--log-level", default="INFO")

    c = sub.add_parser("validate", help="resolve a suite's configs and exit")
    c.add_argument("--suite", required=True)
    c.add_argument("--log-level", default="WARNING")

    a = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, a.log_level.upper(), logging.INFO),
                        format="%(levelname)s %(name)s: %(message)s")

    # Same startup contract as `backend.cli.main`: secrets from `.env`
    # (HF_TOKEN for pyannote) and an `ffmpeg` on PATH for demucs.
    # `validate` touches neither and runs without a `.env`.
    if a.cmd == "run":
        _load_env(allow_insecure_env=a.allow_insecure_env)
        _ensure_ffmpeg_on_path()

    base_config, variants = load_suite(Path(a.suite))

    if a.cmd == "validate":
        report = []
        for v in variants:
            cfg = resolve_config(v, base_config)
            report.append({
                "id": v.id, "method": v.method, "label": v.label,
                "reuse_tts_from": v.reuse_tts_from,
                "sections": {k: (None if cfg.get(k) is None else "set")
                             for k in SECTIONS},
                "resynthesis_reason": cfg.get("resynthesis_reason"),
            })
        print(json.dumps({"suite": a.suite, "variants": report},
                         indent=2, ensure_ascii=False))
        return 0

    res = run_experiment(
        a.input,
        variants=variants,
        base_config=base_config,
        data_dir=Path(a.data_dir),
        exp_id=a.exp_id,
        reuse_prefix=not a.no_reuse_prefix,
        only=[s.strip() for s in a.only.split(",")] if a.only else None,
        dry_run=a.dry_run,
    )
    print(json.dumps(res.to_jsonable(), indent=2, ensure_ascii=False))
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
