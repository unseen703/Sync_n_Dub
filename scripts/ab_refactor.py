#!/usr/bin/env python
"""Dump the observable behaviour of the de-duplicated code, as JSON.

A refactor that claims to change nothing should produce a byte-identical dump
before and after. This exists so that claim is checkable rather than asserted:

    $PY scripts/ab_refactor.py > after.json
    git stash push -- backend models ui          # tracked changes only
    $PY scripts/ab_refactor.py > before.json     # this file is untracked, it survives
    git stash pop
    diff <(jq -S . before.json) <(jq -S . after.json)

Empty diff means every caller sees what it saw. A non-empty one is the list
of things to justify one by one.

What is captured, and why each one:

  registries   Every registry's names, the EXACT ValueError text for an
               unregistered name, the EXACT ImportError text for each
               registered one, and the getter's signature. The messages are
               what an operator reads when a `--lipsync-model` typo reaches
               the CLI, so they are behaviour, not decoration. The
               ImportError is forced by making `importlib.import_module`
               raise, which both the old per-stage code and the shared
               factory route through — so the same probe works on either side.

  probes       `probe_video` / `probe_duration` over whatever small media the
               repo already has. Values, not just "it ran": a probe that
               starts answering 0.0 is the silent-failure mode CLAUDE.md §1
               is about.

  read_json    The ui helpers' answer for present / malformed / missing, so
               the `{}`-vs-`None` divergence between the four copies shows up
               as a diff rather than as a surprise at a call site.

  edits        The op catalogue as the LLM receives it. Phase 1.2 edits those
               strings deliberately, so this section is EXPECTED to differ —
               it is here to show exactly how much, not to come out clean.

Read-only: it imports, it never renders, downloads or calls a network API.
"""
from __future__ import annotations

import importlib
import inspect
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _err(fn, *a, **kw) -> str:
    """Call something that should raise; return "Type: message"."""
    try:
        fn(*a, **kw)
    except Exception as e:                      # noqa: BLE001 — that is the point
        return f"{type(e).__name__}: {e}"
    return "<no exception raised>"


#: (import path, getter name, lister name) per registry.
_REGISTRIES = [
    ("models.align",        "get_aligner",        "available_models"),
    ("models.asd",          "get_asd",            "available_models"),
    ("models.asr",          "get_asr",            "available_models"),
    ("models.mt",           "get_mt",             "available_models"),
    ("models.tts",          "get_tts",            "available_models"),
    ("models.translit",     "get_transliterator", "available_models"),
    ("models.lipsync",      "get_lipsync",        "available_models"),
    ("models.diarization",  "get_diarizer",       "available_models"),
    ("models.emotion",      "get_emotion",        "available_models"),
    ("models.faces.detect", "get_detector",       "available_detectors"),
    ("models.faces.track",  "get_tracker",        "available_trackers"),
]


def capture_registries() -> dict:
    out: dict = {}
    for mod_path, getter_name, lister_name in _REGISTRIES:
        entry: dict = {}
        try:
            mod = importlib.import_module(mod_path)
        except Exception as e:                  # noqa: BLE001
            out[mod_path] = {"import_failed": f"{type(e).__name__}: {e}"}
            continue

        getter = getattr(mod, getter_name, None)
        lister = getattr(mod, lister_name, None)
        entry["getter"] = getter_name
        entry["lister"] = lister_name
        entry["names"] = sorted(getattr(mod, "_REGISTRY_SPEC", {}))
        entry["available"] = list(lister()) if lister else None
        entry["signature"] = str(inspect.signature(getter)) if getter else None
        entry["exports"] = sorted(getattr(mod, "__all__", []))

        # The unregistered-name message.
        entry["unknown_name_error"] = _err(getter, "definitely-not-a-model")

        # The will-not-import message, one per registered name. Forcing it
        # beats waiting for an env that happens to lack the package: this
        # has to be reproducible on a host where everything IS installed.
        real = importlib.import_module

        def _boom(*a, **kw):
            raise ImportError("synthetic: forced by ab_refactor")

        importlib.import_module = _boom          # type: ignore[assignment]
        try:
            entry["import_errors"] = {
                name: _err(getter, name) for name in entry["names"]
            }
        finally:
            importlib.import_module = real       # type: ignore[assignment]

        out[mod_path] = entry
    return out


def _media_samples(limit: int = 3) -> list:
    """Whatever small video the repo already has. Sorted, so the pick is
    stable between the two runs; capped, so this stays fast."""
    cands: list[Path] = []
    for pat in ("data/clips/*.mp4", "data/videos/**/*.mp4", "tests/fixtures/*.mp4"):
        cands.extend(sorted(ROOT.glob(pat)))
    return [p for p in cands if p.is_file()][:limit]


def capture_probes() -> dict:
    out: dict = {"samples": [], "probe_video": {}, "probe_duration": {},
                 "gpu_probe_duration": {}}
    samples = _media_samples()
    out["samples"] = [str(p.relative_to(ROOT)) for p in samples]

    try:
        from backend import media_io
    except Exception as e:                       # noqa: BLE001
        out["media_io_import_failed"] = f"{type(e).__name__}: {e}"
        return out

    pv = getattr(media_io, "probe_video", None)
    out["probe_video_present"] = pv is not None

    # The adapters' own copies, while they still exist.
    for mod_path in ("models.lipsync.wav2lip_adapter",
                     "models.lipsync.musetalk_adapter",
                     "models.lipsync.latentsync_adapter"):
        try:
            m = importlib.import_module(mod_path)
            out.setdefault("adapter_local_probe_video", {})[mod_path] = (
                "_probe_video" in vars(m))
        except Exception as e:                   # noqa: BLE001
            out.setdefault("adapter_local_probe_video", {})[mod_path] = (
                f"import_failed: {type(e).__name__}")

    for p in samples:
        key = str(p.relative_to(ROOT))
        if pv:
            try:
                fps, dur = pv(p)
                out["probe_video"][key] = [round(fps, 6), round(dur, 6)]
            except Exception as e:               # noqa: BLE001
                out["probe_video"][key] = f"{type(e).__name__}: {e}"
        try:
            out["probe_duration"][key] = round(media_io.probe_duration(p), 6)
        except Exception as e:                   # noqa: BLE001
            out["probe_duration"][key] = f"{type(e).__name__}: {e}"

    # backend.gpu's second copy, for as long as it is there.
    try:
        from backend import gpu
        gpd = getattr(gpu, "probe_duration", None)
        out["gpu_probe_duration"]["present"] = gpd is not None
        if gpd:
            for p in samples:
                k = str(p.relative_to(ROOT))
                try:
                    out["gpu_probe_duration"][k] = round(gpd(p), 6)
                except Exception as e:           # noqa: BLE001
                    out["gpu_probe_duration"][k] = f"{type(e).__name__}: {e}"
            # The divergence that matters: a path that is not there.
            out["gpu_probe_duration"]["missing_file"] = _err(
                gpd, ROOT / "no-such-file.mp4")
    except Exception as e:                       # noqa: BLE001
        out["gpu_probe_duration"]["import_failed"] = f"{type(e).__name__}: {e}"

    out["media_io_probe_duration_missing_file"] = _err(
        media_io.probe_duration, ROOT / "no-such-file.mp4")
    return out


def capture_read_json(tmp: Path) -> dict:
    """present / malformed / missing, for each copy of the helper."""
    tmp.mkdir(parents=True, exist_ok=True)
    good = tmp / "good.json"; good.write_text('{"a": 1}')
    bad = tmp / "bad.json"; bad.write_text("{not json")
    missing = tmp / "nope.json"

    out: dict = {}
    for mod_path in ("ui.speakers", "ui.server", "ui.media", "ui.routes_edits",
                     "ui._io"):
        try:
            m = importlib.import_module(mod_path)
        except Exception as e:                   # noqa: BLE001
            out[mod_path] = {"import_failed": f"{type(e).__name__}: {e}"}
            continue
        fn = getattr(m, "_read_json", None) or getattr(m, "read_json", None)
        if fn is None:
            out[mod_path] = {"helper": None}
            continue
        res = {}
        for label, p in (("present", good), ("malformed", bad),
                         ("missing", missing)):
            try:
                res[label] = repr(fn(p))
            except Exception as e:               # noqa: BLE001
                res[label] = f"{type(e).__name__}: {e}"
        out[mod_path] = {"helper": fn.__name__, "answers": res}
    return out


def capture_edits() -> dict:
    """The op catalogue exactly as the LLM is handed it."""
    out: dict = {}
    try:
        from backend import edits as E
    except Exception as e:                       # noqa: BLE001
        return {"import_failed": f"{type(e).__name__}: {e}"}
    out["ops"] = sorted(E.OPS)
    out["specs"] = {
        name: {
            "dirty_from": s.dirty_from,
            "params": dict(s.params),
            "summary": s.summary,
            "examples": list(s.examples),
            "needs_llm": s.needs_llm,
        }
        for name, s in sorted(E.OPS.items())
    }
    try:
        out["action_catalogue"] = E.action_catalogue()
    except Exception as e:                       # noqa: BLE001
        out["action_catalogue"] = f"{type(e).__name__}: {e}"
    try:
        from backend import edit_agent as A
        out["system_prompt_sha"] = __import__("hashlib").sha256(
            A._SYSTEM.encode()).hexdigest()[:16]
        out["system_prompt_len"] = len(A._SYSTEM)
    except Exception as e:                       # noqa: BLE001
        out["system_prompt_sha"] = f"{type(e).__name__}: {e}"
    return out


def capture_bodhan() -> dict:
    """Adapter construction surface: what each Bodhan adapter resolves from
    the environment, without making a single request. Keys are redacted to a
    boolean — this dump is pasted into terminals."""
    out: dict = {}
    specs = [
        ("models.asr.bodhan_adapter", "BodhanASR"),
        ("models.mt.bodhan_adapter", "BodhanMT"),
        ("models.tts.bodhan_adapter", "BodhanTTS"),
        ("models.translit.bodhan_adapter", "BodhanTransliterate"),
    ]
    for mod_path, cls_name in specs:
        try:
            m = importlib.import_module(mod_path)
            cls = getattr(m, cls_name)
            obj = cls()
            out[cls_name] = {
                "base_url": obj.base_url,
                "model": obj.model,
                "api_key_present": bool(getattr(obj, "api_key", None)),
                "timeout_s": obj.timeout_s,
                "max_retries": obj.max_retries,
                "min_interval_s": obj.min_interval_s,
                "max_backoff_s": obj.max_backoff_s,
                "init_signature": str(inspect.signature(cls.__init__)),
                # The pure function, sampled rather than described.
                "backoff": [obj._backoff(i, None) for i in range(6)],
                "backoff_retry_after": [
                    obj._backoff(0, "7"), obj._backoff(0, "0.1"),
                    obj._backoff(3, "nonsense"),
                ],
                "ensure_loaded_without_key": _err(obj._ensure_loaded)
                if not getattr(obj, "api_key", None) else "<key present>",
            }
        except Exception as e:                   # noqa: BLE001
            out[cls_name] = {"failed": f"{type(e).__name__}: {e}",
                             "trace": traceback.format_exc(limit=3)}
    return out


def capture_speed() -> dict:
    """The resolved timing knobs on both the CLI and the UI side."""
    out: dict = {}
    try:
        from ui.presets import UI_TIMING_DEFAULTS
        out["ui_timing_defaults"] = UI_TIMING_DEFAULTS
    except Exception as e:                       # noqa: BLE001
        out["ui_timing_defaults"] = f"{type(e).__name__}: {e}"
    try:
        from backend import speed
        out["backend_speed_module"] = True
        out["segment_setup"] = speed.segment_timing_setup()._asdict() \
            if hasattr(speed.segment_timing_setup(), "_asdict") \
            else repr(speed.segment_timing_setup())
    except Exception:                            # noqa: BLE001
        out["backend_speed_module"] = False
    try:
        from ui import presets
        for name in ("draft", "standard", "best"):
            kw = presets.build_kwargs(name, target_language="hi")
            out.setdefault("build_kwargs", {})[name] = {
                k: kw[k] for k in sorted(kw)
                if k in ("timing_producer", "timing_params", "tts_apply_stretch",
                         "mix_strategy", "mix_clip_source", "mix_tempo_cap")
            }
    except Exception as e:                       # noqa: BLE001
        out["build_kwargs"] = f"{type(e).__name__}: {e}"
    return out


def main() -> int:
    tmp = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / ".ab_tmp"
    doc = {
        "registries": capture_registries(),
        "probes": capture_probes(),
        "read_json": capture_read_json(tmp / "read_json"),
        "edits": capture_edits(),
        "bodhan": capture_bodhan(),
        "speed": capture_speed(),
    }
    json.dump(doc, sys.stdout, indent=2, sort_keys=True, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
