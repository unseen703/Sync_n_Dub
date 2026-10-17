"""CLI for the dubbing evaluation harness.

    python -m backend.eval refbuild  ...   # build golden references
    python -m backend.eval goldcheck ...   # QC those references
    python -m backend.eval run       ...   # score one variant
    python -m backend.eval sweep     ...   # score many
    python -m backend.eval report    ...   # aggregate to markdown
    python -m backend.eval calibrate ...   # fit the duration model

`refbuild`, `goldcheck` and `calibrate` are thin pass-throughs to the module
that owns them, so their own `--help` is the authority on their flags. They
forward argv untouched rather than re-declaring it here, which would mean two
copies of one interface drifting apart.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)


def _delegate(module: str, func: str, argv: List[str], *, prefix: List[str] | None = None) -> int:
    try:
        import importlib
        mod = importlib.import_module(module)
    except Exception as e:
        print(f"error: {module} is not importable ({type(e).__name__}: {e}).",
              file=sys.stderr)
        return 2
    fn = getattr(mod, func, None)
    if fn is None:
        print(f"error: {module}.{func} is not defined.", file=sys.stderr)
        return 2
    return int(fn((prefix or []) + list(argv)) or 0)


def _summary_table(results: List[dict]) -> str:
    """One line per scored variant, for the terminal. Full detail is in JSON."""
    from backend.eval import report as rpt

    keys = ("median_abs_start_ms", "speech_iou", "pause_f1",
            "drift_pct", "overflow_ratio")
    specs = [rpt._SPEC_BY_KEY[k] for k in keys if k in rpt._SPEC_BY_KEY]
    head = ["clip", "variant", "segs", "match", "inv", "miss"] + [s.key for s in specs] + ["n/a"]
    rows = []
    for r in results:
        cells = []
        for s in specs:
            v = rpt.metric_value(r, s)
            cells.append("n/a" if v is None else f"{v:.{s.digits}f}")
        rows.append([str(r.get("clip_id")), str(r.get("variant_id")),
                     str(r.get("n_segments", 0)), str(r.get("n_matched", 0)),
                     str(r.get("n_invented", 0)), str(r.get("n_missed", 0))]
                    + cells + [str(len(r.get("not_measured") or []))])
    widths = [max(len(head[i]), *(len(row[i]) for row in rows)) if rows else len(head[i])
              for i in range(len(head))]
    out = ["  ".join(h.ljust(w) for h, w in zip(head, widths)),
           "  ".join("-" * w for w in widths)]
    for row in rows:
        out.append("  ".join(c.ljust(w) for c, w in zip(row, widths)))
    return "\n".join(out)


# Subcommands whose flags belong to the module that owns them, not to this
# dispatcher. They are peeled off argv BEFORE argparse sees it: an
# `argparse.REMAINDER` positional does not reliably capture a first token that
# looks like an option, so `... refbuild --only X` died with
# "unrecognized arguments: --only" and neither documented refbuild/goldcheck
# command could run at all.
_PASSTHROUGH = {
    "refbuild": ("backend.eval.reference", "main", ["refbuild"]),
    "goldcheck": ("backend.eval.reference", "main", ["goldcheck"]),
    "calibrate": ("backend.duration", "main", ["fit"]),
    "report": ("backend.eval.report", "main", []),
}


def main(argv: Optional[List[str]] = None) -> int:
    args_in = list(sys.argv[1:] if argv is None else argv)
    head = next((a for a in args_in if not a.startswith("-")), None)
    if head in _PASSTHROUGH:
        module, func, prefix = _PASSTHROUGH[head]
        rest = args_in[args_in.index(head) + 1:]
        # A lone `--` right after the subcommand is this dispatcher's own
        # escape hatch; the delegate never needs to see it.
        if rest[:1] == ["--"]:
            rest = rest[1:]
        verbose = any(a in ("-v", "--verbose") for a in args_in[:args_in.index(head)])
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        )
        return _delegate(module, func, rest, prefix=prefix)

    ap = argparse.ArgumentParser(
        prog="python -m backend.eval",
        description="Golden-reference build, variant scoring and reporting "
                    "for the dubbing timing-alignment experiment.",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name, helptext in (
        ("refbuild", "Build golden references (backend.eval.reference)."),
        ("goldcheck", "QC the golden references (backend.eval.reference)."),
        ("calibrate", "Fit the TTS duration model (backend.duration fit)."),
    ):
        p = sub.add_parser(name, help=helptext, add_help=False)
        p.add_argument("rest", nargs=argparse.REMAINDER,
                       help="Passed straight through; try `... "
                            f"{name} --help`.")

    r = sub.add_parser("run", help="Score one variant of one clip.")
    r.add_argument("--clip-id", required=True)
    r.add_argument("--variant-id", required=True)
    r.add_argument("--mt-json", type=Path, required=True)
    r.add_argument("--mix-wav", type=Path, default=None)
    r.add_argument("--mix-mp4", type=Path, default=None)
    r.add_argument("--tts-only-wav", type=Path, default=None)
    r.add_argument("--groups", default=",".join(("timing", "pauses", "rate")),
                   help="Comma-separated subset of "
                        "timing,pauses,rate,audio,intel,mt (or 'all'). "
                        "timing,pauses,rate decode no audio and import no torch.")
    r.add_argument("--golden-root", type=Path, default=Path("data/golden"))
    r.add_argument("--out-dir", type=Path, default=Path("data/eval"))
    r.add_argument("--print-json", action="store_true")

    s = sub.add_parser("sweep", help="Score many variants.")
    s.add_argument("--mt-json", nargs="+", default=["data/mt/*.mt.json"],
                   help="Manifest path(s) or glob(s); quote globs.")
    s.add_argument("--clip", action="append", dest="clips", default=None,
                   help="Restrict to these clip ids (repeatable).")
    s.add_argument("--groups", default=",".join(("timing", "pauses", "rate")))
    s.add_argument("--jobs", type=int, default=1)
    s.add_argument("--golden-root", type=Path, default=Path("data/golden"))
    s.add_argument("--out-dir", type=Path, default=Path("data/eval"))
    s.add_argument("--all-manifests", action="store_true",
                   help="Include manifests with no synthesized audio. Off by "
                        "default: text-stage runs produce empty rows.")

    rep = sub.add_parser("report", help="Aggregate results into markdown.",
                         add_help=False)
    rep.add_argument("rest", nargs=argparse.REMAINDER)

    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    if args.cmd in ("refbuild", "goldcheck"):
        return _delegate("backend.eval.reference", "main", args.rest,
                         prefix=[args.cmd])
    if args.cmd == "calibrate":
        # backend.duration owns the fit; `calibrate` is just its front door
        # from this CLI so there is one entry point for the experiment.
        return _delegate("backend.duration", "main", args.rest, prefix=["fit"])
    if args.cmd == "report":
        return _delegate("backend.eval.report", "main", args.rest)

    from backend.eval import harness

    groups = (list(harness.GROUPS) if args.groups.strip() == "all"
              else [g.strip() for g in args.groups.split(",") if g.strip()])
    bad = [g for g in groups if g not in harness.GROUPS]
    if bad:
        ap.error(f"unknown --groups value(s) {bad}; "
                 f"choose from {list(harness.GROUPS)} or 'all'")

    if args.cmd == "run":
        res = harness.score_variant(
            args.clip_id, args.variant_id,
            harness.VariantInputs(mt_json=args.mt_json, mix_wav=args.mix_wav,
                                  mix_mp4=args.mix_mp4,
                                  tts_only_wav=args.tts_only_wav),
            golden_root=args.golden_root, groups=groups, out_dir=args.out_dir,
        )
        if args.print_json:
            print(json.dumps(res, indent=2, ensure_ascii=False, default=str))
        else:
            print(_summary_table([res]))
            for e in res.get("not_measured") or []:
                print(f"  not measured: {e['key']}: {e['reason']}")
            for e in res.get("errors") or []:
                print(f"  ERROR: {e['key']}: {e['error']}")
        print(f"wrote {res.get('result_path')}")
        return 0

    specs = harness.discover_variants(
        args.mt_json, require_tts=not args.all_manifests,
        golden_root=args.golden_root)
    if not specs:
        print("no manifests matched "
              f"{args.mt_json}"
              + ("" if args.all_manifests else
                 " that carry synthesized audio (pass --all-manifests to "
                 "include text-stage runs)"), file=sys.stderr)
        return 1
    results = harness.sweep(args.clips, specs, groups=groups, jobs=args.jobs,
                            golden_root=args.golden_root, out_dir=args.out_dir)
    print(_summary_table(results))
    print(f"\n{len(results)} result(s) under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
