"""Fetch LLM usage, cost and latency out of Langfuse over its REST API.

Why a script and not the dashboard: the Langfuse UI answers this fine, but
the numbers worth watching here are per-OPERATION (`edit_agent.parse` vs
`edit_agent.rewrite` vs `classify.register`), and a terminal is where the
rest of this repo is driven from.

    $PY scripts/langfuse_usage.py                 # last 7 days
    $PY scripts/langfuse_usage.py --days 30
    $PY scripts/langfuse_usage.py --by model      # group by model, not name
    $PY scripts/langfuse_usage.py --json          # machine-readable
    $PY scripts/langfuse_usage.py --failures      # only ERROR-level calls

AUTH NEEDS BOTH KEYS. Langfuse is HTTP Basic with the PUBLIC key as the
username and the SECRET key as the password, for reading exactly as for
writing — a secret key alone reads nothing. The public key is not a secret;
it is in Langfuse → Project Settings → API Keys next to the one-time
`sk-lf-…`. See `backend/tracing.py` for the same note.

This reads `observations` (the verified endpoint) and aggregates locally,
rather than the Metrics v2 API. Two reasons: the row-level data carries
`latency` and `cost_details` per call, which is what a question like "why
did that edit cost 3 cents" needs; and local aggregation cannot be wrong
about a field name in a beta API. `--raw-metrics` passes a Metrics v2 query
straight through for anyone who wants server-side rollups instead.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# `.env` explicitly: bare `load_dotenv()` searches upward from the CALLING
# file, so run from outside the repo it silently finds nothing and the
# failure reads as "no keys configured". Same convention as
# `scripts/benchmark_gender.py`.
from dotenv import load_dotenv                                  # noqa: E402
load_dotenv(ROOT / ".env")

from backend import tracing                                     # noqa: E402


#: core is always included; `basic` carries the name, `model` the model id,
#: `usage` the token/cost columns and `metrics` the latency.
FIELD_GROUPS = "core,basic,model,usage,metrics"


def _fmt_usd(x: float) -> str:
    if not x:
        return "-"
    return f"${x:.4f}" if x < 1 else f"${x:.2f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=int, default=7,
                    help="how far back to look (default 7)")
    ap.add_argument("--by", choices=("name", "model"), default="name",
                    help="group by operation name (default) or by model")
    ap.add_argument("--limit", type=int, default=1000,
                    help="rows per page (max 1000)")
    ap.add_argument("--max-pages", type=int, default=20,
                    help="stop after this many pages (default 20)")
    ap.add_argument("--failures", action="store_true",
                    help="only ERROR-level calls")
    ap.add_argument("--json", action="store_true", dest="as_json",
                    help="print the aggregate as JSON")
    ap.add_argument("--raw-metrics", default=None,
                    help="a Metrics v2 query (JSON string), passed through "
                         "to the server instead of aggregating locally")
    args = ap.parse_args()

    if not tracing.configured():
        print("Langfuse is not configured. Both LANGFUSE_PUBLIC_KEY and "
              "LANGFUSE_SECRET_KEY are required — Basic auth is "
              "public_key:secret_key. The public key is NOT a secret: it is "
              "in Langfuse → Project Settings → API Keys.", file=sys.stderr)
        return 2
    client = tracing.client()
    if client is None:
        return 2

    if args.raw_metrics:
        print(json.dumps(
            client.api.metrics.metrics(query=args.raw_metrics).dict(),
            indent=2, default=str))
        return 0

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=args.days)

    rows, cursor, pages = [], None, 0
    while pages < args.max_pages:
        page = client.api.observations.get_many(
            type="GENERATION", from_start_time=since, to_start_time=now,
            limit=min(args.limit, 1000), cursor=cursor,
            # `fields` is a comma-separated list of FIELD GROUPS, and the
            # default projection leaves out `model`, `usage` and `metadata`
            # — so every row comes back with usage_details=None and the
            # report reads "we spent nothing" rather than "you did not ask
            # for the column". Groups are defined in the server at
            # packages/shared/src/domain/observation-field-groups.ts.
            #
            # `io` is deliberately NOT requested: the prompts hold the dub's
            # dialogue and this report never prints it, so there is no
            # reason to pull it over the wire.
            fields=FIELD_GROUPS,
            **({"level": "ERROR"} if args.failures else {}),
        )
        got = list(page.data or [])
        rows.extend(got)
        pages += 1
        # Cursor pagination: stop on a short page or a missing next cursor,
        # whichever comes first, so a server that stops sending cursors does
        # not turn into an infinite loop.
        cursor = getattr(getattr(page, "meta", None), "next_cursor", None)
        if not cursor or len(got) < min(args.limit, 1000):
            break

    if not rows:
        print(f"No generations in the last {args.days} day(s). "
              f"If you expected some: tracing writes only when BOTH keys are "
              f"set, and a short-lived process must flush (backend/tracing.py "
              f"registers an atexit hook).")
        return 0

    agg = defaultdict(lambda: {"calls": 0, "errors": 0, "in": 0, "out": 0,
                               "cost": 0.0, "latencies": []})
    for o in rows:
        key = (o.name or "?") if args.by == "name" else (o.model or "?")
        a = agg[key]
        a["calls"] += 1
        if (o.level or "") == "ERROR":
            a["errors"] += 1
        ud = o.usage_details or {}
        a["in"] += int(ud.get("input") or 0)
        a["out"] += int(ud.get("output") or 0)
        # `cost_details` is what we sent from OpenRouter's `usage.cost`;
        # `total_cost` is Langfuse's own roll-up. Prefer ours — it is the
        # price actually charged, not a model-table estimate.
        #
        # The field is `total_cost` because `get_many` returns ObservationV2;
        # the older ObservationsView model (same package, different shape)
        # spells it `calculated_total_cost`, which is what this first used
        # and why it crashed on the first real read.
        cd = o.cost_details or {}
        a["cost"] += float(cd.get("total")
                           or getattr(o, "total_cost", None) or 0.0)
        if o.latency:
            a["latencies"].append(float(o.latency))

    out = {}
    for key, a in sorted(agg.items(), key=lambda kv: -kv[1]["cost"]):
        lat = sorted(a["latencies"])
        out[key] = {
            "calls": a["calls"], "errors": a["errors"],
            "tokens_in": a["in"], "tokens_out": a["out"],
            "cost_usd": round(a["cost"], 6),
            "p50_s": round(lat[len(lat) // 2], 3) if lat else None,
            "p95_s": round(lat[min(len(lat) - 1, int(len(lat) * 0.95))], 3) if lat else None,
        }

    if args.as_json:
        print(json.dumps({"since": since.isoformat(), "until": now.isoformat(),
                          "by": args.by, "groups": out}, indent=2))
        return 0

    width = max(len(k) for k in out) if out else 10
    print(f"Langfuse · last {args.days}d · {len(rows)} generation(s) · "
          f"grouped by {args.by}\n")
    print(f"{'group':{width}s} {'calls':>6s} {'err':>4s} {'tok in':>8s} "
          f"{'tok out':>8s} {'cost':>9s} {'p50':>7s} {'p95':>7s}")
    for key, v in out.items():
        # Built before the f-string: Python 3.10 cannot reuse a quote
        # character inside an f-string expression.
        p50 = f"{v['p50_s']:.2f}s" if v["p50_s"] else "-"
        p95 = f"{v['p95_s']:.2f}s" if v["p95_s"] else "-"
        print(f"{key:{width}s} {v['calls']:6d} {v['errors']:4d} "
              f"{v['tokens_in']:8d} {v['tokens_out']:8d} "
              f"{_fmt_usd(v['cost_usd']):>9s} {p50:>7s} {p95:>7s}")
    total = sum(v["cost_usd"] for v in out.values())
    errs = sum(v["errors"] for v in out.values())
    print(f"\ntotal {_fmt_usd(total)} over {len(rows)} call(s)"
          + (f", {errs} error(s)" if errs else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
