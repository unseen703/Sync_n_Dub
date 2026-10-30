# `ui/web` — the operator UI

Vite + React 18 + TypeScript + Tailwind + Radix + framer-motion + Zustand.
Builds to **`ui/web-dist/`**, which `ui/server.py` serves. With no build the
server still runs API-only, so a four-hour GPU run stays reachable over /api.

## Commands

Node lives in its own conda env. Use `conda run`, never `conda activate`:
`dubui` is build-time only and must never reach the Python server's PATH.

```bash
cd /data1/dipak_data/dubbingstudio/ui/web

conda run -n dubui --no-capture-output npm install          # once
conda run -n dubui --no-capture-output npm run build        # -> ../web-dist
conda run -n dubui --no-capture-output npm run typecheck    # tsc --noEmit

# dev server on :5173, proxying /api and /files to 127.0.0.1:7861
conda run -n dubui --no-capture-output npm run dev

# dev with every endpoint stubbed — no Python server needed
conda run -n dubui --no-capture-output npm run dev:mock
```

Then serve it the usual way:

```bash
/data1/dipak_data/miniconda3/envs/dubbing/bin/python ui/server.py
```

`npm run build` runs `tsc --noEmit` first. `npm run build:nocheck` skips it
when you need a bundle out of a tree that does not typecheck yet.

## Mock mode

`VITE_MOCK` decides which half of the API is real:

| value | reads | mutations |
|---|---|---|
| unset / `writes` | real server | `src/api/mock.ts` |
| `all` | mock | mock |
| `off` | real | real |

The default is `writes`, because `POST /api/youtube/probe`, `POST /api/jobs`,
`POST /api/jobs/{id}/cancel` and the four `/edits` routes do not exist on the
server yet. **Set `VITE_MOCK=off` the day they land.**

While anything is mocked the app shows a fixed banner at the top and an amber
`demo` chip on every control that routes to a stub. That is deliberate: a
submit button that reports success while nothing was queued is worse than a
disabled one.

## Layout

```
19rem JobRail │ Stage
```

The rail is permanent at ≥1024 px and an off-canvas drawer below it.

There was a third column on the right — `SpecPanel`, the run's settings —
until 2026-10-28. It was reference rather than something to act on: read
once per job, then dead width beside the video. `GET /api/jobs/{id}/spec` is
still fetched, because `TopBar` takes the dub's real title from it.

Job selection lives in the hash (`#/job/<id>`), so there is no SPA catch-all
route on the Python side and no react-router in the bundle.

## First render

Two things carry it, both in `index.html`:

1. An inline skeleton inside `#root` at the real layout's dimensions, with
   ~1.5 kB of critical CSS. First paint shows the shape of the app, and React
   arriving causes no layout shift. The skeleton's column order mirrors
   `AppShell` — rail left, stage centre — because a skeleton that disagrees
   makes the layout visibly swap sides at first paint.
2. `window.__BOOT__ = fetch('/api/jobs')` in `<head>`, awaited by
   `api.listJobs()`. The request overlaps bundle download, parse and hydrate
   instead of queueing behind them.

Check it with DevTools Network, cache disabled: eager JS should be ≈115 kB gz
across the `react` / `radix` / `motion` / app chunks, and the skeleton should
be in the first paint.

## Indic fonts

The old inspector defined `.lang-hi` and `.lang-ta` and applied them
**nowhere**, so Indic text rendered in whatever the browser fell back to.

The fix here keys on **script, not language** — 23 languages share 12 scripts,
and Hindi, Marathi, Nepali, Sanskrit, Konkani, Maithili, Bodo and Dogri all
want the same Devanagari file:

- `src/lib/lang.ts` maps language → script → family, mirroring
  `ui/langs.py::LANGUAGES` and `SCRIPT_FONTS`.
- `langClass(code)` goes on every element rendering target-language text.
- `ensureScriptFont(code)` injects the `@font-face` pair and the
  `.lang-<script>` rule for one script, the first time it is needed.

Self-hosted, no CDN — `public/fonts/<script>-{400,600}.woff2`, the
script-subset woff2 from the matching Noto family (the Latin ranges are
already covered by the UI font). All 24 files are present:

```
deva beng gujr guru knda mlym orya taml telu arab olck mtei
```

Replacing one means dropping in `<script>-400.woff2` and `<script>-600.woff2`
with the same names; nothing else changes. Until a file exists the class has
no rule and the text renders in the UI font — degraded, never broken, which
is what makes the lazy load safe.

## Where things are

```
src/
  api/        types.ts (the contract), client.ts (fetch), mock.ts, index.ts (the switch)
  store/      useStore.ts — all state, no fetching cadence
  hooks/      usePoll, useJobProgress, useHashRoute, useShortcuts, useNow
  lib/        lang, format, centre (the reveal order), audio (one shared element), cn
  components/
    ui/       Button, Card, Select, Dialog, Tooltip, StateBadge, ProgressBar, Toasts, Icons
    layout/   AppShell, BrandBar, JobRail, JobCard, TopBar, Stage, Landing, StartPanel
    progress/ StageRibbon, StageDetailList, LogTail
    video/    VideoStage, DualVideoPlayer, TrackToggle, AudioBar
    speakers/ SpeakerTable, SpeakerRow, VoicePicker, PendingBar, ApplyDialog
    newjob/   NewJobDialog, RangePicker
```

`src/api/types.ts` is the reconciliation document between this app and
`ui/*.py`. Every shape in it was transcribed from the Python that produces
it. When one of those changes, change this file in the same commit — a
silently-wrong type here shows up as an empty panel, not as an error.

## Rules that are load-bearing

- **Every conditional render keys off `progress.milestones`**, never off
  filename inspection. `ui/media.py` owns "which file is the lip-synced
  render" and has five ranked patterns for it; if this app ever re-derives
  that, the two will disagree.
- **Voice names, not ids.** Show `voice_options[].display` ("Swara"), never
  `hi-IN-SwaraNeural`.
- **The tone control is absent, not disabled**, when `supports_tone` is
  false.
- **There is no user/admin view switch** (removed 2026-10-28). It was
  presentation only, never a permission — this app has no auth and every
  route is open to anyone who can reach the port. Tracebacks, run config,
  face measurements and raw voice overrides are now simply not rendered;
  they are on disk and in the runner log. If a technical view is wanted
  again it is a separate surface, not a toggle on the product screen.
- **The Lip-synced toggle segment never falls back to the original.** When
  the render does not exist it is disabled with its own progress; when the
  job has no lip-sync stage at all the toggle is not rendered.
- **`full_quality: false` audio must show its note.** It means the only
  original audio on disk is the 16 kHz mono file made for diarization.
- **No CDN at runtime** — repo convention.

## Deviation from the plan

The plan's decisions table says Tailwind v4. This ships **Tailwind 3.4.17**
with the classic `tailwind.config.js` + PostCSS setup. v4 renames a number of
utilities (`shadow-sm` → `shadow-xs`, the default border colour, the ring
defaults), and this source was written without being able to run a build, so
every class here is a v3 class. Moving to v4 later is a config swap plus a
utility-rename pass, not a rewrite.
