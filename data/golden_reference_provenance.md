# Reference provenance — measured, not assumed

Generated during Phase 1 of the timing-alignment experiment. Every row below was
measured against the real media on disk, not read off YouTube's metadata.

## Audio-track provenance (`yt-dlp` `format_note`)

| Video | id | dur | Hindi track | Other dubs |
|---|---|---|---|---|
| #5 Introduction to Design Thinking (NPTEL-NOC IITM) | `6-NRiom8K9Y` | 494 s | **dubbed** (human) | bn gu kn ml mr te — all `dubbed` |
| Narendra Modi — Lex Fridman #460 | `ZPUtA3W-7_I` | 11 854 s | **dubbed** (human) | hi-Latn, ru |
| Pavel Durov — Lex Fridman #482 | `qjPH9njnaVU` | 16 450 s | **dubbed** (human) | fr ru uk |
| TED — SixthSense, Pranav Mistry | `YrtANPtnhyg` | 920 s | none (en only) | — |

All Hindi tracks are `dubbed`, i.e. human/professional. This is the key upgrade over the
previously-held corpus (`svwrTOm55y4`, `dIYDPtHWBNA`), whose Hindi was `dubbed-auto` — YouTube's
machine dubbing. Results against this corpus may be claimed against a human dub.

## Do the Hindi subtitles carry dub timing?

Different cue *boundaries* do not prove a subtitle tracks the dub — a re-wrapped translation on the
English timeline also has different boundaries. The test actually run: energy-VAD both audio tracks,
then count how many subtitle cue onsets land within 500 ms of a speech onset in each track.
A subtitle timed to the dub matches the Hindi audio more often and more tightly.

| Video | cues matched vs EN audio | vs HI audio | median offset | Verdict |
|---|---|---|---|---|
| `6-NRiom8K9Y` | **64** | 39 | 0.260 s / 0.239 s | **English timeline — NO dub timing** |
| `ZPUtA3W-7_I` | 88 | **105** | 0.133 s / **0.097 s** | Timed to the Hindi dub |
| `qjPH9njnaVU` | 115 | **124** | 0.084 s / **0.074 s** | Timed to the Hindi dub (slim margin) |

Consequences, per video:

- **`6-NRiom8K9Y`** — the Hindi subtitle is a translation laid on the English timeline, so it
  serves as reference **text only**. Hindi timing ground truth comes from whisperx forced-align on
  `6-NRiom8K9Y.hi.webm` using that text as the transcript, cross-checked against VAD.
- **`ZPUtA3W-7_I`, `qjPH9njnaVU`** — Hindi cue times are usable as a timing reference. Still refine
  to word level with whisperx forced-align; the cue-level offsets above (74–97 ms) are the floor on
  what the subtitle alone can resolve.

## Word-level timestamps

Inline `<00:00:00.378><c>word</c>` timestamps exist only in the auto-caption tracks:

| File | cues | inline words |
|---|---|---|
| `ZPUtA3W-7_I.en-orig.vtt` | 8 483 | 19 518 |
| `qjPH9njnaVU.en-orig.vtt` | 11 855 | 29 127 |
| `YrtANPtnhyg.en-orig.vtt` | 715 | 2 084 |
| every manual subtitle | — | **0** |

There is **no `hi-orig` track on any video in this corpus**, so there are no published Hindi word
timings. Hindi word-level ground truth is whisperx forced-align on the dub audio, for all four.

Note: these auto-caption files are rolling captions, which `parse_vtt_words` handles and
`parse_vtt` does not — the latter inflates word counts roughly 2x (see Phase 0.4).

## Duration lock

`6-NRiom8K9Y`: en 493.575 s vs hi 493.978 s — 0.4 s over 8 min. The human dub holds the source
duration almost exactly, which is the target behaviour the eight methods are being measured against.

## Gotchas found while acquiring

- **Three of the four `en*` subtitle tracks on `6-NRiom8K9Y` are not English.** Measured by
  Latin-script ratio over the cue text:

  | track | Latin ratio | actually |
  |---|---|---|
  | `en-6UJrWS5jR_I` | **1.000** | **English — the real one** |
  | `en-LUU0EuDKgKo` | 0.047 | not English (listed as "NPTEL Official") |
  | `en-PVQ7OO5RyjI` | 0.027 | Bengali |
  | `en` | 0.027 | Bengali (byte-identical to `en-PVQ7OO5RyjI`) |

  An earlier revision of this file named `en-LUU0EuDKgKo` as the English track, on the strength of
  its "NPTEL Official" label. That was wrong, and a label is not evidence: `backend/eval/reference.py`
  `resolve_tracks()` now picks by **measured** Latin-script ratio and writes the choice and the ratio
  into each clip's `meta.json`. Requesting plain `en` returns Bengali, so subtitle track keys must
  come from the probe JSON and then be verified against content.
- `format_note` carries the dub marker only on the m3u8 `233-*` variants; selecting the
  highest-bitrate format alone reports every track as `original`.
- `ZPUtA3W-7_I.hi.webm` first downloaded as 0 bytes and a naive existence check treated it as cached.
  Media smaller than 1 MiB is now always refetched.

## Golden set as built (18 clips)

`python -m backend.eval refbuild --clips data/golden/index.json` — all 18 built, 0 errors on the
15 clips that have a Hindi dub.

| Tier | Video | Clips | Hindi dub | GT floor (median) |
|---|---|---|---|---|
| A | `6-NRiom8K9Y` (NPTEL) | 6 | ✅ human | **not measurable** — no `*-orig` caption stream |
| A | `ZPUtA3W-7_I` (Modi) | 5 | ✅ human | 53 ms (43–61) |
| B | `qjPH9njnaVU` (Durov) | 4 | ✅ human | 53 ms (50–57) |
| C | `YrtANPtnhyg` (TED) | 3 | ❌ none | 77 ms (65–79) |

**Ground-truth noise floor: median 55 ms, range 43–79 ms**, over the 12 clips where it is computable.
It is the disagreement between whisperx forced alignment on the manual subtitle text (rank 2) and
YouTube's own inline caption word timestamps (rank 3) — the study's own measurement floor.
**No timing improvement smaller than this may be claimed.** Combined with the n=12 bootstrap floor of
80 ms for median |Δstart|, the reported timing resolution is **max(80, 55) = 80 ms**.

Two limits on what that number covers:

- **It is a source-side (English) floor only.** No `hi-orig` auto-caption exists on any of the four
  videos, so the Hindi dub side has no independent second opinion and its floor is recorded as
  `not_measured` in every `meta.json`. Dub-side word timings come from whisperx forced alignment on
  the human dub audio using the manual Hindi subtitle as transcript, with no cross-check.
- **The six NPTEL clips have no floor at all.** That video carries manual subtitles but no
  auto-captions, so there is no second word stream to disagree with. They are still the primary
  ranker — domain-matched IIT Madras lecture content — but a timing claim resting only on NPTEL
  clips inherits the 55 ms floor from the other videos by assumption, not by measurement, which
  the report records alongside the number.

## Which clips may carry which metric

A per-clip gate (`backend/eval/reference.py`, `word_vad_coverage`) checks each aligned word stream
against the **text-free** VAD for that same clip. A text-derived instrument is scored against one
that cannot be fooled by a bad transcript.

Measured word rate over all 18 clips, which is what the gate uses:

| population | words per second of VAD speech |
|---|---|
| healthy dub (Modi, Durov, dense NPTEL) | 3.12 – 5.15 |
| healthy source (English) | 2.51 – 4.05 |
| **under-subtitled NPTEL windows** | **0.96 – 1.45** |

The threshold is **2.0**, which sits in the empty gap between 1.45 and 2.51. An earlier attempt
gated on *span coverage* at 0.55 and had to be withdrawn: word spans exclude the inter-word
micro-gaps that VAD counts, so healthy English source tracks sit at 0.53–0.58 and were falsely
rejected. Coverage is still recorded, because it is what makes a flagged clip legible; it is not
what decides.

Result — four clips lose word-level eligibility, all from NPTEL's under-subtitled first 5.5 minutes:

| | clips | used for |
|---|---|---|
| Tier A, word-level eligible | **7** (2 NPTEL + 5 Modi) | Δstart/Δend, word-anchored placement, pause F1 from word gaps |
| Tier A, VAD-level only | 4 (NPTEL 5–325 s) | speech IoU, VAD pause F1, drift, stretch distribution |
| Tier B generalization | 4 (Durov) | separate table, never pooled into the ranking |
| Tier C source-side only | 3 (TED) | component tests; no Hindi dub exists |

**The headline n is 7, not 11, and not 18.** VAD-based metrics keep n=11 on tier A because VAD is
text-free and unaffected by the sparse subtitle. The two n's are reported separately rather than
pooled — pooling would let a metric borrow confidence from clips that cannot support it.

Two consequences worth stating plainly:

- **NPTEL contributes only 2 of the 7.** It is the domain-matched video (IIT Madras lecture content,
  which is what this pipeline targets), so the ranking leans more on the Modi interview than the
  corpus design intended. Per-video breakdowns accompany every pooled number.
- **At n=7 the bootstrap floors widen** against the n=12 figures quoted elsewhere in the plan. The
  floors are recomputed at the real n before a method is declared a winner, and the resolution
  remains `max(bootstrap floor, 55 ms ground-truth floor)`.

## Metric calibration: the no-op baseline, and what it disqualifies

The plan requires that every threshold be set from a floor run rather than asserted. The floor run
here is the **no-op**: score the untouched **English** source audio against the **human Hindi dub**,
i.e. the result of doing no dubbing at all. Measured on all 15 clips that have a dub:

| metric | no-op floor (median) | range | headroom to perfect |
|---|---|---|---|
| speech IoU | **0.872** | 0.580 – 0.970 | 0.128 |
| pause F1 @ 200 ms | **0.000** | 0.000 – 0.519 | 1.000 |

### Speech IoU is disqualified as a ranking metric

The plan proposed `IoU >= 0.60 (calibrate)` as the mix pass criterion. Calibration shows **0.60 is
far below the do-nothing baseline**: a dubbed run scoring 0.62 would be *worse than shipping the
original English audio*. The threshold as written would have passed a regression.

The cause is structural, not a bug. Both tracks talk almost continuously — source speech fraction
median 0.885 (0.648–0.999), dub 0.884 (0.811–0.975) — so any two tracks of this corpus overlap
heavily no matter how badly aligned they are. The metric's entire usable range is 0.872 → 1.000,
and its clip-to-clip spread at the floor (0.580–0.970, i.e. 0.390) is **three times** that headroom.
A difference between methods cannot be distinguished from a difference between clips.

Speech IoU therefore moves out of the headline table and is reported as a **guard** — it catches
a method that stops producing audio, and nothing finer. This supersedes the plan's
`IoU >= 0.60` row.

### Pause F1 @ 200 ms is the discriminator

Its no-op floor is **0.000** on 10 of 14 clips: the English pause structure essentially never lands
within 200 ms of the Hindi dub's. Every point above zero is real signal, which is what makes this
the metric that can actually adjudicate the user's constraint — *preserve where speech starts and
stops and where pauses occur*.

One caveat on where it can be measured. VAD finds **0–3 pauses** on the NPTEL English source against
5–14 in its dub, because the lecture audio is near-continuous; pause F1 computed from so few
reference events is unstable, and it is `n/a` outright on `6-NRiom8K9Y_00265to00325` (zero source
pauses). The two interview videos carry 7–14 pauses on both sides and are where this metric is
trustworthy. Per-video pause counts accompany the pooled figure.

## The Hindi reference has no word-level noise floor — and its cue-level floor is 185 ms

The plan's keystone (§1.2) was to publish the rank-2 vs rank-3 disagreement — whisperx forced-align
against the `*-orig` auto-caption word stream — as the ground truth's own noise floor, and to refuse
any timing claim smaller than it. That number is now measured, and **it does not apply to the Hindi
side.**

`refbuild` writes it per clip at `meta.agreement_median_ms`:

| Side | Clips measured | Median | Range |
|---|---|---|---|
| Source (English), word-level | 12 of 18 | **55 ms** | 43–79 ms |
| Reference dub (Hindi), word-level | **0 of 18** | not measured | — |

The Hindi column is empty for a corpus reason, not a bug: **no video in the corpus carries an
`hi-orig` auto-caption.** Only `6-NRiom8K9Y`, `ZPUtA3W-7_I`, `qjPH9njnaVU` and `YrtANPtnhyg` have
`en-orig`; the Hindi tracks are manual cue-level subtitles only. So every Hindi word stream on disk
comes from a single producer — `whisperx forced-align (hi)` — with nothing to cross-check it against.
`meta.agreement_median_ms.ref_dub` records this verbatim as
`"needs BOTH a caption word stream and whisperx words; caption=0, whisperx=<n>"`.

The 55 ms figure is therefore **not** "the ground truth's noise floor" for the method
ranking. It is the floor of the *English source* word stream, on a language with two independent
alignments available. The Hindi reference is the side every timing metric is actually scored against,
and it has no such check.

### The measurable substitute, and what it bounds

What *is* independently measurable in Hindi: human-authored cue onsets from the manual `.hi.vtt`
against the CTC onset whisperx assigned to that cue's first word. The two share their text but not
their timing, so the disagreement is a genuine timing cross-check — at **cue** resolution rather than
word resolution. Computed by `scripts/hi_cue_floor.py`, gated to clips where ≥90 % of cues matched a
whisperx token (below that the median rests on two or three cues and means nothing):

| Video | Clips used | Median | Range |
|---|---|---|---|
| `qjPH9njnaVU` (Durov) | 4 | 118 ms | 79–190 ms |
| `ZPUtA3W-7_I` (Modi) | 4 | 203 ms | 158–245 ms |
| `6-NRiom8K9Y` (NPTEL) | 1 of 6 | 338 ms | — |
| **Pooled** | **9** | **185 ms** | **79–338 ms** |

A cue onset is a coarser instrument than a word onset, so **185 ms is an upper bound on the Hindi
word-level floor, not an estimate of it.** The true value lies somewhere between the ~55 ms that CTC
achieves where it can be checked and the 185 ms measured here.

**What this disqualifies.** The plan's statistical header assumed an 80 ms noise floor for
median |Δstart| at n=12. On the Hindi side that assumption is unsupported: the reference's own
disagreement with itself is more than twice it. Until an independent Hindi word stream exists, the
honest rule is that **no timing improvement below 185 ms can be distinguished from reference noise**,
and differences smaller than that are reported as ties. Reported timing resolution is
`max(bootstrap floor, 185 ms)`.

**Why NPTEL is worst.** Five of its six clips fail the match-rate gate (0–25 % of cues matched a
whisperx token) and the one that passes sits at 338 ms — the highest of any clip. This is the same
sparse-subtitle defect already documented above, now visible in a second instrument. It is a further
reason NPTEL carries VAD-based metrics only and does not carry word-level ones.
