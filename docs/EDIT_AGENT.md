# The edit agent — what it accepts, and what it refuses

Reference for anyone driving or maintaining `backend/pipeline/edit_agent.py`: the nine
operations it can plan, the scopes they take, the stages each one invalidates,
and — the half that usually costs the most time — every way a request gets
rejected and what the message means.

Verified against the code on **2026-10-19**. Where a figure was measured
rather than read, it says so.

---

## 1. What it is

A finished dub is edited in plain language, not re-run from scratch. A
sentence becomes typed operations, which are applied to the run's merged
ASR+MT JSON, and only the stages the change invalidated are replayed. Stages
1–9a are never re-run — that JSON is already their output.

Two entry points, same planner:

| | CLI | UI |
|---|---|---|
| route | `python -m backend.edit_agent -j <job>` | `POST /api/jobs/{id}/agent` |
| default | dry run | stages into `<job>/edits/pending.json` |
| commits | `--apply` | `POST /api/jobs/{id}/edits/apply` |
| re-renders | `--run` | the apply launches it detached |

Parsing is LLM-first with a regex fallback. Without `OPENROUTER_API_KEY` the
regex parser still handles the common phrasings; only the three ops that
rewrite wording refuse outright (§4).

---

## 2. Supported operations

All nine are reachable from the agent. `dirties` is the earliest stage the
edit invalidates; everything from there on in the job's configured pipeline
is replayed.

| op | what it does | required params | optional | dirties |
|---|---|---|---|---|
| `set_text` | replace the translated line verbatim | `text` | — | `tts` |
| `replace_term` | swap one word/phrase, rest of the line byte-identical | `find` | `replace` (omit → translated for you) | `tts` |
| `retranslate` | re-translate from the source per an instruction | `instruction` | `target_language` | `tts` |
| `set_tone` | change delivery register | `tone` | `rewrite_text` (default `true`) | `tts` |
| `code_mix` | keep English technical terms in Latin script | — | `terms`, `mode` (`stage`\|`llm`), `termbase` | `codemix`, or `tts` when `mode=llm` |
| `set_gender` | recast the speaker from the opposite-gender voice pool | `gender` (`male`\|`female`) | — | `voices` |
| `set_speaker` | reattribute dialogue to another existing speaker | `speaker` | — | `voices` |
| `set_voice` | pin a specific TTS voice | `voice` | `engine` | `voices` |
| `set_speed` | re-fit speech into a shorter/longer slot | `factor` (0.5–2.0) | — | `tts` |
| `lipsync_window` | lip-sync only a time window | — | `model` (`wav2lip`\|`musetalk`\|`latentsync15`) | `lipsync` |
| `mute_cue` | silence scoped cues, release the slot as pause | — | — | `timing` |
| `set_target_language` | switch the dub's output language via Bodhan MT | `target_language` (ISO code of a supported Indic language) | `target_script` | `codemix` (whole dub) or `tts` (per-cue) |
| `merge_cues` | combine two consecutive same-speaker cues into one, LLM-rewritten | — (scope carries both ids) | — | `codemix` |

Stage order for replay:
`codemix → duration → voices → tts → timing → mix → lipsync`.

> `vc` (stage 9f, voice conversion) was removed from the chain on
> **2026-10-28** along with `backend/vc/` and `backend/voice_convert.py`. No
> run on disk ever set it — the registry had been empty since 2026-10-23 —
> so a legacy `request.json` carrying `"vc": null` still replays unchanged.

> **`set_speaker` was invisible until 2026-10-19.** It has been registered in
> `backend/pipeline/edits.py` all along, but the generated `op_catalogue()` advertised
> only eight of the nine, so the model was never told it could reattribute a
> line. The catalogue now lives in the system prompt and lists all nine, with
> a test asserting both directions (§7).

### Scope

Scope keys are `start_s`, `end_s`, `speaker`, `segments`, `all`. Smallest
scope that expresses the request wins.

- **A window selects every segment that OVERLAPS it**, not only those
  contained in it — "speed up 0:30 to 0:45" must catch a cue straddling 0:44
  or the edit audibly stops mid-sentence.
- **`start_s` with no `end_s` is a POINT**: the segment containing it, or the
  nearest one when it falls in a gap. That is what "change the gender at
  1:20" means.
- **A zero-width window (`start_s == end_s`) is treated as that point.** It
  is what a small model emits for "after 0:45" instead of omitting `end_s`,
  and under the strict overlap test it would otherwise match nothing.
- Scope terms **intersect**. `{speaker: SPK_00, start_s: 60, end_s: 90}` is
  "that speaker, in that window", not either.

---

## 3. Rejections — the request never becomes an op

These fire in the parser, before anything is applied. CLI: a non-zero exit
with the message. UI: **HTTP 422** carrying the same text.

| condition | message / behaviour |
|---|---|
| sentence the parser cannot place, no LLM | `could not turn that request into an edit` + the supported phrasings |
| op name not in the registry | `unknown op 'x'. Known: code_mix, lipsync_window, …` |
| empty plan | `no edit operations to apply` |
| message missing or > 2000 chars | `message is required` / `message is too long` |
| job has no transcript yet | **409** `This job has no transcript yet…` |
| job id not found | **404** |

**The model may also refuse by design.** The system prompt instructs it to
return `{"ops": []}` rather than guess when a request names a speaker or cue
that does not exist, or omits a value it cannot invent. Measured: "make SPK_99
formal" correctly returns an empty plan.

---

## 4. Rejections — the op is valid but cannot be applied

These fire during `apply_edits`, i.e. after the op parsed. Same surface: CLI
message, UI 422.

### Scope could not be resolved

| condition | message |
|---|---|
| cue id does not exist | `no such segment id(s): SEG_9999` |
| scope selects nothing | `<op>: scope '0:10.00–0:20.00' matched no segments` |
| unparseable timestamp | `unparseable timestamp '…'` |

A scope matching nothing is usually **correct, not a bug** — a window landing
in a silence gap genuinely has no cues. `lipsync_window` is exempt, because it
addresses the video timeline rather than cues.

### Missing or out-of-range parameters

| op | refusal |
|---|---|
| `set_text` | `set_text needs a non-empty 'text'` |
| `set_tone` | `set_tone needs a 'tone'` |
| `set_speed` | `set_speed needs a positive 'factor' (1.2 = 20% faster)` |
| `set_speed` | `set_speed factor 3.0 is outside 0.5–2.0; beyond that the stretch stops being intelligible` |
| `set_voice` | `set_voice needs a 'voice'` |
| `set_gender` | `gender must be 'male' or 'female', got '…'` |
| `set_speaker` | `unknown speaker 'SPK_99'; this job has SPK_00, SPK_01. A new id would have no reference clip and no assigned voice.` |
| `lipsync_window` | `lipsync_window needs a time range (start and end)` |
| `lipsync_window` | `lipsync window 0.30s is too short to render` |
| `set_gender` | `cannot honour gender='female': <why>` — the voice pool was exhausted |
| `set_target_language` | `set_target_language needs a 'target_language' ISO code, e.g. hi, mr, ta` |
| `set_target_language` | `unknown or non-Indic target language 'xx'. Supported: as, bn, …` |
| `set_target_language` | `set_target_language cannot scope to a speaker alone — add a time range …` |
| `set_target_language` | `Bodhan MT refused the translation: <why>` — the API call failed; `BODHAN_MT_KEY` missing surfaces here |
| `merge_cues` | `merge_cues needs exactly two cue ids in its scope` — the user named one or more-than-two |
| `merge_cues` | `merge_cues: unknown cue(s): SEG_XXXX` — a named id is not in the live cue list |
| `merge_cues` | `merge_cues: SEG_a and SEG_b are not consecutive` — they are not index-adjacent in the segments list |
| `merge_cues` | `merge_cues: speakers differ (SPK_00 vs SPK_01)` — explicit same-speaker requirement |
| `merge_cues` | `merge_cues: N.NNs gap between SEG_a and SEG_b (max 2.0s)` — pause between the cues is too long |
| `merge_cues` | `merge_cues needs the LLM (OPENROUTER_API_KEY unset)` — the handler calls the LLM, no fallback |

### No LLM configured

Three ops rewrite wording and refuse by name without `OPENROUTER_API_KEY`:

- `retranslate` → *"needs the LLM … Use set_text with the exact wording instead."*
- `code_mix` with `mode=llm` → *"needs the LLM …, and no transliteration model is configured for this job."*
- `set_tone` with `rewrite_text=true` → the tone hint is still staged; only the rewrite is skipped, with a warning.

Everything else works with no key at all.

### The UI's structured endpoint is narrower than the agent

`POST /api/jobs/{id}/edits` (the form controls) stages **only `set_voice` and
`set_tone`**, because both are deterministic — which is what lets it promise
that the quoted cost is the applied cost. The agent endpoint accepts all nine
and carries a per-op warning instead. Applying a set staged by the form with
any other op in it is refused: `refusing to apply 'set_speed'`.

`retranslate`, `set_text` and `code_mix` are flagged **non-deterministic**: an
LLM rewrite is not repeatable, so the text shown in the preview is not
guaranteed to be the text written at apply, and different wording is a
different length — a different timing fit.

---

## 5. Accepted, but silently limited by the engine

These do **not** error. They apply, warn, and then do less than the wording
implies — the failure mode worth knowing about, because every artifact looks
correct.

| limitation | effect |
|---|---|
| `set_tone` on a non-Bodhan engine | *"tts engine 'indicf5' ignores the style hint … the tone change will come from the rewritten wording alone"* — only the Bodhan adapter forwards `instructions.style` |
| per-segment rate | only engines with `supports_rate = True` honour it; since Edge was removed that is Bodhan alone |
| IndicF5 (removed 2026-10-28) | read neither style nor rate — on a job recorded against it a tone edit changes the wording and nothing about the delivery |
| `set_gender` | a hard filter on the voice catalog, not a weight; if the same-gender pool is exhausted it crosses gender and says so |

## 6. Accepted, but the stage is not in this job

An op may dirty a stage the run never configured — `set_speed` on a job with
no timing producer, `lipsync_window` on a job that never lip-synced. The
rerun set is the dirty stage onward **intersected with the stages the job
actually configured**, so the plan never promises work that will not happen
and the approved cost matches the executed run. A stage the *edit itself*
introduces (code-mix on a job that never code-mixed, lip-sync on one that
never lip-synced) is added back.

---

## 7. Measured model behaviour

`google/gemma-3-12b-it`, system prompt of 2026-10-19, against a real 13-cue
job. 8 representative requests, **7 resolved correctly**:

| request | result |
|---|---|
| make SPK_01 more formal | `set_tone` scope `{speaker: SPK_01}` — no spurious time range |
| speed up 0:30 to 0:45 by 1.2x | `set_speed` `{start_s: 30, end_s: 45}`, factor 1.2 |
| SEG_0003 should say '…' | `set_text` scope `{segments: [SEG_0003]}` |
| keep gradient and weight in English | `code_mix` `{all: true}`, both terms extracted |
| the line at 0:35 is actually SPK_00 | `set_speaker` |
| lip-sync 0:20–0:35 with musetalk | `lipsync_window` + model |
| make SPK_99 formal | `[]` — correctly refused |
| **make it a little faster** | `set_speed` with **no `factor`** — the prompt asks for `{"ops": []}` here |

The eighth fails safely: `backend/pipeline/edits.py` answers *"set_speed needs a
positive `factor`"* rather than inventing a speed.

**Two traps worth carrying forward.**

`OPENROUTER_JSON_MODE` defaults to **off**, and that is a measurement.
`response_format={"type":"json_object"}` — the flag whose purpose is to
guarantee parseable JSON — is what was corrupting it on this model and
provider: `{"ops":[{}]}` and keys prefixed with a full stop (`".op"`). Both
are *valid JSON*, so nothing downstream can tell; the symptom was
`unknown op ''`. With it off the model returns correct JSON in ```json
fences, which `extract_json` already strips.

The catalogue moving into the system prompt **did not reduce tokens**: input
per parse went 1653 → 2156, up 30%, because the prompt is longer than the
catalogue it replaced. The win is correctness.

Because the prompt is hand-written while `OPS` is generated, the two must be
kept in sync by hand — there's no automated check if a registered op goes
missing from the prompt, or if the prompt advertises one that does not exist.

---

## 8. Not supported

- **Word-level edits are `replace_term`, not `retranslate`.** `retranslate`
  regenerates the whole line through an LLM, so it drifts on wording the
  request never mentioned — measured on one cue, a "translate the word
  'conditional'" instruction also rendered `text generators` two
  different ways in adjacent lines.
- **"from X to the end"** has no scope key. `after 0:45` resolves to the
  point at 0:45 — a single cue — not everything following it.
- **No op creates or deletes a segment**, or changes timings directly;
  `set_speed` re-fits speech inside the slots that already exist.
- **No op touches stages 1–9a.** Re-running diarization, ASR or MT means a
  new run.
- **A new speaker id** cannot be introduced — it would have no reference clip
  and no assigned voice.
- **Nothing gates lip-sync on who is speaking.** `lipsync_window` restricts
  *when*, never *whose* mouth.

## 9. Reversing an edit

Every overwritten key is kept in the segment's `edit_backup`, and each
request is journalled to `extra.edits.history` and `<job>/edits/edit_NNN.json`
— so `--revert` restores, and a finished run can be audited without re-running
anything. `pending.json` is working state and is deleted on commit.

Traces of every LLM call (prompt, raw reply, tokens, cost, latency) go to
Langfuse when configured; see `backend/common/tracing.py`.
