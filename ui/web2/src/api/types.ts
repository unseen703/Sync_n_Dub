/**
 * The reconciliation document between `ui/*.py` and this app.
 *
 * Every shape here was transcribed from the Python that produces it, not
 * guessed: `ui/progress.py::read_progress`, `ui/media.py::resolve`,
 * `ui/speakers.py::build_table`, `ui/spec.py::build`, and the route bodies
 * in `ui/server.py`. When one of those changes, change this file in the
 * same commit — a silently-wrong type here shows up as an empty panel, not
 * as an error.
 *
 * The load-bearing piece is `Progress.milestones`. Every conditional render
 * in this app keys off one of those booleans. Nothing in `src/` is allowed
 * to decide "is the lip-sync ready" by inspecting a filename; `ui/media.py`
 * owns that question and has five ranked patterns for it.
 */

// ---------------------------------------------------------------------------
// progress — GET /api/jobs/{id}/progress
// ---------------------------------------------------------------------------

export type JobState = 'queued' | 'running' | 'done' | 'failed' | 'cancelled' | 'unknown';
export type StageState = 'pending' | 'running' | 'done' | 'skipped' | 'failed';

export interface Stage {
  key: string;
  title: string | null;
  label: string;
  state: StageState;
  started_at: string | null;
  wall_s: number | null;
}

/** One pill in the ribbon. `ui/progress.py::PHASE_GROUPS` sets the order. */
export interface Phase {
  id: string;
  label: string;
  state: StageState;
  wall_s: number | null;
  substages: Stage[];
}

/**
 * What the UI is allowed to show yet. Computed server-side from stage states
 * OR'd with existence checks on disk, so it is correct for all three
 * on-disk job shapes (progress.json / STATUS.json / manifest.json).
 */
export interface Milestones {
  video_ready: boolean;
  diarization_done: boolean;
  speakers_ready: boolean;
  mix_done: boolean;
  lipsync_ready: boolean;
}

/**
 * `ui/errors.py::explain` — a user-facing reading of the failure.
 *
 * This is what the UI renders. `Progress.error` is a traceback line and is
 * NOT shown: the raw runner log was removed from this app entirely, because
 * it is user-facing. `raw` is kept to one line and lives behind a
 * "Technical details" disclosure.
 */
export interface Failure {
  title: string;
  detail: string;
  /** What to do next. Null when there is nothing honest to suggest. */
  action: string | null;
  /** False when this failure matched none of the known patterns. */
  recognised: boolean;
  raw: string | null;
}

export interface Progress {
  version: number;
  job_id: string;
  kind?: string;
  pid?: number | null;
  state: JobState;
  started_at: string | null;
  updated_at: string | null;
  heartbeat?: string | null;
  current: string | null;
  error: string | null;
  log?: string | null;
  stages: Stage[];
  phases: Phase[];
  milestones: Milestones;
  /** Render THIS, not `error`. Null when nothing is wrong. */
  failure: Failure | null;
  n_done: number;
  n_total: number;
  /** STATUS.json jobs have no planned stage list — the total grows. */
  n_total_is_partial?: boolean;
  pct: number;
  /** Which of the three on-disk shapes answered. */
  source: 'progress.json' | 'STATUS.json' | 'manifest.json' | 'files' | null;
  /** manifest.json and flat-file jobs cannot say which stage is running. */
  coarse?: boolean;
  /** Set by the client, not the server: queue position from POST /api/jobs. */
  queue_position?: number | null;
}

// ---------------------------------------------------------------------------
// media — GET /api/jobs/{id}/media
// ---------------------------------------------------------------------------

export interface MediaRef {
  url: string;
  download_url: string;
  label: string;
  name: string;
  bytes: number;
  /** video.lipsync only */
  model?: string | null;
  window?: [number, number] | null;
  /** audio.* only — false means 16 kHz mono, and `note` says why. */
  full_quality?: boolean;
  note?: string | null;
}

export interface Media {
  poster_url: string | null;
  video: {
    original: MediaRef | null;
    lipsync: MediaRef | null;
    dubbed_no_lipsync: MediaRef | null;
  };
  audio: {
    original: MediaRef | null;
    dubbed: MediaRef | null;
  };
}

// ---------------------------------------------------------------------------
// speakers — GET /api/jobs/{id}/speakers
// ---------------------------------------------------------------------------

export interface Speaker {
  id: string;
  label: string | null;
  speech_s: number;
  n_segments: number;
  /** ECAPA's prediction from the reference clip. Null when 9d never ran. */
  gender: 'male' | 'female' | null;
  voice: string | null;
  /** 'assigned' | 'edit' | 'derived' — how confidently we know the voice. */
  voice_source: string | null;
  voice_score: number | null;
  /**
   * NOT a string. `ui/speakers.py` passes `rationale` straight through from
   * `voices/*.voices.json`, and the voice-assignment stage writes a dict:
   * `{"gender_match": 1.0, "embed_cos": 0.158}`. Rendering it as a React
   * child throws. `lib/format.ts::fmtRationale` turns it into a sentence.
   */
  voice_rationale: string | Record<string, unknown> | null;
  tone: string | null;
  reference_url: string | null;
  color: string;
}

/**
 * One entry of `tone_options`, from `ui/speakers.py::tone_vocabulary`,
 * which reads `models.tts.bodhan_adapter.INDIC_SPEAK_STYLES` verbatim.
 *
 * `value` is the WIRE value and is significant — "AIR style news",
 * "single person narration audiobook". Case and spacing matter and an
 * unknown value is a 422 from the API, never a silent fallback. Send
 * `value` untouched; show `label`; group by `group`.
 */
export interface ToneOption {
  value: string;
  label: string;
  /** "Delivery" (a format) or "Emotion" (one of the six Ekman moods). */
  group: string;
}

export interface VoiceOption {
  voice: string;
  engine: string | null;
  gender: string | null;
  age: string | null;
  language: string | null;
  tier: string | null;
  /** Always render this. `hi-IN-SwaraNeural` is an id; `Swara` is a name. */
  display: string;
  /**
   * Speaker ids this voice suits, matched on the predicted gender by
   * `ui/speakers.py::build_table`. A HINT, not a filter: the picker opens
   * on these and keeps every other voice one click away, because a
   * prediction is a prediction. Empty for every voice when no speaker has
   * a recorded gender.
   */
  recommended_for: string[];
}

export interface SpeakerTable {
  job_id: string;
  engine: string | null;
  engine_label: string;
  language: string | null;
  /** Bodhan / indic-speak only. When false the tone column is ABSENT. */
  supports_tone: boolean;
  /** Empty when `supports_tone` is false — the control is then ABSENT. */
  tone_options: ToneOption[];
  voice_options: VoiceOption[];
  speakers: Speaker[];
}

// ---------------------------------------------------------------------------
// spec — GET /api/jobs/{id}/spec
// ---------------------------------------------------------------------------

export interface SpecLine {
  key: string;
  label: string;
  /** Already English, already formatted. Render verbatim. */
  value: string;
  hint: string | null;
  tone: 'default' | 'warn';
}

export interface Spec {
  title: string | null;
  thumbnail_url: string | null;
  lines: SpecLine[];
  raw_config: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// job list — GET /api/jobs
// ---------------------------------------------------------------------------

export interface JobRow {
  job_id: string;
  display_name?: string;
  timestamp: string | null;
  ts_raw?: string;
  slug: string;
  config_slug: string | null;
  state: string;
  updated: string | null;
  lipsync_model: string | null;
  language: string | null;
  has_final: boolean;
  /**
   * `display_name` is always present: the server writes it at creation from
   * the URL ("YouTube L3oOldViIgY") and upgrades it to the real title once
   * the probe returns. `title` is only set when that real title exists, so
   * the two together say whether the name is final or a placeholder.
   */
  title?: string | null;
}

export interface JobList {
  jobs: JobRow[];
  count: number;
}

// ---------------------------------------------------------------------------
// options — GET /api/options?tts=
// ---------------------------------------------------------------------------

export interface LanguageOption {
  code: string;
  english: string;
  native: string;
  script: string;
  font: string;
}

export interface TtsEngineOption {
  id: string;
  label: string;
  supports_tone: boolean;
  note: string;
}

export interface LipsyncModelOption {
  id: string;
  label: string;
  note?: string;
  speed?: string;
}

export interface Options {
  languages: LanguageOption[];
  tts_engines: TtsEngineOption[];
  lipsync_models: LipsyncModelOption[];
  limits: { max_lipsync_s: number; note: string };
}


// ---------------------------------------------------------------------------
// The write side. These were reconciled against the real route handlers in
// `ui/routes_create.py` and `ui/routes_edits.py`, not against the design
// sketch — the two had drifted, and the Python is what actually answers.
// ---------------------------------------------------------------------------

/** One preset's measured runtime estimate, from `ui/presets.py`. */
export interface RuntimeEstimate {
  low_s: number;
  high_s: number;
  /** Ready to print: "about 4 minutes to 11 minutes". */
  text: string;
  /** Why those numbers — the measured rates this host was benchmarked at. */
  basis: string;
  label?: string;
  note?: string;
}

/**
 * `POST /api/youtube/probe`. The form needs all of this before the job
 * exists: the title for the card, the duration to bound the range picker,
 * and a per-preset estimate so nobody starts a four-hour render blind.
 */
export interface ProbeResult {
  video_id: string | null;
  title: string | null;
  uploader: string | null;
  duration_s: number | null;
  thumbnail_url: string | null;
  webpage_url: string;
  live_status: string | null;
  estimates: Record<string, RuntimeEstimate>;
  /** The lip-sync cap, in seconds. The form enforces it before submitting. */
  max_lipsync_s: number;
}

export interface CreateJobRequest {
  url: string;
  target_language: string;
  /** Omitted or null means the whole video. */
  window?: { start_s: number; end_s: number } | null;
  preset?: string;
  tts?: string | null;
  lipsync_model?: string | null;
  num_speakers?: number | null;
  skip_stems?: boolean | null;
}

/** `POST /api/jobs` — 201 when it started, 202 when it was queued. */
export interface CreateJobResult {
  job_id: string;
  display_name: string;
  state: JobState;
  window: { start_s: number; end_s: number } | null;
  target_language: string;
  preset: string;
  estimate: RuntimeEstimate;
  /** Present only when `state === 'queued'`. */
  position?: number;
}

/** One staged, not-yet-applied change to a speaker. */
export interface PendingEdit {
  speaker_id: string;
  voice?: string;
  tone?: string | null;
}

/** One line of the cost dialog. `gpu` marks the minutes-not-seconds one. */
export interface CostStage {
  id: string;
  label: string;
  gpu: boolean;
  est_s: number | null;
}

/**
 * What re-running will cost. `unaffected_labels` matters as much as
 * `stages`: it is what tells the user their hour of diarization, ASR and
 * translation is not at risk, which is the thing they actually fear.
 */
export interface CostPreview {
  stages: CostStage[];
  unaffected_labels: string[];
  total_est_s: number | null;
  gpu_required: boolean;
  warnings: string[];
}

/**
 * `POST /api/jobs/{id}/edits` (stage) and `GET` (read back). The server
 * persists this to `<job>/edits/pending.json`, because apply runs in a
 * different process and cannot inherit anything held in memory.
 */
export interface EditPlan {
  version?: number;
  job_id: string;
  updated: string | null;
  edits: PendingEdit[];
  ops?: Array<Record<string, unknown>>;
  dirty_stages: string[];
  speaker_voice_overrides?: Record<string, { engine?: string; voice?: string }>;
  segments?: string[];
  warnings: string[];
  cost: CostPreview | null;
}

/** `POST /api/jobs/{id}/edits/apply`. */
export interface ApplyResult {
  job_id: string;
  status: string;
  dirty_stages: string[];
  cost: CostPreview | null;
  warnings: string[];
  /** The committed record the re-render reads; `pending.json` is gone by now. */
  edit_record: string | null;
  rerun: { launched: boolean; cmd?: string[]; pid?: number; error?: string };
}

/**
 * `POST /api/jobs/{job_id}/speakers/{speaker_id}/label`, body
 * `{"label": "Prof. Mitra"}` (an empty string clears it).
 *
 * Display only. `ui/speakers.py` keeps these in a job-local
 * `speaker_labels.json` precisely so a rename dirties NO pipeline stage —
 * it must never appear as a pending edit or trigger a re-render.
 */
export interface SpeakerLabels {
  job_id: string;
  labels: Record<string, string>;
}

/** `POST /api/jobs/{id}/revert`. */
export interface RevertResult {
  job_id: string;
  restored: string[];
  n_restored: number;
  warnings: string[];
}
