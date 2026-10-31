
/** Shared API contracts. These types describe data received from or sent to the UI server. */
export type JobState = 'queued' | 'running' | 'done' | 'failed' | 'cancelled' | 'unknown';
export type StageState = 'pending' | 'running' | 'done' | 'skipped' | 'failed';

export interface Stage {
  key: string;
  title: string | null;
  label: string;
  state: StageState;
  started_at: string | null;
  wall_s: number | null;

  deferred?: boolean;
}

export interface Phase {
  id: string;
  label: string;
  state: StageState;

  deferred?: boolean;

  contains?: { key: string; label: string }[];
  wall_s: number | null;
  substages: Stage[];
}

export interface Milestones {
  video_ready: boolean;
  diarization_done: boolean;

  awaiting_cast: boolean;
  speakers_ready: boolean;
  mix_done: boolean;
  lipsync_ready: boolean;
}

export interface CastStarted {
  job_id: string;
  status: string;
  gender_overrides: Record<string, string>;
  speaker_voice_overrides: Record<string, { voice: string; engine?: string }>;
}

export interface CastSelection {
  mode: 'autodetect' | 'select';
  gender?: string | null;
  voice?: string | null;
  engine?: string | null;
}

export interface Failure {
  title: string;
  detail: string;

  action: string | null;

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

  failure: Failure | null;
  n_done: number;
  n_total: number;

  n_total_is_partial?: boolean;
  pct: number;

  source: 'progress.json' | 'STATUS.json' | 'manifest.json' | 'files' | null;

  coarse?: boolean;

  queue_position?: number | null;
}

export type VideoTrack = 'enhanced' | 'lipsync' | 'original';

export interface MediaRef {
  url: string;
  download_url: string;
  label: string;
  name: string;
  bytes: number;

  model?: string | null;
  window?: [number, number] | null;

  full_quality?: boolean;
  note?: string | null;
}

export interface Media {
  poster_url: string | null;
  video: {
    original: MediaRef | null;
    lipsync: MediaRef | null;

    enhanced: MediaRef | null;
    dubbed_no_lipsync: MediaRef | null;
    lipsync_warning: string | null;

    enhanced_warning: string | null;
  };
  audio: {
    original: MediaRef | null;
    dubbed: MediaRef | null;
  };
}

export interface Speaker {
  id: string;
  label: string | null;
  speech_s: number;
  n_segments: number;

  gender: 'male' | 'female' | null;
  voice: string | null;

  voice_source: string | null;
  voice_score: number | null;

  voice_rationale: string | Record<string, unknown> | null;
  tone: string | null;
  reference_url: string | null;
  color: string;
}

export interface ToneOption {
  value: string;
  label: string;

  group: string;
}

export interface VoiceOption {
  voice: string;
  engine: string | null;
  gender: string | null;
  age: string | null;
  language: string | null;
  tier: string | null;

  display: string;

  recommended_for: string[];
}

export interface SpeakerTable {
  job_id: string;
  engine: string | null;
  engine_label: string;
  language: string | null;

  supports_tone: boolean;

  tone_options: ToneOption[];
  voice_options: VoiceOption[];
  speakers: Speaker[];
}

export interface SpecLine {
  key: string;
  label: string;

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

  title?: string | null;
}

export interface JobList {
  jobs: JobRow[];
  count: number;
}

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

  measured?: boolean;
}

export interface StyleOption {
  value: string;
  label: string;
}

export interface Options {
  languages: LanguageOption[];

  styles: StyleOption[];

  default_source_language: string | null;
  tts_engines: TtsEngineOption[];
  lipsync_models: LipsyncModelOption[];
  limits: { max_lipsync_s: number; note: string };
}

export interface RuntimeEstimate {
  low_s: number;
  high_s: number;

  text: string;

  basis: string;
  label?: string;
  note?: string;
}

export interface ProbeResult {
  video_id: string | null;
  title: string | null;
  uploader: string | null;
  duration_s: number | null;
  thumbnail_url: string | null;
  webpage_url: string;
  live_status: string | null;

  estimates: Record<string, RuntimeEstimate>;

  lipsync_estimates: Record<string, RuntimeEstimate>;

  max_lipsync_s: number;
  detected_source_language?: { code: string; language?: string; method?: string };
}

export interface UploadResult {
  upload_id: string;
  source: 'upload';
  title: string;
  duration_s: number;
  width: number | null;
  height: number | null;
  has_audio: boolean;
  size_bytes: number;
  estimates: Record<string, RuntimeEstimate>;
  lipsync_estimates: Record<string, RuntimeEstimate>;
  max_lipsync_s: number;
  detected_source_language?: { code: string; language?: string; method?: string };
}

export interface FaceSizeSpeaker {
  speaker: string;
  frames: number;
  avg_frac: number;
  avg_px: number;
}

export interface FaceSize {
  available: boolean;
  reason?: string;
  source?: 'asd';
  frame_height?: number;
  avg_frac?: number;
  avg_px?: number;
  per_speaker?: FaceSizeSpeaker[];

  suggests?: string;
  why?: string;
}

export interface LipsyncReadiness {
  job_id: string;
  can_start: boolean;
  blockers: string[];
  pending_edits: number;
  already_rendered: boolean;

  model_required: boolean;

  models: string[];

  model_rows?: LipsyncModelOption[];
  clip_s: number | null;
  max_lipsync_s: number;
  estimates: Record<string, string | null>;
  face: FaceSize;
}

export interface LipsyncStarted {
  job_id: string;
  status: string;
  model: string;
  dirty_stages: string[];
  warnings: string[];
}

export interface CreateJobRequest {

  url?: string;
  upload_id?: string;

  style?: string | null;

  source_language?: string | null;
  target_language: string;

  window?: { start_s: number; end_s: number } | null;
  preset?: string;
  tts?: string | null;
  lipsync_model?: string | null;
  num_speakers?: number | null;
  skip_stems?: boolean | null;
  /** Keep English technical terms in Latin script.
   *  true  → Bodhan MT receives `target_script=codemix`.
   *  false → the argument is omitted and the server picks the script. */
  code_mix?: boolean | null;
}

export interface CreateJobResult {
  job_id: string;
  display_name: string;
  state: JobState;
  window: { start_s: number; end_s: number } | null;
  target_language: string;
  preset: string;
  estimate: RuntimeEstimate;

  position?: number;
}

export interface PendingEdit {
  speaker_id: string;
  voice?: string;
  tone?: string | null;
  gender?: string;
}

export interface CostStage {
  id: string;
  label: string;
  gpu: boolean;
  est_s: number | null;
}

export interface CostPreview {
  stages: CostStage[];
  unaffected_labels: string[];
  total_est_s: number | null;
  gpu_required: boolean;
  warnings: string[];
}

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

export interface ApplyResult {
  job_id: string;
  status: string;
  dirty_stages: string[];
  cost: CostPreview | null;
  warnings: string[];

  edit_record: string | null;
  rerun: { launched: boolean; cmd?: string[]; pid?: number; error?: string };
}

export interface SpeakerLabels {
  job_id: string;
  labels: Record<string, string>;
}

export interface RevertResult {
  job_id: string;
  restored: string[];
  n_restored: number;
  warnings: string[];
}

export interface AgentField<T> {
  current: T | null;
  staged: T | null;
}

export interface AgentSpeaker {
  id: string;
  label: string | null;

  gender: AgentField<string>;
  speech_s: number | null;
  voice: AgentField<string>;
  tone: AgentField<string>;
}

export interface AgentTurn {
  at: string;
  message: string;

  source: string;
  ops: Array<Record<string, unknown>>;
}

export interface AgentParams {
  job_id: string;
  clip: {
    current: { start_s: number; end_s: number | null;
               duration_s: number | null; n_segments: number };
    staged: null;
  };

  language: { current: string | null; staged: null; editable: false; note: string };
  lipsync: {
    current: string | null;
    staged: string | null;
    window: { current: null; staged: { start_s: number; end_s: number } | null };
  };
  tts_engine: string | null;
  speakers: AgentSpeaker[];
  voice_options: VoiceOption[];
  tone_options: ToneOption[];
  dirty_stages: string[];
  cost: CostPreview | null;

  actions: AgentAction[];

  prompts: AgentPrompt[];
  turns: AgentTurn[];
  warnings: string[];
}

export interface AgentScriptRow {
  id: string;

  cue_no: number;
  speaker: string | null;
  start: number;
  end: number;
  src_text: string;
  text: string;
  tts_voice: string | null;
  tts_style: string | null;

  staged: boolean;

  staged_text: string | null;

  speed: number;

  staged_speed: number | null;
}

export interface AgentScript {
  job_id: string;
  segments: AgentScriptRow[];
  total: number;
}

export interface ScriptEdit {
  segment_id: string;
  text: string;
}

export interface AgentAction {

  op: string;
  title: string;

  takes: string;

  changes: string;

  replays: string;
  examples: string[];

  category: string;

  scopes: string[];

  quick: boolean;

  fill: string;

  needs: 'voice' | 'tone' | 'language' | 'speaker' | 'text' | null;

  options: string[] | null;
}

export interface AgentPrompt {
  text: string;
  does: string;
}

export interface EnhanceReadiness {
  can_start: boolean;
  blockers: string[];
  already: boolean;
  status: EnhanceStatus;

  renders: string[];
  render_id?: string | null;
  render_name?: string | null;
  render?: string | null;
  width?: number | null;
  height?: number | null;
  frames?: number | null;

  estimate_s: [number, number] | null;

  estimate_s_ungated: [number, number] | null;
}

export interface EnhanceStatus {
  state: 'idle' | 'running' | 'done' | 'failed';
  error?: string;
  video?: string;
  frames?: number;
  restored?: number;
  passed_through?: number;
  changed_frames?: number | null;
  gated?: boolean;
  strength?: number;
  wall_s?: number;
  s_per_frame?: number;
  expected_s?: number | null;
  started_at?: number;
}

export interface EditOpSummary {
  op: string;
  target: string;
  params: Record<string, unknown>;
  kind?: string;
}

export interface EditRecord {
  file: string;
  index: number;
  nlq: string | null;
  parsed_by: string | null;
  ops: EditOpSummary[];
  dirty_stages: string[];
  warnings: string[];
  speaker_voice_overrides: Record<string, unknown>;
  applied_count: number;
  mtime: string | null;
}

export interface VersionSnapshot {
  file: string;
  producer: string;
  mtime: string;
}

export interface EditHistory {
  job_id: string;
  edits: EditRecord[];
  snapshots: VersionSnapshot[];
  has_pending: boolean;
  is_running: boolean;
  outdated: boolean;
}

export interface EnhanceOptions {

  strength?: number;

  gated?: boolean;
  scale?: 'same' | '2x';
  only_center_face?: boolean;

  render?: string;
}

export interface ReviewVideoStatus {
  job_id?: string;

  state: 'absent' | 'running' | 'done' | 'failed';
  ready: boolean;

  url?: string | null;
  frames_total?: number;
  frames_done?: number;
  threshold?: number;
  error?: string;
  updated_at?: number;
}
