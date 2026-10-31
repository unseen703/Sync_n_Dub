/** In-browser API substitute for development. Inputs mirror the live client; outputs are deterministic sample data. */
import type {
  ApplyResult,
  CreateJobRequest,
  CreateJobResult,
  EditPlan,
  JobList,
  Media,
  Options,
  PendingEdit,
  ProbeResult,
  UploadResult,
  Progress,
  Spec,
  SpeakerLabels,
  SpeakerTable,
} from './types';

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

const YT_ID = /(?:v=|youtu\.be\/|\/shorts\/|\/embed\/)([A-Za-z0-9_-]{11})/;

export async function probeYouTube(url: string): Promise<ProbeResult> {
  await sleep(420);
  const m = YT_ID.exec(url);
  if (!m) {
    throw Object.assign(new Error('That does not look like a YouTube link.'), { status: 400 });
  }
  return {
    video_id: m[1],
    title: 'Thermal Physics — Lecture 4: Entropy and the Second Law',
    uploader: 'NPTEL',
    duration_s: 725,
    thumbnail_url: null,
    live_status: null,
    webpage_url: `https://www.youtube.com/watch?v=${m[1]}`,

    estimates: {
      dub: { low_s: 90, high_s: 150, text: 'about 2 minutes',
             basis: 'Mocked.', label: 'Dub the audio', note: 'Audio only' },
    },

    lipsync_estimates: {
      wav2lip: { low_s: 2900, high_s: 5800, text: 'about 48 to 97 minutes',
                 basis: 'Mocked.', label: 'wav2lip', note: '3-5x' },
      musetalk: { low_s: 5800, high_s: 15200, text: 'about 1.6 to 4.2 hours',
                  basis: 'Mocked.', label: 'musetalk', note: '8-21x' },
    },
    max_lipsync_s: 1200,
  };
}

const invented: CreateJobResult[] = [];

export async function uploadVideo(file: File): Promise<UploadResult> {
  await sleep(300);
  return {
    upload_id: 'f'.repeat(32),
    source: 'upload',
    title: file.name.replace(/\.[^.]+$/, ''),
    duration_s: 92,
    width: 1920, height: 1080, has_audio: true,
    size_bytes: file.size,
    estimates: {
      dub: { low_s: 300, high_s: 700, text: 'about 5 minutes to 12 minutes',
             basis: 'Mocked.', label: 'Dub the audio', note: 'Audio only' },
    },
    lipsync_estimates: {
      wav2lip: { low_s: 300, high_s: 500, text: 'about 5 to 8 minutes',
                 basis: 'Mocked.', label: 'wav2lip', note: '3-5x' },
    },
    max_lipsync_s: 1200,
  };
}

export async function createJob(p: CreateJobRequest): Promise<CreateJobResult> {
  await sleep(300);
  const stamp = new Date().toISOString().replace(/[-:]/g, '').slice(0, 15);
  const span = p.window ? p.window.end_s - p.window.start_s : 300;
  const low = Math.round(span * 4);
  const high = Math.round(span * 8);
  const r: CreateJobResult = {
    job_id: `${stamp}__mock${invented.length + 1}__${p.target_language}`,
    display_name: 'Mock job',
    state: invented.length ? 'queued' : 'running',
    window: p.window ?? null,
    target_language: p.target_language,
    preset: p.preset ?? 'standard',
    estimate: {
      low_s: low,
      high_s: high,
      text: `about ${Math.ceil(low / 60)} to ${Math.ceil(high / 60)} minutes`,
      basis: 'Mocked — the real basis comes from server/presets.py.',
    },
    ...(invented.length ? { position: invented.length } : {}),
  };
  invented.push(r);
  return r;
}

export async function cancelJob(id: string) {
  await sleep(200);
  return { job_id: id, state: 'cancelled' };
}

const staged = new Map<string, PendingEdit[]>();

function plan(id: string): EditPlan {
  const pending = staged.get(id) ?? [];
  const n = pending.length;
  const stages = n
    ? [
        { id: 'voices', label: 'Casting voices', gpu: false, est_s: 2 },
        { id: 'tts', label: 'Speaking the translation', gpu: false, est_s: 95 },
        { id: 'mix', label: 'Mixing the new soundtrack', gpu: false, est_s: 20 },
        { id: 'lipsync', label: 'Lip-syncing', gpu: true, est_s: 1260 },
      ]
    : [];
  return {
    version: 1,
    job_id: id,
    updated: n ? new Date().toISOString() : null,
    edits: pending,
    dirty_stages: stages.map((x) => x.id),
    warnings: n
      ? ['Re-synthesis covers every speaker — stage 9e has no per-speaker incremental mode.']
      : [],
    cost: {
      stages,
      unaffected_labels: [
        'Fetching the video',
        'Working out who speaks when',
        'Transcribing',
        'Translating',
      ],
      total_est_s: n ? 1377 : null,
      gpu_required: stages.some((x) => x.gpu),
      warnings: [],
    },
  };
}

export async function getEdits(id: string): Promise<EditPlan> {
  await sleep(80);
  return plan(id);
}

export async function stageEdits(id: string, pending: PendingEdit[]): Promise<EditPlan> {
  await sleep(120);
  staged.set(id, pending);
  return plan(id);
}

export async function discardEdits(id: string): Promise<EditPlan> {
  await sleep(80);
  staged.delete(id);
  return plan(id);
}

export async function applyEdits(id: string): Promise<ApplyResult> {
  await sleep(500);
  staged.delete(id);
  return {
    job_id: id,
    status: 'applied',
    dirty_stages: ['voices', 'tts', 'mix', 'lipsync'],
    cost: null,
    warnings: [],
    edit_record: `data/jobs/${id}/edits/edit_000.json`,
    rerun: { launched: true, pid: 12345 },
  };
}

const MOCK_JOBS: JobList = {
  jobs: [
    {
      job_id: '20261016T142201__svwrTOm55y4__hi_musetalk',
      timestamp: '2026-10-16 14:22:01',
      slug: 'svwrTOm55y4',
      config_slug: 'hi_musetalk',
      state: 'running',
      updated: new Date().toISOString(),
      lipsync_model: 'musetalk',
      language: 'hi',
      has_final: false,
      title: 'Thermal Physics — Lecture 4: Entropy and the Second Law',
    },
    {
      job_id: '20261015T090300__qXz19Lm2kP0__ta_wav2lip',
      timestamp: '2026-10-15 09:03:00',
      slug: 'qXz19Lm2kP0',
      config_slug: 'ta_wav2lip',
      state: 'failed',
      updated: '2026-10-15T09:11:40Z',
      lipsync_model: 'wav2lip',
      language: 'ta',
      has_final: false,
      title: 'Quantum Imaging — Seminar',
    },
    {
      job_id: '20261014T201155__NAGtalk__hi_wav2lip',
      timestamp: '2026-10-14 20:11:55',
      slug: 'NAGtalk',
      config_slug: 'hi_wav2lip',
      state: 'done',
      updated: '2026-10-14T21:40:02Z',
      lipsync_model: 'wav2lip',
      language: 'hi',
      has_final: true,
      title: null,
    },
  ],
  count: 3,
};

function mkStage(
  key: string,
  label: string,
  state: Progress['stages'][number]['state'],
): Progress['stages'][number] {
  return { key, title: label, label, state, started_at: null, wall_s: state === 'done' ? 12.4 : null };
}

const MOCK_PROGRESS: Record<string, Progress> = {};
for (const j of MOCK_JOBS.jobs) {
  const running = j.state === 'running';
  const failed = j.state === 'failed';
  const stages = [
    mkStage('1', 'Fetching the video', 'done'),
    mkStage('2', 'Extracting the audio', 'done'),
    mkStage('4', 'Working out who speaks when', 'done'),
    mkStage('7', 'Transcribing', failed ? 'failed' : 'done'),
    mkStage('9', 'Translating', failed ? 'pending' : 'done'),
    mkStage('9d', 'Casting voices', failed ? 'pending' : 'done'),
    mkStage('9e', 'Speaking the translation', running ? 'running' : failed ? 'pending' : 'done'),
    mkStage('9g', 'Mixing the new soundtrack', running || failed ? 'pending' : 'done'),
    mkStage('9h', 'Lip-syncing', running || failed ? 'pending' : 'done'),
  ];
  MOCK_PROGRESS[j.job_id] = {
    version: 1,
    job_id: j.job_id,
    state: j.state as Progress['state'],
    started_at: '2026-10-16T14:22:01Z',
    updated_at: new Date().toISOString(),
    current: running ? '9e' : null,
    error: failed ? 'ASR failed: no speech detected in the first 90 s of the clip.' : null,

    failure: failed
      ? {
          title: 'The job failed',
          detail: 'ASR failed: no speech detected in the first 90 s of the clip.',
          action: null,
          recognised: false,
          raw: 'ASR failed: no speech detected in the first 90 s of the clip.',
        }
      : null,
    stages,
    phases: [],
    milestones: {
      video_ready: true,
      diarization_done: true,
      awaiting_cast: false,
      speakers_ready: !failed,
      mix_done: j.state === 'done',
      lipsync_ready: j.state === 'done',
    },
    n_done: stages.filter((s) => s.state === 'done').length,
    n_total: stages.length,
    pct: j.state === 'done' ? 100 : failed ? 44 : 62,
    source: 'progress.json',
  };

  const groups: Array<[string, string, string[]]> = [
    ['prepare', 'Prepare', ['1', '2']],
    ['translate', 'Translate', ['4', '7', '9']],
    ['dub', 'Dub audio', ['9d', '9e', '9g']],
    ['lipsync', 'Lip-sync', ['6', '9h', '9i']],
  ];
  const phases = groups.map(([id, label, keys]) => {
    const subs = stages.filter((s) => keys.includes(s.key));
    const st = subs.some((s) => s.state === 'running')
      ? 'running'
      : subs.some((s) => s.state === 'failed')
        ? 'failed'
        : subs.every((s) => s.state === 'done')
          ? 'done'
          : subs.some((s) => s.state === 'done')
            ? 'running'
            : 'pending';
    return {
      id,
      label,
      state: st as Progress['stages'][number]['state'],
      deferred: subs.length > 0 && subs.every((s) => s.state === 'pending' && s.deferred),
      wall_s: null,
      substages: subs,
    };
  });

  const complete = phases.every((ph) => ph.state === 'done' || ph.state === 'skipped')
    && !phases.some((ph) => ph.substages.some((s) => s.state === 'pending' && s.deferred));
  phases.push({
    id: 'finished',
    label: 'Finished',
    state: (phases.some((ph) => ph.state === 'failed')
      ? 'failed'
      : complete ? 'done' : 'pending') as Progress['stages'][number]['state'],
    deferred: false,
    wall_s: null,
    substages: [],
  });
  MOCK_PROGRESS[j.job_id].phases = phases;
}

export async function listJobs(): Promise<JobList> {
  await sleep(60);
  return MOCK_JOBS;
}

export async function getProgress(id: string): Promise<Progress> {
  await sleep(40);
  const p = MOCK_PROGRESS[id];
  if (!p) throw Object.assign(new Error(`No progress for ${id}`), { status: 404 });
  return p;
}

export async function getMedia(id: string): Promise<Media> {
  await sleep(50);
  const done = MOCK_PROGRESS[id]?.state === 'done';
  const mk = (label: string, name: string, bytes: number, extra = {}) => ({
    url: '',
    download_url: '',
    label,
    name,
    bytes,
    ...extra,
  });
  return {
    poster_url: null,
    video: {
      original: mk('Original', 'clip.mp4', 41_200_000),
      lipsync: done ? mk('Lip-synced', 'clip.musetalk.lipsync.mp4', 46_800_000) : null,
      // Stage 9i is never part of a submission, so the mock's "finished"
      // job does not have one either — `done` means the pipeline ran, not
      // that someone chose to enhance afterwards. Flip this to `mk(...)` to
      // exercise the third track.
      enhanced: null,
      dubbed_no_lipsync: done ? mk('Dubbed (no lip-sync)', 'clip.dubbed.mp4', 44_100_000) : null,
      lipsync_warning: null,
      enhanced_warning: null,
    },
    audio: {
      original: mk('Original audio', 'clip.16k.mono.wav', 8_400_000, {
        full_quality: false,
        note: '16 kHz mono — extracted for speech processing. The full-quality original is inside the video.',
      }),
      dubbed: done
        ? mk('Dubbed audio', 'clip.dubbed.wav', 26_900_000, { full_quality: true, note: null })
        : null,
    },
  };
}

const MOCK_TONES = [
  { value: 'AIR style news', label: 'AIR-style news', group: 'Delivery' },
  { value: 'TV style news', label: 'TV-style news', group: 'Delivery' },
  { value: 'news', label: 'News', group: 'Delivery' },
  { value: 'advertisements', label: 'Advertisements', group: 'Delivery' },
  { value: 'Customer Care', label: 'Customer care', group: 'Delivery' },
  { value: 'educational lecture', label: 'Educational lecture', group: 'Delivery' },
  { value: "children's stories", label: "Children's stories", group: 'Delivery' },
  { value: 'single person narration audiobook', label: 'Audiobook narration', group: 'Delivery' },
  { value: 'happy', label: 'Happy', group: 'Emotion' },
  { value: 'sad', label: 'Sad', group: 'Emotion' },
  { value: 'anger', label: 'Anger', group: 'Emotion' },
  { value: 'fear', label: 'Fear', group: 'Emotion' },
  { value: 'disgust', label: 'Disgust', group: 'Emotion' },
  { value: 'surprise', label: 'Surprise', group: 'Emotion' },
];

export async function getSpeakers(id: string): Promise<SpeakerTable> {
  await sleep(70);
  const voice = (
    v: string,
    gender: 'male' | 'female',
    tier: string,
    recommended_for: string[],
  ) => ({
    voice: v, engine: 'indic-speak', gender, age: 'adult', language: 'hi',
    tier, display: v, recommended_for,
  });
  return {
    job_id: id,
    engine: 'bodhan',
    engine_label: 'Bodhan Indic voices',
    language: 'hi',
    supports_tone: true,
    tone_options: MOCK_TONES,
    voice_options: [
      voice('Kavya', 'female', 'lead', ['SPEAKER_00']),
      voice('Suhani', 'female', 'support', ['SPEAKER_00']),
      voice('Dhara', 'female', 'support', ['SPEAKER_00']),
      voice('Amit', 'male', 'lead', ['SPEAKER_01']),
      voice('Ankur', 'male', 'support', ['SPEAKER_01']),
      voice('Sourav', 'male', 'support', ['SPEAKER_01']),
    ],
    speakers: [
      {
        id: 'SPEAKER_00', label: 'Prof. Mitra', speech_s: 252.4, n_segments: 38,
        gender: 'female', voice: 'Kavya', voice_source: 'assigned', voice_score: 0.81,

        voice_rationale: { gender_match: 1.0, embed_cos: 0.81 },
        tone: 'educational lecture', reference_url: null, color: '#6366f1',
      },
      {
        id: 'SPEAKER_01', label: null, speech_s: 110.0, n_segments: 14,
        gender: 'male', voice: 'Amit', voice_source: 'derived', voice_score: null,
        voice_rationale: null, tone: null, reference_url: null, color: '#06b6d4',
      },
      {
        id: 'SPEAKER_02', label: null, speech_s: 18.5, n_segments: 5,

        gender: null, voice: null, voice_source: null, voice_score: null,
        voice_rationale: null, tone: null, reference_url: null, color: '#f59e0b',
      },
    ],
  };
}

const MOCK_LABELS: Record<string, string> = { SPEAKER_00: 'Prof. Mitra' };

export async function setSpeakerLabel(
  id: string,
  speakerId: string,
  label: string,
): Promise<SpeakerLabels> {
  await sleep(120);
  if (label.trim()) MOCK_LABELS[speakerId] = label.trim();
  else delete MOCK_LABELS[speakerId];
  return { job_id: id, labels: { ...MOCK_LABELS } };
}

export async function getSpec(id: string): Promise<Spec> {
  await sleep(60);
  return {
    title: MOCK_JOBS.jobs.find((j) => j.job_id === id)?.title ?? null,
    thumbnail_url: null,
    lines: [
      { key: 'part', label: 'Part dubbed', value: '3:42 – 7:24 of 12:05', hint: null, tone: 'default' },
      { key: 'language', label: 'Dubbing into', value: 'Hindi (हिन्दी)', hint: null, tone: 'default' },
      { key: 'speakers', label: 'Speakers found', value: '2', hint: null, tone: 'default' },
      { key: 'voices', label: 'Voices from', value: 'Bodhan indic speak', hint: null, tone: 'default' },
      { key: 'tone', label: 'Tone control', value: 'educational lecture ', hint: 'Only the Bodhan voices accept a delivery style.', tone: 'warn' },
      { key: 'lipsync', label: 'Lip-sync', value: 'On — MuseTalk', hint: null, tone: 'default' },
      { key: 'started', label: 'Started', value: '12 minutes ago', hint: null, tone: 'default' },
    ],
    raw_config: { target_language: 'hi', tts: 'bodhan', lipsync: 'musetalk', skip_stems: false },
  };
}

export async function getOptions(_tts?: string): Promise<Options> {
  await sleep(40);
  return {
    languages: [
      { code: 'hi', english: 'Hindi', native: 'हिन्दी', script: 'deva', font: 'Noto Sans Devanagari' },
      { code: 'ta', english: 'Tamil', native: 'தமிழ்', script: 'taml', font: 'Noto Sans Tamil' },
      { code: 'bn', english: 'Bengali', native: 'বাংলা', script: 'beng', font: 'Noto Sans Bengali' },
      { code: 'te', english: 'Telugu', native: 'తెలుగు', script: 'telu', font: 'Noto Sans Telugu' },
    ],

    default_source_language: 'en',
    styles: [
      { value: 'TV style news', label: 'TV-style news' },
      { value: 'advertisements', label: 'Advertisements' },
      { value: 'educational lecture', label: 'Educational lecture' },
      { value: "children's stories", label: "Children's stories" },
    ],
    tts_engines: [
      { id: 'bodhan', label: 'Bodhan Indic voices', supports_tone: true,
        note: '22 Indian languages · a distinct voice per speaker' },
    ],

    lipsync_models: [
      { id: 'wav2lip', label: 'Wav2Lip', speed: '3–5× clip length' },
      { id: 'musetalk', label: 'MuseTalk', speed: '5-8× clip length' },
    ],
    limits: {
      max_lipsync_s: 1200,
      note: 'Lip-sync is capped at 20 minutes of video. It runs at 3–8× the clip length on this machine’s GPUs, so a longer clip means hours.',
    },
  };
}
