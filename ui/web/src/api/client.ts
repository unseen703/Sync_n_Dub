/**
 * Plain `fetch` against the FastAPI server on the same origin.
 *
 * No client-side cache layer: there are six endpoints and exactly one hot
 * path (`/progress` at 1.5 s), and the cadence rules we actually need —
 * back off on no-change, stop dead on terminal, fire side effects on a
 * milestone flip — are the part a query cache is weakest at. `usePoll`
 * owns the cadence; this file only speaks HTTP.
 */
import type {
  EditHistory,
  EnhanceOptions,
  EnhanceReadiness,
  EnhanceStatus,
  ReviewVideoStatus,
  ApplyResult,
  CreateJobRequest,
  CreateJobResult,
  EditPlan,
  JobList,
  LipsyncReadiness,
  LipsyncStarted,
  CastStarted,
  CastSelection,
  Media,
  Options,
  PendingEdit,
  AgentParams,
  AgentScript,
  AgentTurn,
  ProbeResult,
  ScriptEdit,
  UploadResult,
  Progress,
  Spec,
  SpeakerLabels,
  SpeakerTable,
} from './types';

export class ApiError extends Error {
  status: number;
  body: unknown;
  constructor(status: number, message: string, body?: unknown) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.body = body;
  }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: {
      accept: 'application/json',
      ...(init?.body ? { 'content-type': 'application/json' } : null),
      ...init?.headers,
    },
  });
  if (!res.ok) {
    let body: unknown = null;
    let detail = res.statusText;
    try {
      body = await res.json();
      const d = (body as { detail?: unknown })?.detail;
      if (typeof d === 'string') detail = d;
    } catch {
      /* non-JSON error body; the status line is all we get */
    }
    throw new ApiError(res.status, detail || `HTTP ${res.status}`, body);
  }
  if (res.status === 204) return null as T;
  return (await res.json()) as T;
}

const enc = encodeURIComponent;

// ---------------------------------------------------------------------------
// read path — all of these exist and work today
// ---------------------------------------------------------------------------

/**
 * The job list, preferring the in-flight request `index.html` started before
 * the bundle parsed. Consumed once; every later call goes over the wire.
 */
let bootConsumed = false;
export async function listJobs(): Promise<JobList> {
  if (!bootConsumed) {
    bootConsumed = true;
    const boot = (window as unknown as { __BOOT__?: Promise<JobList | null> }).__BOOT__;
    if (boot) {
      const early = await boot.catch(() => null);
      if (early && Array.isArray(early.jobs)) return early;
    }
  }
  return req<JobList>('/api/jobs');
}

export const getProgress = (id: string) => req<Progress>(`/api/jobs/${enc(id)}/progress`);
export const getMedia = (id: string) => req<Media>(`/api/jobs/${enc(id)}/media`);
export const getSpeakers = (id: string) => req<SpeakerTable>(`/api/jobs/${enc(id)}/speakers`);
export const getSpec = (id: string) => req<Spec>(`/api/jobs/${enc(id)}/spec`);
export const getOptions = (tts?: string) =>
  req<Options>(`/api/options${tts ? `?tts=${enc(tts)}` : ''}`);

// ---------------------------------------------------------------------------
// write path — NOT YET BUILT on the server.
//
// These are written against the shapes in the plan. `src/api/index.ts`
// swaps them for `mock.ts` when VITE_MOCK is set, so the whole app is
// runnable before the endpoints land. When they land, delete nothing here —
// just stop setting VITE_MOCK.
// ---------------------------------------------------------------------------

export const probeYouTube = (url: string) =>
  req<ProbeResult>('/api/youtube/probe', { method: 'POST', body: JSON.stringify({ url }) });

/**
 * Upload a local video. Multipart, so it does NOT go through `req()` —
 * that sets `Content-Type: application/json`, and a hand-set content type
 * on a FormData body omits the multipart boundary the server needs.
 */
export const uploadVideo = async (file: File): Promise<UploadResult> => {
  const body = new FormData();
  body.append('file', file);
  const r = await fetch('/api/uploads', { method: 'POST', body });
  if (!r.ok) {
    let detail = `Upload failed (${r.status})`;
    try {
      const j = await r.json();
      if (j?.detail) detail = typeof j.detail === 'string' ? j.detail : detail;
    } catch {
      /* a non-JSON error body is not more informative than the status */
    }
    throw new Error(detail);
  }
  return (await r.json()) as UploadResult;
};

export const createJob = (payload: CreateJobRequest) =>
  req<CreateJobResult>('/api/jobs', { method: 'POST', body: JSON.stringify(payload) });

export const cancelJob = (id: string) =>
  req<{ job_id: string; state: string }>(`/api/jobs/${enc(id)}/cancel`, { method: 'POST' });

export const getEdits = (id: string) => req<EditPlan>(`/api/jobs/${enc(id)}/edits`);
export const getEditHistory = (id: string) => req<EditHistory>(`/api/jobs/${enc(id)}/history`);

// --- the plain-language edit agent -----------------------------------------
// `backend/edit_agent.py` behind three routes. `agentSend` STAGES; it never
// re-renders. `applyEdits` is still the only call that spends GPU time.
export const getAgentParams = (id: string) =>
  req<AgentParams>(`/api/jobs/${enc(id)}/agent/params`);

export const getAgentScript = (id: string, limit = 2000) =>
  req<AgentScript>(`/api/jobs/${enc(id)}/agent/script?limit=${limit}`);

export const agentSend = (id: string, message: string, replace = false) =>
  req<EditPlan & { turns?: AgentTurn[]; source?: string }>(
    `/api/jobs/${enc(id)}/agent`,
    { method: 'POST', body: JSON.stringify({ message, replace }) },
  );

// The body key is `edits`, NOT `pending`. This sent `{ pending }` while
// `ui/routes_edits.py::stage_edits` read `body.get("edits")`, so every voice
// and tone change answered 422 and the following apply answered 409. The
// backend now accepts both, but this is the documented key and the one
// `tests/test_ui_edits.py` posts.
export const stageEdits = (id: string, pending: PendingEdit[]) =>
  req<EditPlan>(`/api/jobs/${enc(id)}/edits`, {
    method: 'POST',
    body: JSON.stringify({ edits: pending }),
  });

export const discardEdits = (id: string) =>
  req<EditPlan>(`/api/jobs/${enc(id)}/edits`, { method: 'DELETE' });

/**
 * Rename a speaker. Display only — `ui/server.py::api_set_speaker_label`
 * writes a job-local file that nothing downstream reads, so this dirties no
 * stage and must never be routed through the staged-edit machinery.
 * An empty string clears the name.
 */
export const setSpeakerLabel = (id: string, speakerId: string, label: string) =>
  req<SpeakerLabels>(`/api/jobs/${enc(id)}/speakers/${enc(speakerId)}/label`, {
    method: 'POST',
    body: JSON.stringify({ label }),
  });

// Hand-edited script lines. The user typed the wording, so unlike the agent's
// `set_text` -- where an LLM invents it -- the quoted cost is the applied
// cost. Staging only; `applyEdits` is still the single call that spends GPU.
export const stageScript = (id: string, edits: ScriptEdit[]) =>
  req<EditPlan>(`/api/jobs/${enc(id)}/script`, {
    method: 'POST',
    body: JSON.stringify({ edits }),
  });

/** Mute a cue directly — not an agent parse. Deduped on the
 *  server: muting the same cue twice replaces rather than stacking. */
export const muteCue = (id: string, segmentId: string) =>
  req<EditPlan & { turns?: AgentTurn[] }>(
    `/api/jobs/${enc(id)}/script/mute`,
    { method: 'POST', body: JSON.stringify({ segment_id: segmentId }) },
  );

export const applyEdits = (id: string) =>
  req<ApplyResult>(`/api/jobs/${enc(id)}/edits/apply`, { method: 'POST' });

/**
 * Remove ONE staged change and keep the rest. `DELETE .../edits` throws the
 * whole set away; this drops a single turn and re-prices what is left, so a
 * misparsed sentence no longer costs the four good ones beside it.
 */
export const dropStagedTurn = (id: string, index: number) =>
  req<EditPlan>(`/api/jobs/${enc(id)}/agent/turns/${index}`, {
    method: 'DELETE',
  });

/**
 * What the lip-sync step needs: blockers, the models, and the face size ASD
 * measured. Read-only — safe to poll while edits are being staged.
 */
export const getLipsyncReadiness = (id: string) =>
  req<LipsyncReadiness>(`/api/jobs/${enc(id)}/lipsync/readiness`);

/**
 * Start the lip-sync render. `model` is REQUIRED — the server 422s without
 * one rather than defaulting, because musetalk is 8-21x clip length here
 * against wav2lip's 3-5x and guessing costs real hours.
 */
export const startLipsync = (id: string, model: string) =>
  req<LipsyncStarted>(`/api/jobs/${enc(id)}/lipsync`, {
    method: 'POST',
    body: JSON.stringify({ model }),
  });

/**
 * Finish the cast step for a job stopped after MT, then run the post-MT
 * stages. Each speaker carries a `mode`: `select` pins the chosen gender/
 * voice as an override, `autodetect` lets the pipeline predict it.
 */
export const castJob = (id: string, speakers: Record<string, CastSelection>) =>
  req<CastStarted>(`/api/jobs/${enc(id)}/cast`, {
    method: 'POST',
    body: JSON.stringify({ speakers }),
  });

/** Stage 9i readiness — drives the Enhance resolution button's state. */
export const getEnhanceReadiness = (id: string, render?: string) =>
  req<EnhanceReadiness>(
    `/api/jobs/${enc(id)}/enhance/readiness`
    + (render ? `?render=${encodeURIComponent(render)}` : ''));

/** Poll one job's enhancement. The server treats the FILE as the arbiter. */
export const getEnhanceStatus = (id: string) =>
  req<EnhanceStatus>(`/api/jobs/${enc(id)}/enhance`);

/**
 * Start the pass. Returns immediately — it is a background thread, because
 * even gated and split over both GPUs this is minutes, and an inline
 * request would die to a browser timeout while the GPU work carried on.
 */
export const startEnhance = (id: string, opts: EnhanceOptions = {}) =>
  req<EnhanceStatus>(`/api/jobs/${enc(id)}/enhance`, {
    method: 'POST',
    body: JSON.stringify({ gated: true, scale: 'same', ...opts }),
  });

/** Poll the ASD review ("bbox") video. `url` appears once it is ready. */
export const getReviewVideo = (id: string) =>
  req<ReviewVideoStatus>(`/api/jobs/${enc(id)}/review-video`);

/**
 * Render the ASD review video in a background thread, returning immediately.
 * A full decode + composite over a 1080p clip is minutes, so the panel polls
 * `getReviewVideo` rather than waiting on this.
 */
export const makeReviewVideo = (id: string) =>
  req<ReviewVideoStatus>(`/api/jobs/${enc(id)}/review-video`, { method: 'POST' });
