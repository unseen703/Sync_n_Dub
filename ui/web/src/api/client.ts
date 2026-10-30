/** HTTP client for the Dubbing Studio API. Inputs are route parameters and request bodies; results are typed API payloads. */
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
  RevertResult,
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
    } catch {}
    throw new ApiError(res.status, detail || `HTTP ${res.status}`, body);
  }
  if (res.status === 204) return null as T;
  return (await res.json()) as T;
}

const enc = encodeURIComponent;

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

export const probeYouTube = (url: string) =>
  req<ProbeResult>('/api/youtube/probe', { method: 'POST', body: JSON.stringify({ url }) });

export const uploadVideo = async (file: File): Promise<UploadResult> => {
  const body = new FormData();
  body.append('file', file);
  const r = await fetch('/api/uploads', { method: 'POST', body });
  if (!r.ok) {
    let detail = `Upload failed (${r.status})`;
    try {
      const j = await r.json();
      if (j?.detail) detail = typeof j.detail === 'string' ? j.detail : detail;
    } catch {}
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

export const getAgentParams = (id: string) =>
  req<AgentParams>(`/api/jobs/${enc(id)}/agent/params`);

export const getAgentScript = (id: string, limit = 2000) =>
  req<AgentScript>(`/api/jobs/${enc(id)}/agent/script?limit=${limit}`);

export const agentSend = (id: string, message: string, replace = false) =>
  req<EditPlan & { turns?: AgentTurn[]; source?: string }>(
    `/api/jobs/${enc(id)}/agent`,
    { method: 'POST', body: JSON.stringify({ message, replace }) },
  );

export const stageEdits = (id: string, pending: PendingEdit[]) =>
  req<EditPlan>(`/api/jobs/${enc(id)}/edits`, {
    method: 'POST',
    body: JSON.stringify({ edits: pending }),
  });

export const discardEdits = (id: string) =>
  req<EditPlan>(`/api/jobs/${enc(id)}/edits`, { method: 'DELETE' });

export const setSpeakerLabel = (id: string, speakerId: string, label: string) =>
  req<SpeakerLabels>(`/api/jobs/${enc(id)}/speakers/${enc(speakerId)}/label`, {
    method: 'POST',
    body: JSON.stringify({ label }),
  });

export const stageScript = (id: string, edits: ScriptEdit[]) =>
  req<EditPlan>(`/api/jobs/${enc(id)}/script`, {
    method: 'POST',
    body: JSON.stringify({ edits }),
  });

export const muteCue = (id: string, segmentId: string, on = true) =>
  req<EditPlan & { turns?: AgentTurn[] }>(
    `/api/jobs/${enc(id)}/script/mute`,
    { method: 'POST', body: JSON.stringify({ segment_id: segmentId, on }) },
  );

export const applyEdits = (id: string) =>
  req<ApplyResult>(`/api/jobs/${enc(id)}/edits/apply`, { method: 'POST' });

export const dropStagedTurn = (id: string, index: number) =>
  req<EditPlan>(`/api/jobs/${enc(id)}/agent/turns/${index}`, {
    method: 'DELETE',
  });

export const revertEdits = (id: string, segments?: string[]) =>
  req<RevertResult>(`/api/jobs/${enc(id)}/revert`, {
    method: 'POST',
    body: JSON.stringify(segments ? { segments } : {}),
  });

export const getLipsyncReadiness = (id: string) =>
  req<LipsyncReadiness>(`/api/jobs/${enc(id)}/lipsync/readiness`);

export const startLipsync = (id: string, model: string) =>
  req<LipsyncStarted>(`/api/jobs/${enc(id)}/lipsync`, {
    method: 'POST',
    body: JSON.stringify({ model }),
  });

export const castJob = (id: string, speakers: Record<string, CastSelection>) =>
  req<CastStarted>(`/api/jobs/${enc(id)}/cast`, {
    method: 'POST',
    body: JSON.stringify({ speakers }),
  });

export const getEnhanceReadiness = (id: string, render?: string) =>
  req<EnhanceReadiness>(
    `/api/jobs/${enc(id)}/enhance/readiness`
    + (render ? `?render=${encodeURIComponent(render)}` : ''));

export const getEnhanceStatus = (id: string) =>
  req<EnhanceStatus>(`/api/jobs/${enc(id)}/enhance`);

export const startEnhance = (id: string, opts: EnhanceOptions = {}) =>
  req<EnhanceStatus>(`/api/jobs/${enc(id)}/enhance`, {
    method: 'POST',
    body: JSON.stringify({ gated: true, scale: 'same', ...opts }),
  });

export const getReviewVideo = (id: string) =>
  req<ReviewVideoStatus>(`/api/jobs/${enc(id)}/review-video`);

export const makeReviewVideo = (id: string) =>
  req<ReviewVideoStatus>(`/api/jobs/${enc(id)}/review-video`, { method: 'POST' });
