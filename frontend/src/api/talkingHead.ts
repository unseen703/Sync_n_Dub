
/** Talking-head API client. Inputs are a photo, language, dialogue, and render settings; outputs are typed server responses. */
export interface THLanguage {
  code: string;
  english: string;
  native: string;
  n_dialogues: number;
  has_dialogues: boolean;
}

export interface THCategory {
  id: string;
  label: string;
}

export interface THDialogue {
  id: string;
  category: string;
  text: string;
  audio_url: string | null;

  playable: boolean;
}

export interface THOptions {
  languages: THLanguage[];
  categories: THCategory[];
  note: string;
  max_image_mb: number;
}

export interface THPhoto {
  photo_id: string;
  url: string | null;
  ok: boolean;
  reason: string;
  message?: string;
  width?: number;
  height?: number;
  n_faces?: number;
  face_box?: [number, number, number, number];
  face_px?: number;
  warnings?: string[];
  note?: string;
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `${res.status}`;
    try {
      const body = await res.json();
      detail = body?.detail ?? body?.message ?? detail;
    } catch {}
    throw new Error(String(detail));
  }
  return (await res.json()) as T;
}

export function thOptions(): Promise<THOptions> {
  return fetch('/api/talking-head/options').then(json<THOptions>);
}

export interface THDialogueList {
  dialogues: THDialogue[];
  categories: THCategory[];
  n_playable: number;

  message: string | null;
}

export function thDialogues(language: string): Promise<THDialogueList> {
  return fetch(
    `/api/talking-head/dialogues?language=${encodeURIComponent(language)}`,
  ).then(json<THDialogueList>);
}

export function thUploadPhoto(file: File): Promise<THPhoto> {
  const fd = new FormData();
  fd.append('file', file);
  return fetch('/api/talking-head/photo', { method: 'POST', body: fd }).then(
    json<THPhoto>,
  );
}

export interface THRender {
  render_id: string;
  state: 'running' | 'done' | 'failed' | 'absent';
  ready: boolean;
  url?: string | null;
  model?: string;
  error?: string;
  duration_s?: number;
  fps?: number;
  note?: string;
}

export function thRender(body: {
  photo_id: string;
  dialogue_id: string;
  language: string;
}): Promise<THRender> {
  return fetch('/api/talking-head/render', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  }).then(json<THRender>);
}

export function thRenderStatus(renderId: string): Promise<THRender> {
  return fetch(
    `/api/talking-head/render/${encodeURIComponent(renderId)}`,
  ).then(json<THRender>);
}
