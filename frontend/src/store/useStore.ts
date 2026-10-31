/** Application state and actions. Inputs are user or API values; selectors return derived UI state. */
import { create } from 'zustand';
import * as api from '../api';
import type {
  AgentParams,
  CastSelection,
  EditHistory,
  EditPlan,
  JobRow,
  Media,
  PendingEdit,
  Progress,
  Spec,
  SpeakerTable,
  VideoTrack,
} from '../api/types';
import { ensureScriptFont } from '../lib/lang';
import { normalizeState } from '../lib/format';

export interface Toast {
  id: number;
  kind: 'info' | 'success' | 'warn' | 'error';
  title: string;
  body?: string;
  action?: { label: string; run: () => void };
}

export interface StagedChange {
  voice?: string | null;
  tone?: string | null;
  gender?: string | null;
  fromVoice?: string | null;
  fromTone?: string | null;
  fromGender?: string | null;
}

interface State {

  jobs: JobRow[];
  jobsLoaded: boolean;
  jobsError: string | null;
  search: string;

  selected: string | null;

  progress: Progress | null;
  media: Media | null;
  speakers: SpeakerTable | null;
  spec: Spec | null;
  jobError: string | null;

  railOpen: boolean;
  theme: 'dark' | 'light';
  toggleTheme: () => void;
  shortcutsOpen: boolean;

  agentParams: AgentParams | null;
  refreshAgentParams: () => Promise<AgentParams | null>;

  talkingHead: boolean;
  setTalkingHead: (v: boolean) => void;

  videoTrack: VideoTrack | null;

  videoTime: number;

  videoPlayerHeight: number | null;
  setVideoPlayerHeight: (h: number | null) => void;

  videoSeek: { t: number; nonce: number } | null;

  selectedCue: string | null;
  selectedSpeaker: string | null;
  toasts: Toast[];

  editHistory: EditHistory | null;

  staged: Record<string, StagedChange>;

  castMode: Record<string, 'autodetect' | 'select'>;
  editPlan: EditPlan | null;
  applying: boolean;

  bootstrap: () => Promise<void>;
  refreshJobs: () => Promise<void>;
  selectJob: (id: string | null) => void;
  refreshProgress: () => Promise<Progress | null>;
  refreshMedia: () => Promise<void>;
  refreshSpeakers: () => Promise<void>;
  refreshSpec: () => Promise<void>;
  setSearch: (s: string) => void;
  setRailOpen: (v: boolean) => void;
  setNewJobOpen: (v: boolean) => void;
  setShortcutsOpen: (v: boolean) => void;
  renameSpeaker: (speakerId: string, label: string) => Promise<void>;

  setVideoTrack: (t: VideoTrack) => void;
  setVideoTime: (t: number) => void;
  requestSeek: (t: number) => void;
  setSelectedCue: (id: string | null) => void;
  setSelectedSpeaker: (id: string | null) => void;
  pushToast: (t: Omit<Toast, 'id'>) => void;
  dismissToast: (id: number) => void;
  refreshEditHistory: () => Promise<void>;
  stageVoice: (speaker: string, voice: string) => void;
  stageTone: (speaker: string, tone: string) => void;
  stageGender: (speaker: string, gender: string) => void;
  setCastMode: (speaker: string, mode: 'autodetect' | 'select') => void;
  cast: () => Promise<void>;
  revertSpeaker: (speaker: string) => void;
  discardStaged: () => Promise<void>;
  previewPlan: () => Promise<EditPlan | null>;
  applyStaged: () => Promise<void>;
}

export function toPendingEdits(staged: Record<string, StagedChange>): PendingEdit[] {
  const out: PendingEdit[] = [];
  for (const [speaker, c] of Object.entries(staged)) {
    const edit: PendingEdit = { speaker_id: speaker };
    let touched = false;
    if (c.voice != null && c.voice !== c.fromVoice) {
      edit.voice = c.voice;
      touched = true;
    }
    if (c.tone !== undefined && c.tone !== c.fromTone) {
      edit.tone = c.tone;
      touched = true;
    }
    if (c.gender != null && c.gender !== c.fromGender) {
      edit.gender = c.gender;
      touched = true;
    }
    if (touched) out.push(edit);
  }
  return out;
}

export interface StagedDiff {
  speaker: string;
  field: 'voice' | 'tone' | 'gender';
  from: string | null;

  to: string | null;
}

export function toStagedDiffs(staged: Record<string, StagedChange>): StagedDiff[] {
  const out: StagedDiff[] = [];
  for (const [speaker, c] of Object.entries(staged)) {
    if (c.voice != null && c.voice !== c.fromVoice) {
      out.push({ speaker, field: 'voice', from: c.fromVoice ?? null, to: c.voice });
    }
    if (c.tone !== undefined && c.tone !== c.fromTone) {
      out.push({ speaker, field: 'tone', from: c.fromTone ?? null, to: c.tone });
    }
    if (c.gender != null && c.gender !== c.fromGender) {
      out.push({ speaker, field: 'gender', from: c.fromGender ?? null, to: c.gender });
    }
  }
  return out;
}

const LS_THEME = 'ds.theme';

function readTheme(): 'dark' | 'light' {
  try {
    const v = localStorage.getItem(LS_THEME);
    if (v === 'light' || v === 'dark') return v;
    return window.matchMedia?.('(prefers-color-scheme: light)').matches
      ? 'light'
      : 'dark';
  } catch {
    return 'dark';
  }
}

export function applyTheme(theme: 'dark' | 'light'): void {
  document.documentElement.setAttribute('data-theme', theme);
  try {
    localStorage.setItem(LS_THEME, theme);
  } catch {}
}

let toastSeq = 1;

export const useStore = create<State>()((set, get) => ({
  jobs: [],
  jobsLoaded: false,
  jobsError: null,
  search: '',
  selected: null,
  progress: null,
  media: null,
  speakers: null,
  spec: null,
  jobError: null,

  railOpen: true,
  theme: readTheme(),
  shortcutsOpen: false,
  agentParams: null,
  talkingHead: false,
  videoTrack: null,
  videoTime: 0,
  videoPlayerHeight: null,
  setVideoPlayerHeight: (h) => set({ videoPlayerHeight: h }),
  videoSeek: null,
  selectedCue: null,
  selectedSpeaker: null,
  toasts: [],
  editHistory: null,
  staged: {},
  castMode: {},
  editPlan: null,
  applying: false,

  async bootstrap() {
    await get().refreshJobs();

    const fromHash = decodeURIComponent((location.hash.match(/^#\/job\/(.+)$/) ?? [])[1] ?? '');
    const jobs = get().jobs;

    const atHome = location.hash === '#/' || location.hash === '#/new';
    const pick = (fromHash && jobs.some((j) => j.job_id === fromHash) && fromHash)
      || (atHome ? null : jobs[0]?.job_id)
      || null;
    if (pick) get().selectJob(pick);
  },

  async refreshJobs() {
    try {
      const res = await api.listJobs();
      set({ jobs: res.jobs ?? [], jobsLoaded: true, jobsError: null });
    } catch (e) {
      set({ jobsLoaded: true, jobsError: (e as Error).message });
    }
  },

  selectJob(id) {
    if (get().selected === id) return;
    set({
      selected: id,
      ...(id ? { talkingHead: false } : {}),
      agentParams: null,
      editHistory: null,
      progress: null,
      media: null,
      speakers: null,
      spec: null,
      jobError: null,
      staged: {},
      castMode: {},
      editPlan: null,

      videoTrack: null,
      videoTime: 0,
      videoSeek: null,
      selectedCue: null,
      selectedSpeaker: null,
      railOpen: true,
    });
    const target = id ? `#/job/${encodeURIComponent(id)}` : '#/';
    if (location.hash !== target) history.replaceState(null, '', target);
    if (!id) return;

    void get().refreshProgress();
    void get().refreshMedia();
    void get().refreshSpec();
    void get().refreshSpeakers();
  },

  async refreshProgress() {
    const id = get().selected;
    if (!id) return null;
    try {
      const p = await api.getProgress(id);
      if (get().selected !== id) return null;
      const before = get().progress;
      set({ progress: p, jobError: null });

      if (p.milestones?.lipsync_ready && !before?.milestones?.lipsync_ready) {
        void get().refreshMedia();
        get().pushToast({
          kind: 'success',
          title: 'The lip-synced version is ready',
          body: 'Switch between it and the original above the player to compare them.',
          action: {
            label: 'Watch it now',
            run: () => set({ videoTrack: 'lipsync' }),
          },
        });
      }
      if (p.milestones?.diarization_done && !before?.milestones?.diarization_done) {
        void get().refreshSpeakers();
      }

      if (p.milestones?.awaiting_cast && !before?.milestones?.awaiting_cast) {
        void get().refreshSpeakers();
        get().pushToast({
          kind: 'info',
          title: 'Translation complete — pick the voices to carry on',
          body: 'Choose a voice for each speaker;',
          action: {
            label: 'Go to the cast',
            run: () => document.getElementById('cast-panel')
              ?.scrollIntoView({ behavior: 'smooth', block: 'start' }),
          },
        });
      }
      if (p.milestones?.speakers_ready && !before?.milestones?.speakers_ready) {
        void get().refreshSpeakers();
      }
      if (p.milestones?.mix_done && !before?.milestones?.mix_done) {
        void get().refreshMedia();
        get().pushToast({
          kind: 'success',
          title: 'The dubbed soundtrack is ready',
          body: 'Play it against the original under the video.',
        });
      }
      if (before && before.state !== p.state) {
        void get().refreshJobs();
        void get().refreshSpec();
        if (p.state === 'failed' && p.failure) {
          get().pushToast({ kind: 'error', title: p.failure.title, body: p.failure.detail });
        }
      }
      return p;
    } catch (e) {
      if (get().selected === id) set({ jobError: (e as Error).message });
      return null;
    }
  },

  async refreshMedia() {
    const id = get().selected;
    if (!id) return;
    try {
      const m = await api.getMedia(id);
      if (get().selected === id) set({ media: m });
    } catch {}
  },

  async refreshSpeakers() {
    const id = get().selected;
    if (!id) return;
    try {
      const s = await api.getSpeakers(id);
      if (get().selected !== id) return;
      ensureScriptFont(s.language);
      set({ speakers: s });
    } catch {}
  },

  async refreshSpec() {
    const id = get().selected;
    if (!id) return;
    try {
      const s = await api.getSpec(id);
      if (get().selected === id) set({ spec: s });
    } catch {}
  },

  async refreshAgentParams() {
    const id = get().selected;
    if (!id) return null;
    try {
      const p = await api.getAgentParams(id);
      if (get().selected !== id) return null;
      set({ agentParams: p });
      return p;
    } catch {
      if (get().selected === id) set({ agentParams: null });
      return null;
    }
  },

  async refreshEditHistory() {
    const id = get().selected;
    if (!id) return;
    try {
      const h = await api.getEditHistory(id);
      if (get().selected === id) set({ editHistory: h });
    } catch {
      if (get().selected === id) set({ editHistory: null });
    }
  },

  setSearch: (s) => set({ search: s }),

  setRailOpen: (v) => set({ railOpen: v }),
  toggleTheme: () => {
    const next = get().theme === 'dark' ? 'light' : 'dark';
    applyTheme(next);
    set({ theme: next });
  },

  setNewJobOpen: (v) => {
    if (!v) return;
    if (location.hash !== '#/') location.hash = '#/';
    useStore.getState().selectJob(null);

    setTimeout(() => {
      const el = document.querySelector<HTMLInputElement>('#start-url');
      el?.focus();
    }, 60);
  },
  setShortcutsOpen: (v) => set({ shortcutsOpen: v }),
  setTalkingHead: (v) => set({ talkingHead: v }),
  setVideoTrack: (t) => set({ videoTrack: t }),
  setVideoTime: (t) => set({ videoTime: t }),
  requestSeek: (t) => set({ videoSeek: { t, nonce: Date.now() } }),
  setSelectedCue: (id) => set({ selectedCue: id, selectedSpeaker: null }),
  setSelectedSpeaker: (id) => set({ selectedSpeaker: id, selectedCue: null }),

  async renameSpeaker(speakerId, label) {
    const id = get().selected;
    const table = get().speakers;
    if (!id || !table) return;
    const clean = label.trim().slice(0, 80);
    const before = table.speakers.find((sp) => sp.id === speakerId)?.label ?? null;
    const patch = (value: string | null) => {
      const t = get().speakers;
      if (!t || t.job_id !== id) return;
      set({
        speakers: {
          ...t,
          speakers: t.speakers.map((sp) => (sp.id === speakerId ? { ...sp, label: value } : sp)),
        },
      });
    };
    patch(clean || null);
    try {
      const res = await api.setSpeakerLabel(id, speakerId, clean);
      if (get().selected === id) patch(res.labels?.[speakerId] ?? null);
    } catch (e) {
      patch(before);
      get().pushToast({
        kind: 'error',
        title: 'That name was not saved',
        body: (e as Error).message,
      });
    }
  },

  pushToast(t) {
    const id = toastSeq++;
    set({ toasts: [...get().toasts, { ...t, id }] });
    setTimeout(() => get().dismissToast(id), 9000);
  },

  dismissToast(id) {
    set({ toasts: get().toasts.filter((t) => t.id !== id) });
  },

  stageVoice(speaker, voice) {
    const row = get().speakers?.speakers.find((s) => s.id === speaker);
    const prev = get().staged[speaker] ?? {};
    const fromVoice = prev.fromVoice ?? row?.voice ?? null;
    const next: Record<string, StagedChange> = { ...get().staged };
    if (voice === fromVoice && prev.tone === undefined) {
      delete next[speaker];
    } else {
      next[speaker] = { ...prev, voice, fromVoice };
    }
    set({ staged: next, editPlan: null });
  },

  stageTone(speaker, tone) {
    const row = get().speakers?.speakers.find((s) => s.id === speaker);
    const prev = get().staged[speaker] ?? {};
    const fromTone = prev.fromTone ?? row?.tone ?? null;
    const next: Record<string, StagedChange> = { ...get().staged };
    if (tone === fromTone && prev.voice === undefined) {
      delete next[speaker];
    } else {
      next[speaker] = { ...prev, tone, fromTone };
    }
    set({ staged: next, editPlan: null });
  },

  stageGender(speaker, gender) {
    const row = get().speakers?.speakers.find((s) => s.id === speaker);
    const prev = get().staged[speaker] ?? {};
    const fromGender = prev.fromGender ?? row?.gender ?? null;
    const next: Record<string, StagedChange> = { ...get().staged };
    if (gender === fromGender && prev.voice === undefined && prev.tone === undefined) {
      delete next[speaker];
    } else {
      next[speaker] = { ...prev, gender, fromGender };
    }
    set({ staged: next, editPlan: null });
  },

  revertSpeaker(speaker) {
    const next = { ...get().staged };
    delete next[speaker];
    set({ staged: next, editPlan: null });
  },

  setCastMode(speaker, mode) {
    set({ castMode: { ...get().castMode, [speaker]: mode } });
  },

  async cast() {
    const id = get().selected;
    const table = get().speakers;
    if (!id || !table) return;
    const staged = get().staged;
    const payload: Record<string, CastSelection> = {};
    for (const sp of table.speakers) {
      const st = staged[sp.id];
      payload[sp.id] = {
        mode: 'select',
        gender: st?.gender !== undefined ? st.gender : sp.gender,
        voice: st?.voice !== undefined ? st.voice : sp.voice,
      };
    }
    set({ applying: true });
    try {
      await api.castJob(id, payload);
      set({ staged: {}, castMode: {}, editPlan: null });
      get().pushToast({
        kind: 'success',
        title: 'Casting the voices',
        body: 'Running the speech, mix and timing stages with the chosen cast. Lip-sync stays a separate step.',
      });
      await get().refreshProgress();
      await get().refreshJobs();
    } catch (e) {
      get().pushToast({
        kind: 'error',
        title: 'Could not start casting',
        body: (e as Error).message,
      });
    } finally {
      set({ applying: false });
    }
  },

  async discardStaged() {
    const id = get().selected;
    set({ staged: {}, editPlan: null });
    if (id) {
      try {
        await api.discardEdits(id);
      } catch {}
    }
  },

  async previewPlan() {
    const id = get().selected;
    if (!id) return null;
    const pending = toPendingEdits(get().staged);
    if (!pending.length) return null;
    try {
      const plan = await api.stageEdits(id, pending);
      if (get().selected === id) set({ editPlan: plan });
      return plan;
    } catch (e) {
      get().pushToast({
        kind: 'error',
        title: 'Could not work out what that change costs',
        body: `${(e as Error).message} You can still apply it — you just will not see an estimate first.`,
      });
      return null;
    }
  },

  async applyStaged() {
    const id = get().selected;
    if (!id) return;
    set({ applying: true });
    try {
      await api.applyEdits(id);
      set({ staged: {}, editPlan: null });
      get().pushToast({
        kind: api.MOCKED.edits ? 'warn' : 'success',
        title: api.MOCKED.edits
          ? 'Staged only — the apply endpoint is not built yet'
          : 'Re-rendering with the new voices',
        body: api.MOCKED.edits
          ? 'Nothing was re-rendered. POST /api/jobs/{id}/edits/apply does not exist on the server yet.'
          : 'Only the speech, mix and lip-sync stages re-run — the transcript and translation are kept.',
      });
      await get().refreshProgress();
      await get().refreshJobs();
    } catch (e) {
      get().pushToast({
        kind: 'error',
        title: 'Those changes were not applied',
        body: `${(e as Error).message} Nothing was re-rendered, and your changes are still staged.`,
      });
    } finally {
      set({ applying: false });
    }
  },
}));

export function filterJobs(jobs: JobRow[], search: string): JobRow[] {
  const q = search.trim().toLowerCase();
  if (!q) return jobs;
  return jobs.filter(
    (j) =>
      (j.title ?? '').toLowerCase().includes(q)
      || j.slug.toLowerCase().includes(q)
      || j.job_id.toLowerCase().includes(q)
      || (j.language ?? '').toLowerCase().includes(q),
  );
}

export const selectStagedCount = (s: State): number => toStagedDiffs(s.staged).length;

export const selectIsTerminal = (s: State): boolean => {
  const st = normalizeState(s.progress?.state);
  return st === 'done' || st === 'failed' || st === 'cancelled';
};
