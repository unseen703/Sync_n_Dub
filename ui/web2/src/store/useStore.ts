/**
 * All app state, in one Zustand store.
 *
 * Zustand rather than a query library: there are six endpoints and one hot
 * path, so the cache-invalidation graph a query client exists to manage does
 * not exist here. What we do need — selector subscriptions, so a 1.5 s
 * progress tick does not re-render the speaker table — is exactly what
 * Zustand gives for ~1.2 kB.
 *
 * Fetching cadence is NOT here. `hooks/usePoll.ts` owns it, and calls the
 * `refresh*` actions below. That split is what makes a later swap to SSE a
 * one-hook change.
 */
import { create } from 'zustand';
import * as api from '../api';
import type {
  EditPlan,
  JobRow,
  Media,
  PendingEdit,
  Progress,
  Spec,
  SpeakerTable,
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

/** One speaker's staged, not-yet-applied changes. */
export interface StagedChange {
  voice?: string | null;
  tone?: string | null;
  fromVoice?: string | null;
  fromTone?: string | null;
}

interface State {
  // -- job list ------------------------------------------------------------
  jobs: JobRow[];
  jobsLoaded: boolean;
  jobsError: string | null;
  search: string;

  // -- selection -----------------------------------------------------------
  selected: string | null;

  // -- the selected job's four payloads ------------------------------------
  progress: Progress | null;
  media: Media | null;
  speakers: SpeakerTable | null;
  spec: Spec | null;
  jobError: string | null;

  // -- chrome --------------------------------------------------------------
  specOpen: boolean;
  railOpen: boolean;
  theme: 'dark' | 'light';
  toggleTheme: () => void;
  shortcutsOpen: boolean;
  /**
   * Which video the player shows. `null` means "nobody has chosen" — the
   * player then FOLLOWS THE DEFAULT, which is the lip-synced version
   * wherever one exists and the original until it does.
   *
   * The three states are not two: an explicit `'original'` has to outrank
   * the default, or the next poll after a render finishes would yank a
   * user who deliberately went back to the original onto the lip-synced
   * track. Null is what distinguishes "not chosen yet" from "chose the
   * original".
   */
  /** True on `#/job/<id>/edit` — the plain-language agent page. */
  editing: boolean;
  setEditing: (v: boolean) => void;
  /** `#/talking-head` — the still-portrait page. */
  talkingHead: boolean;
  setTalkingHead: (v: boolean) => void;
  videoTrack: 'original' | 'lipsync' | null;
  toasts: Toast[];

  // -- staged voice/tone edits --------------------------------------------
  staged: Record<string, StagedChange>;
  editPlan: EditPlan | null;
  applying: boolean;

  // -- actions -------------------------------------------------------------
  bootstrap: () => Promise<void>;
  refreshJobs: () => Promise<void>;
  selectJob: (id: string | null) => void;
  refreshProgress: () => Promise<Progress | null>;
  refreshMedia: () => Promise<void>;
  refreshSpeakers: () => Promise<void>;
  refreshSpec: () => Promise<void>;
  setSearch: (s: string) => void;
  toggleSpec: () => void;
  setRailOpen: (v: boolean) => void;
  setNewJobOpen: (v: boolean) => void;
  setShortcutsOpen: (v: boolean) => void;
  renameSpeaker: (speakerId: string, label: string) => Promise<void>;
  /** A deliberate choice: it pins, and outranks the default from then on. */
  setVideoTrack: (t: 'original' | 'lipsync') => void;
  pushToast: (t: Omit<Toast, 'id'>) => void;
  dismissToast: (id: number) => void;
  stageVoice: (speaker: string, voice: string) => void;
  stageTone: (speaker: string, tone: string) => void;
  revertSpeaker: (speaker: string) => void;
  discardStaged: () => Promise<void>;
  previewPlan: () => Promise<EditPlan | null>;
  applyStaged: () => Promise<void>;
}

const LS_SPEC = 'ds.specOpen';

function readSpecOpen(): boolean {
  try {
    const v = localStorage.getItem(LS_SPEC);
    if (v === null) return window.innerWidth >= 1536;
    return v === '1';
  } catch {
    return false;
  }
}

/**
 * `staged` -> the wire shape `POST /api/jobs/{id}/edits` takes.
 *
 * ONE entry per speaker carrying the fields that changed — not one entry per
 * field. `ui/routes_edits.py::_build_ops` reads `speaker_id` plus optional
 * `voice` / `tone` off each item and builds one `EditOp` per field present.
 */
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
    if (touched) out.push(edit);
  }
  return out;
}

/**
 * The same changes, shaped for DISPLAY — one row per changed field, with the
 * previous value so the dialog can say "Kavya -> Amit".
 *
 * Deliberately separate from the wire type: the server does not need the
 * old value (it reads that from the job itself), and widening `PendingEdit`
 * to carry it would make the API type a description of this component
 * rather than of the endpoint.
 */
export interface StagedDiff {
  speaker: string;
  field: 'voice' | 'tone';
  from: string | null;
  /** null only for a tone reset — a voice is never cleared. */
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
  }
  return out;
}

const LS_THEME = 'ds.theme';

/**
 * The saved theme, or the one the operating system asks for.
 *
 * Read in a try/catch because `localStorage` throws outright in a private
 * window and in some embedded webviews — a preference is not worth a blank
 * screen.
 */
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

/** Sets the attribute `index.css` keys the light palette off. */
export function applyTheme(theme: 'dark' | 'light'): void {
  document.documentElement.setAttribute('data-theme', theme);
  try {
    localStorage.setItem(LS_THEME, theme);
  } catch {
    /* preference only */
  }
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
  specOpen: readSpecOpen(),
  // The job list is navigation, so it starts open. The details panel is
  // reference and remembers whatever the user last chose.
  railOpen: true,
  theme: readTheme(),
  shortcutsOpen: false,
  editing: false,
  talkingHead: false,
  videoTrack: null,
  toasts: [],
  staged: {},
  editPlan: null,
  applying: false,

  async bootstrap() {
    await get().refreshJobs();
    // The hash is the router: `#/job/<id>`. No react-router, no SPA
    // catch-all route on the Python side.
    const fromHash = decodeURIComponent((location.hash.match(/^#\/job\/(.+)$/) ?? [])[1] ?? '');
    const jobs = get().jobs;

    // `#/` and `#/new` are DESTINATIONS, not an absent hash. Falling back
    // to `jobs[0]` on them made the landing page unreachable: the router
    // cleared the selection on `#/`, then this re-selected the newest job
    // a moment later when `refreshJobs()` resolved, so clicking the
    // wordmark changed the hash and nothing else. Only a hash that names
    // no destination at all gets the convenience pick.
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
      // Selecting a job means LOOKING at it, never editing it. This has to
      // be set here and not left to `useHashRoute`, because the line below
      // moves the hash with `history.replaceState`, which does NOT fire
      // `hashchange` — so the route handler never runs and `editing` stayed
      // true from whatever page you were on.
      //
      // The visible symptom was "Back to the dub does nothing": picking
      // another job from the rail while on the agent page left `editing`
      // true with the hash already at `#/job/<newId>`, so the Back link
      // pointed at the hash it was already on, no event fired, and the
      // click was a no-op.
      editing: false,
      progress: null,
      media: null,
      speakers: null,
      spec: null,
      jobError: null,
      staged: {},
      editPlan: null,
      // Back to following the default — the next job gets its own
      // lip-synced-first treatment rather than inheriting this one's pin.
      videoTrack: null,
      railOpen: true,
    });
    const target = id ? `#/job/${encodeURIComponent(id)}` : '#/';
    if (location.hash !== target) history.replaceState(null, '', target);
    if (!id) return;
    // Kick the four payloads off together; the pollers take over after.
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

      // Side effects on a milestone FLIP, not on every poll.
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
    } catch {
      /* media is optional; the centre column degrades to 'preparing' */
    }
  },

  async refreshSpeakers() {
    const id = get().selected;
    if (!id) return;
    try {
      const s = await api.getSpeakers(id);
      if (get().selected !== id) return;
      ensureScriptFont(s.language);
      set({ speakers: s });
    } catch {
      /* no speaker table yet — the panel stays hidden */
    }
  },

  async refreshSpec() {
    const id = get().selected;
    if (!id) return;
    try {
      const s = await api.getSpec(id);
      if (get().selected === id) set({ spec: s });
    } catch {
      /* spec is reference, not required to run the page */
    }
  },

  setSearch: (s) => set({ search: s }),

  toggleSpec() {
    const next = !get().specOpen;
    set({ specOpen: next });
    try {
      localStorage.setItem(LS_SPEC, next ? '1' : '0');
    } catch {
      /* private mode */
    }
    document.documentElement.classList.toggle('spec-open', next);
  },

  setRailOpen: (v) => set({ railOpen: v }),
  toggleTheme: () => {
    const next = get().theme === 'dark' ? 'light' : 'dark';
    applyTheme(next);
    set({ theme: next });
  },
  // The new-dub form lives ON the landing page now, not in a dialog, so
  // "open the new-job form" means "navigate home". Kept under the old
  // name so every caller — the brand bar, the job rail, the failure card,
  // the `N` shortcut — keeps working without each one learning the route.
  setNewJobOpen: (v) => {
    if (!v) return;
    if (location.hash !== '#/') location.hash = '#/';
    useStore.getState().selectJob(null);
    // The link field is the first thing to do, so put the caret in it.
    setTimeout(() => {
      const el = document.querySelector<HTMLInputElement>('#start-url');
      el?.focus();
    }, 60);
  },
  setShortcutsOpen: (v) => set({ shortcutsOpen: v }),
  setEditing: (v) => set({ editing: v }),
  setTalkingHead: (v) => set({ talkingHead: v }),
  setVideoTrack: (t) => set({ videoTrack: t }),

  /**
   * Rename a speaker. Deliberately NOT part of the staged-edit machinery.
   *
   * `POST .../speakers/{id}/label` writes a job-local `speaker_labels.json`
   * that nothing downstream reads, so a rename dirties no pipeline stage and
   * must never show up in the pending bar or put work on the GPU. It is
   * therefore written straight through, and the speaker table is patched
   * optimistically so the row does not flicker back to `SPEAKER_01` while
   * the request is in flight. On failure the optimistic value is rolled
   * back and the toast says so — a rename that silently did not stick is
   * the one outcome worse than an error.
   */
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

  revertSpeaker(speaker) {
    const next = { ...get().staged };
    delete next[speaker];
    set({ staged: next, editPlan: null });
  },

  async discardStaged() {
    const id = get().selected;
    set({ staged: {}, editPlan: null });
    if (id) {
      try {
        await api.discardEdits(id);
      } catch {
        /* nothing was persisted server-side yet */
      }
    }
  },

  /**
   * Stage the edits server-side and get the cost back. The quoted cost
   * equals the applied cost only because this never asks for a retranslate:
   * `set_voice` is deterministic and `set_tone` runs with
   * `rewrite_text=False`, so no LLM is in the loop.
   */
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

// --- helpers used in more than one place ------------------------------------

/**
 * NOT a store selector, deliberately.
 *
 * `useStore(fn)` compares by reference, and a filter that builds a new array
 * on every call makes `useSyncExternalStore` re-render on every unrelated
 * store write — which, with a 1.5 s progress tick, means the rail re-renders
 * forever. Callers select `jobs` and `search` separately and wrap this in
 * `useMemo`.
 */
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

/** Safe as a selector: returns a primitive. */
export const selectStagedCount = (s: State): number => toStagedDiffs(s.staged).length;

export const selectIsTerminal = (s: State): boolean => {
  const st = normalizeState(s.progress?.state);
  return st === 'done' || st === 'failed' || st === 'cancelled';
};
