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
  gender?: string | null;
  fromVoice?: string | null;
  fromTone?: string | null;
  fromGender?: string | null;
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
  /**
   * How much of the machinery the screen admits to.
   *
   * `user` is the default and hides the parts that only mean something to
   * someone who knows the pipeline — raw run config, tracebacks, job ids,
   * model names. `admin` shows them.
   *
   * This is PRESENTATION ONLY. It is not a permission: there is no auth in
   * this app (removed 2026-10-25), every route is open to anyone who can
   * reach the port, and a user can flip this switch themselves. Nothing
   * server-side reads it, and nothing here should start doing so — a check
   * a client can turn off is not a check.
   */
  mode: 'user' | 'admin';
  toggleMode: () => void;
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
  /**
   * What the edit agent can change, and what is staged over it.
   *
   * Lives in the store rather than inside the agent panel because the CAST
   * TABLE needs it too: a gender/voice/tone change can arrive from the
   * picker on a row (local, `staged` below) or from a sentence in the chat
   * box (server-side, `pending.json`), and a row that reads only the first
   * shows nothing at all for the second. One snapshot, one place, so the
   * two renderings of a staged edit cannot disagree by a poll — which is
   * the reason `/agent/params` is a single endpoint in the first place.
   */
  agentParams: AgentParams | null;
  refreshAgentParams: () => Promise<AgentParams | null>;
  /** `#/talking-head` — the still-portrait page. */
  talkingHead: boolean;
  setTalkingHead: (v: boolean) => void;
  videoTrack: 'original' | 'lipsync' | null;
  /**
   * The active player's current time, in seconds. Pushed from the `<video>`'s
   * `timeupdate` (~4×/sec) so the follow-along cue panel can light up the line
   * being spoken. Shared through the store because the player and the cue
   * panel are in different columns of the edit screen.
   */
  videoTime: number;
  /**
   * A one-shot seek request: a clicked cue asks the player to jump to its
   * start. The `nonce` is what makes clicking the SAME cue twice fire the
   * effect again — a bare number would compare equal and be ignored.
   */
  videoSeek: { t: number; nonce: number } | null;
  /**
   * What the edit workspace aims its actions at. A clicked cue or speaker row
   * sets one of these; the quick actions then fold the selection into their
   * phrasing ("cue 5: …", "SPEAKER_00: …") so no timestamp or id is typed.
   *
   * Mutually exclusive on purpose — one context at a time keeps "which actions
   * are relevant" unambiguous. `setSelectedCue` clears the speaker and vice
   * versa.
   */
  selectedCue: string | null;
  selectedSpeaker: string | null;
  toasts: Toast[];

  // -- edit history --------------------------------------------------------
  editHistory: EditHistory | null;

  // -- staged voice/tone edits --------------------------------------------
  staged: Record<string, StagedChange>;
  // Per-speaker cast mode for the post-MT gate. `select` pins the chosen
  // gender/voice; `autodetect` (the default) lets the pipeline predict. Only
  // read when `progress.milestones.awaiting_cast` is up.
  castMode: Record<string, 'autodetect' | 'select'>;
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
 * `voice` / `tone` / `gender` off each item and builds one `EditOp` per
 * field present.
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
    if (c.gender != null && c.gender !== c.fromGender) {
      edit.gender = c.gender;
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
  field: 'voice' | 'tone' | 'gender';
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
    if (c.gender != null && c.gender !== c.fromGender) {
      out.push({ speaker, field: 'gender', from: c.fromGender ?? null, to: c.gender });
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

const LS_MODE = 'ds.mode';

/**
 * The saved view mode. Defaults to `user`, so a fresh browser gets the
 * uncluttered screen and the technical detail is opted into rather than
 * out of. Same try/catch as `readTheme` and for the same reason.
 */
function readMode(): 'user' | 'admin' {
  try {
    return localStorage.getItem(LS_MODE) === 'admin' ? 'admin' : 'user';
  } catch {
    return 'user';
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
  mode: readMode(),
  shortcutsOpen: false,
  agentParams: null,
  talkingHead: false,
  videoTrack: null,
  videoTime: 0,
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
      // Back to following the default — the next job gets its own
      // lip-synced-first treatment rather than inheriting this one's pin.
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
      // The MT gate. `refreshSpeakers` matters as much as the toast: the
      // table has to re-fetch or it shows the pre-gate rows, without the
      // gender stage 6b just predicted for them.
      if (p.milestones?.awaiting_cast && !before?.milestones?.awaiting_cast) {
        void get().refreshSpeakers();
        get().pushToast({
          kind: 'info',
          title: 'Translation complete — pick the voices to carry on',
          body: 'Choose a voice for each speaker; nothing is spoken until you do. '
              + 'Worth checking the translated lines while you are there.',
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

  /**
   * Reload the agent's view of the job.
   *
   * Swallows its error on purpose. `/agent/params` 409s on a run that never
   * reached MT — there is no merged transcript, so there is nothing to edit
   * — and that is a normal state for a job still in its first stages, not a
   * failure worth a banner. The panel that needs the data shows its own
   * message; the cast table simply falls back to local staging.
   */
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
  toggleMode: () => {
    const next = get().mode === 'admin' ? 'user' : 'admin';
    set({ mode: next });
    try {
      localStorage.setItem(LS_MODE, next);
    } catch {
      /* preference only */
    }
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
  setTalkingHead: (v) => set({ talkingHead: v }),
  setVideoTrack: (t) => set({ videoTrack: t }),
  setVideoTime: (t) => set({ videoTime: t }),
  requestSeek: (t) => set({ videoSeek: { t, nonce: Date.now() } }),
  setSelectedCue: (id) => set({ selectedCue: id, selectedSpeaker: null }),
  setSelectedSpeaker: (id) => set({ selectedSpeaker: id, selectedCue: null }),

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

  /**
   * Submit the cast gate: for a job stopped after MT, run the post-MT stages
   * with the chosen gender/voice. A speaker on `select` sends its staged (or
   * current) gender and voice as an override; `autodetect` sends none, so the
   * pipeline predicts it. Lip-sync stays a separate step.
   */
  async cast() {
    const id = get().selected;
    const table = get().speakers;
    if (!id || !table) return;
    const staged = get().staged;
    const modes = get().castMode;
    const payload: Record<string, CastSelection> = {};
    for (const sp of table.speakers) {
      const mode = modes[sp.id] ?? 'autodetect';
      if (mode === 'select') {
        const st = staged[sp.id];
        payload[sp.id] = {
          mode,
          gender: st?.gender !== undefined ? st.gender : sp.gender,
          voice: st?.voice !== undefined ? st.voice : sp.voice,
        };
      } else {
        payload[sp.id] = { mode: 'autodetect' };
      }
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
