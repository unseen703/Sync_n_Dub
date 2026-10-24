/**
 * The one import site for everything that talks to the server.
 *
 * `VITE_MOCK` picks the mode:
 *
 *   unset / "off"     nothing mocked — every call hits the real server.
 *                     This is the DEFAULT as of 2026-10-17, when
 *                     POST /api/jobs, /cancel, /edits and /edits/apply
 *                     landed. It was "writes" while those were being built.
 *   "writes"          reads are live, the mutation endpoints are mocked.
 *                     Only useful if the server predates those routes.
 *   "all"             everything mocked — no Python server needed, for
 *                     working on the UI offline.
 *
 * `MOCKED` is exported so the UI can *say* which controls are pretending.
 * A mocked Apply that reports success while nothing re-rendered is the
 * exact failure CLAUDE.md §3 is about, so every control routed here wears
 * a visible amber "demo" chip.
 *
 * Each binding is declared with the LIVE function's type, so the mock is
 * type-checked against the real contract rather than the two drifting into
 * a union that happens to call.
 */
import * as live from './client';
import * as mock from './mock';

export * from './types';
export { ApiError } from './client';

const MODE = (import.meta.env.VITE_MOCK as string | undefined) ?? 'off';
const mockAll = MODE === 'all' || MODE === '1' || MODE === 'true';
const mockWrites = mockAll || MODE === 'writes';

/** Endpoint groups currently answered by `mock.ts`. Drives the demo chips. */
export const MOCKED = {
  reads: mockAll,
  newJob: mockWrites,
  cancel: mockWrites,
  edits: mockWrites,
};

export const anyMocked = mockAll || mockWrites;

// -- read path --------------------------------------------------------------
export const listJobs: typeof live.listJobs = mockAll ? mock.listJobs : live.listJobs;
export const getProgress: typeof live.getProgress = mockAll ? mock.getProgress : live.getProgress;
export const getMedia: typeof live.getMedia = mockAll ? mock.getMedia : live.getMedia;
export const getSpeakers: typeof live.getSpeakers = mockAll ? mock.getSpeakers : live.getSpeakers;
export const getSpec: typeof live.getSpec = mockAll ? mock.getSpec : live.getSpec;
export const getOptions: typeof live.getOptions = mockAll ? mock.getOptions : live.getOptions;

// -- write path -------------------------------------------------------------
export const probeYouTube: typeof live.probeYouTube = mockWrites
  ? (url: string) => mock.probeYouTube(url)
  : live.probeYouTube;
export const createJob: typeof live.createJob = mockWrites ? mock.createJob : live.createJob;
export const uploadVideo: typeof live.uploadVideo =
  mockWrites ? mock.uploadVideo : live.uploadVideo;

// The agent routes are read/stage only — `agentSend` writes `pending.json`
// and nothing else, so there is no mock: a staged plan that did not stage
// is exactly the kind of false success the mock chips exist to flag.
export const getAgentParams = live.getAgentParams;
export const getAgentScript = live.getAgentScript;
export const agentSend = live.agentSend;
// Retraction, so it is live for the same reason `agentSend` is: a mocked
// "removed" that left the op staged would be a false success about the one
// thing this control exists to guarantee.
export const dropStagedTurn = live.dropStagedTurn;
export const cancelJob: typeof live.cancelJob = mockWrites ? mock.cancelJob : live.cancelJob;
export const getEdits: typeof live.getEdits = mockWrites ? mock.getEdits : live.getEdits;
export const stageEdits: typeof live.stageEdits = mockWrites ? mock.stageEdits : live.stageEdits;
// Hand-typed script lines. Staging only, like `stageEdits` -- `applyEdits`
// remains the single call that spends GPU time. No mock: the mock writer
// has no notion of per-segment text, and silently accepting an edit that
// goes nowhere is worse in a mocked UI than the call simply failing.
export const stageScript = live.stageScript;
export const discardEdits: typeof live.discardEdits = mockWrites
  ? mock.discardEdits
  : live.discardEdits;
export const applyEdits: typeof live.applyEdits = mockWrites ? mock.applyEdits : live.applyEdits;
export const setSpeakerLabel: typeof live.setSpeakerLabel = mockWrites
  ? mock.setSpeakerLabel
  : live.setSpeakerLabel;

// Lip-sync. NOT mocked, for the same reason the agent routes are not: this
// is the one call that commits GPU hours, and a mocked "started" that
// rendered nothing is exactly the false success the chips exist to flag.
// The readiness read stays live alongside it so the panel cannot show a
// green button against a server that would refuse.
export const getLipsyncReadiness = live.getLipsyncReadiness;
export const startLipsync = live.startLipsync;
export const getEnhanceReadiness = live.getEnhanceReadiness;
export const getEnhanceStatus = live.getEnhanceStatus;
export const startEnhance = live.startEnhance;
