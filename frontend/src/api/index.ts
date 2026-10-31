/** Selects live or mock API calls. Inputs match `client.ts`; outputs keep the same typed responses. */
import * as live from './client';
import * as mock from './mock';

export * from './types';
export { ApiError } from './client';

const MODE = (import.meta.env.VITE_MOCK as string | undefined) ?? 'off';
const mockAll = MODE === 'all' || MODE === '1' || MODE === 'true';
const mockWrites = mockAll || MODE === 'writes';

export const MOCKED = {
  reads: mockAll,
  newJob: mockWrites,
  cancel: mockWrites,
  edits: mockWrites,
};

export const anyMocked = mockAll || mockWrites;

export const listJobs: typeof live.listJobs = mockAll ? mock.listJobs : live.listJobs;
export const getProgress: typeof live.getProgress = mockAll ? mock.getProgress : live.getProgress;
export const getMedia: typeof live.getMedia = mockAll ? mock.getMedia : live.getMedia;
export const getSpeakers: typeof live.getSpeakers = mockAll ? mock.getSpeakers : live.getSpeakers;
export const getSpec: typeof live.getSpec = mockAll ? mock.getSpec : live.getSpec;
export const getOptions: typeof live.getOptions = mockAll ? mock.getOptions : live.getOptions;

export const probeYouTube: typeof live.probeYouTube = mockWrites
  ? (url: string) => mock.probeYouTube(url)
  : live.probeYouTube;
export const createJob: typeof live.createJob = mockWrites ? mock.createJob : live.createJob;
export const uploadVideo: typeof live.uploadVideo =
  mockWrites ? mock.uploadVideo : live.uploadVideo;

export const getEditHistory = live.getEditHistory;

export const getAgentParams = live.getAgentParams;
export const getAgentScript = live.getAgentScript;
export const agentSend = live.agentSend;

export const dropStagedTurn = live.dropStagedTurn;

export const revertEdits = live.revertEdits;
export const cancelJob: typeof live.cancelJob = mockWrites ? mock.cancelJob : live.cancelJob;
export const getEdits: typeof live.getEdits = mockWrites ? mock.getEdits : live.getEdits;
export const stageEdits: typeof live.stageEdits = mockWrites ? mock.stageEdits : live.stageEdits;

export const stageScript = live.stageScript;

export const muteCue = live.muteCue;
export const discardEdits: typeof live.discardEdits = mockWrites
  ? mock.discardEdits
  : live.discardEdits;
export const applyEdits: typeof live.applyEdits = mockWrites ? mock.applyEdits : live.applyEdits;
export const rerunDub = live.rerunDub;
export const setSpeakerLabel: typeof live.setSpeakerLabel = mockWrites
  ? mock.setSpeakerLabel
  : live.setSpeakerLabel;

export const getLipsyncReadiness = live.getLipsyncReadiness;
export const startLipsync = live.startLipsync;

export const castJob = live.castJob;
export const getEnhanceReadiness = live.getEnhanceReadiness;
export const getEnhanceStatus = live.getEnhanceStatus;
export const startEnhance = live.startEnhance;

export const getReviewVideo = live.getReviewVideo;
export const makeReviewVideo = live.makeReviewVideo;
