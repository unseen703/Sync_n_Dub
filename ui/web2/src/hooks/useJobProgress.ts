import { useRef } from 'react';
import { usePoll } from './usePoll';
import { useStore } from '../store/useStore';
import type { Progress } from '../api/types';

const FAST_MS = 1500;
const SLOW_MS = 4000;
const UNCHANGED_BEFORE_BACKOFF = 10;

const TERMINAL = new Set(['done', 'failed', 'cancelled']);

/** Everything that would make the screen look different. */
function signature(p: Progress | null): string {
  if (!p) return '';
  const m = p.milestones ?? ({} as Progress['milestones']);
  return [
    p.state,
    p.current ?? '',
    p.n_done,
    p.n_total,
    p.pct,
    p.updated_at ?? '',
    p.error ?? '',
    m.video_ready,
    m.diarization_done,
    m.speakers_ready,
    m.mix_done,
    m.lipsync_ready,
  ].join('|');
}

/**
 * The only consumer of the polling mechanism for `/progress`.
 *
 * Swapping this for SSE later is a change to this file and nothing else —
 * which is the reason the cadence lives in a hook rather than in the store.
 *
 * Cadence: 1.5 s, backing off to 4 s after ten unchanged polls, resetting
 * the moment anything changes. On a terminal state it takes ONE more full
 * sweep (media, speakers, spec, job list) and then stops for good: a done
 * job must not keep a request in flight for the rest of the session.
 */
export function useJobProgress(jobId: string | null): void {
  const unchanged = useRef(0);
  const lastSig = useRef('');
  const sweptFor = useRef<string | null>(null);

  const refreshProgress = useStore((s) => s.refreshProgress);

  usePoll(
    jobId,
    async () => {
      const p = await refreshProgress();
      const sig = signature(p);
      if (sig === lastSig.current) unchanged.current += 1;
      else {
        unchanged.current = 0;
        lastSig.current = sig;
      }
    },
    () => {
      const s = useStore.getState();
      const state = s.progress?.state ?? 'unknown';
      if (TERMINAL.has(state)) {
        // One final sweep, once per job, then stop.
        if (sweptFor.current !== s.selected) {
          sweptFor.current = s.selected;
          void s.refreshMedia();
          void s.refreshSpeakers();
          void s.refreshSpec();
          void s.refreshJobs();
          return 1200; // let the sweep land, then the next call stops us
        }
        return null;
      }
      sweptFor.current = null;
      return unchanged.current >= UNCHANGED_BEFORE_BACKOFF ? SLOW_MS : FAST_MS;
    },
    Boolean(jobId),
  );
}

/**
 * The rail. 10 s while anything is running, 45 s when everything is idle —
 * a list of finished jobs changes only when the user creates one, and that
 * path refreshes explicitly.
 */
export function useJobsPoll(): void {
  const refreshJobs = useStore((s) => s.refreshJobs);
  usePoll(
    'jobs',
    refreshJobs,
    () => {
      const jobs = useStore.getState().jobs;
      const busy = jobs.some((j) => {
        const st = (j.state ?? '').toLowerCase();
        return st.startsWith('running') || st === 'queued';
      });
      return busy ? 10_000 : 45_000;
    },
    true,
  );
}
