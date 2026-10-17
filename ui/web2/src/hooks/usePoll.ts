import { useEffect, useRef } from 'react';

/**
 * A self-scheduling poll with the four properties this app needs and a
 * generic fetch cache does not give:
 *
 *  - the next delay is decided AFTER each tick, so a caller can back off
 *    on no-change and speed up on change;
 *  - returning `null` from `getDelay` stops the poll permanently, which is
 *    how a terminal job stops costing requests;
 *  - nothing is scheduled while `document.hidden`, and returning to the tab
 *    fires an immediate refresh rather than waiting out the interval;
 *  - a tick never overlaps the previous one. A 1.5 s interval against a
 *    server busy with a GPU render would otherwise pile up.
 *
 * `setTimeout` chaining, not `setInterval`, for exactly that last reason.
 */
export function usePoll(
  key: string | null,
  tick: () => Promise<unknown> | unknown,
  getDelay: () => number | null,
  enabled = true,
): void {
  const tickRef = useRef(tick);
  const delayRef = useRef(getDelay);
  tickRef.current = tick;
  delayRef.current = getDelay;

  useEffect(() => {
    if (!enabled || !key) return undefined;

    let cancelled = false;
    let stopped = false;
    let inFlight = false;
    let timer: number | null = null;

    const clear = () => {
      if (timer !== null) {
        window.clearTimeout(timer);
        timer = null;
      }
    };

    const schedule = (ms: number) => {
      clear();
      if (cancelled || stopped) return;
      timer = window.setTimeout(() => void run(), ms);
    };

    const run = async () => {
      if (cancelled || stopped) return;
      // Paused while hidden: nothing is scheduled, and `visibilitychange`
      // is what wakes it. A background tab costs zero requests.
      if (document.hidden) return;
      if (inFlight) {
        schedule(200);
        return;
      }
      inFlight = true;
      try {
        await tickRef.current();
      } catch {
        /* the tick owns its own error reporting */
      } finally {
        inFlight = false;
      }
      if (cancelled) return;
      const next = delayRef.current();
      if (next === null) {
        stopped = true;
        clear();
        return;
      }
      schedule(next);
    };

    const onVisibility = () => {
      if (document.hidden || cancelled || stopped) return;
      void run();
    };

    void run();
    document.addEventListener('visibilitychange', onVisibility);

    return () => {
      cancelled = true;
      clear();
      document.removeEventListener('visibilitychange', onVisibility);
    };
  }, [key, enabled]);
}
