import { useEffect, useRef } from 'react';

/** Runs an async poller. Inputs supply work and the next delay; it returns nothing and cleans up on unmount. */
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

      if (document.hidden) return;
      if (inFlight) {
        schedule(200);
        return;
      }
      inFlight = true;
      try {
        await tickRef.current();
      } catch {} finally {
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
