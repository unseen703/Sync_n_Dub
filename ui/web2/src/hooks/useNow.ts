import { useEffect, useState } from 'react';

/**
 * A 1 s clock, used only for the "14:22 elapsed" readout on a running job.
 *
 * `enabled` is not a nicety: left on, this re-renders the whole header once
 * a second for a finished job that will never change again.
 */
export function useNow(enabled: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!enabled) return undefined;
    const t = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(t);
  }, [enabled]);
  return now;
}
