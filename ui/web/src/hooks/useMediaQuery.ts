import { useEffect, useState } from 'react';

export function useMediaQuery(query: string): boolean {
  const [match, setMatch] = useState(() =>
    typeof window === 'undefined' ? false : window.matchMedia(query).matches,
  );
  useEffect(() => {
    const mq = window.matchMedia(query);
    const on = () => setMatch(mq.matches);
    on();
    mq.addEventListener('change', on);
    return () => mq.removeEventListener('change', on);
  }, [query]);
  return match;
}

/** Below 1024 px the rail becomes an off-canvas drawer. */
export const useIsNarrow = () => !useMediaQuery('(min-width: 1024px)');

/** At 1536 px and up the spec panel pushes instead of overlaying. */
export const useIsWide = () => useMediaQuery('(min-width: 1536px)');
