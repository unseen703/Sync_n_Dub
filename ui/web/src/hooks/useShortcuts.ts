import { useEffect } from 'react';
import { useStore } from '../store/useStore';
import type { VideoTrack } from '../api/types';

/** Lists and registers application keyboard shortcuts. It updates focus or navigation state and returns nothing. */
export const SHORTCUTS: Array<{ keys: string[]; what: string }> = [
  { keys: ['N'], what: 'Start a new dub' },
  { keys: ['/'], what: 'Search your dubs' },
  { keys: ['J'], what: 'Next dub in the list' },
  { keys: ['K'], what: 'Previous dub in the list' },
  { keys: ['O'], what: 'Switch between the original and the lip-synced video' },
  { keys: ['?'], what: 'This list' },
  { keys: ['Esc'], what: 'Close whatever is open' },
];

function isTyping(el: EventTarget | null): boolean {
  const n = el as HTMLElement | null;
  if (!n || !n.tagName) return false;
  const tag = n.tagName.toLowerCase();
  return (
    tag === 'input'
    || tag === 'textarea'
    || tag === 'select'
    || n.isContentEditable === true
  );
}

export function useShortcuts(): void {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const s = useStore.getState();

      if (e.key === 'Escape') {
        if (s.railOpen) s.setRailOpen(false);
        else if (s.shortcutsOpen) s.setShortcutsOpen(false);
        return;
      }

      if (isTyping(e.target)) return;

      switch (e.key) {
        case 'n':
        case 'N':
          e.preventDefault();
          s.setNewJobOpen(true);
          break;
        case '/':
          e.preventDefault();
          s.setRailOpen(true);

          requestAnimationFrame(() => {
            document.getElementById('job-search')?.focus();
          });
          break;
        case '?':
          e.preventDefault();
          s.setShortcutsOpen(true);
          break;
        case 'o':
        case 'O': {

          const hasLip = Boolean(s.media?.video.lipsync && s.progress?.milestones?.lipsync_ready);
          const ring: VideoTrack[] = [
            ...(s.media?.video.enhanced ? (['enhanced'] as const) : []),
            ...(hasLip ? (['lipsync'] as const) : []),
            'original',
          ];
          if (ring.length < 2) return;
          e.preventDefault();

          const at = s.videoTrack ? ring.indexOf(s.videoTrack) : 0;
          s.setVideoTrack(ring[(Math.max(0, at) + 1) % ring.length]);
          break;
        }
        case 'j':
        case 'J':
        case 'k':
        case 'K': {
          const list = s.jobs;
          if (!list.length) return;
          e.preventDefault();
          const i = list.findIndex((j) => j.job_id === s.selected);
          const step = e.key.toLowerCase() === 'j' ? 1 : -1;
          const next = i < 0 ? 0 : Math.max(0, Math.min(list.length - 1, i + step));
          s.selectJob(list[next].job_id);
          break;
        }
        default:
          break;
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
}
