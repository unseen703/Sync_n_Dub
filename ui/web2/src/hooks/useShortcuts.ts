import { useEffect } from 'react';
import { useStore } from '../store/useStore';

/**
 * App-wide keyboard shortcuts.
 *
 * Three rules keep them from fighting the rest of the UI:
 *
 *  1. Nothing fires while focus is in a text field, a textarea or anything
 *     `contenteditable`. Typing "n" into the speaker-name box must type an
 *     "n", not open the new-dub dialog — which is exactly the bug a naive
 *     `window.keydown` listener ships with.
 *  2. Nothing fires with a modifier held, so browser and OS shortcuts are
 *     left alone.
 *  3. Escape is handled by Radix inside dialogs and popovers; here it only
 *     closes the two drawers this app owns.
 *
 * The list is duplicated once, in `ShortcutsDialog`, because a shortcut
 * nobody can discover is a shortcut nobody has.
 */
export const SHORTCUTS: Array<{ keys: string[]; what: string }> = [
  { keys: ['N'], what: 'Start a new dub' },
  { keys: ['/'], what: 'Search your dubs' },
  { keys: ['J'], what: 'Next dub in the list' },
  { keys: ['K'], what: 'Previous dub in the list' },
  { keys: ['I'], what: 'Show or hide the job settings panel' },
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
          // The rail may be mounting; focus on the next frame either way.
          requestAnimationFrame(() => {
            document.getElementById('job-search')?.focus();
          });
          break;
        case '?':
          e.preventDefault();
          s.setShortcutsOpen(true);
          break;
        case 'i':
        case 'I':
          if (s.selected) {
            e.preventDefault();
            s.toggleSpec();
          }
          break;
        case 'o':
        case 'O': {
          // Only when there is something to switch TO. Toggling to a track
          // that does not exist would land on a dead player.
          const hasLip = Boolean(s.media?.video.lipsync && s.progress?.milestones?.lipsync_ready);
          if (!hasLip) return;
          e.preventDefault();
          // `videoTrack` is null until someone chooses, and the default it
          // stands for is lip-synced once a render exists — which `hasLip`
          // has just established. So null toggles to the original, same as
          // an explicit 'lipsync' would.
          s.setVideoTrack(s.videoTrack === 'original' ? 'lipsync' : 'original');
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
