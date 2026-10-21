import { Dialog } from './Dialog';
import { Kbd } from '../layout/Landing';
import { SHORTCUTS } from '../../hooks/useShortcuts';

/** The discoverable half of `useShortcuts`. Loaded on demand. */
export function ShortcutsDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (v: boolean) => void;
}) {
  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      width="max-w-md"
      title="Keyboard shortcuts"
      description="These work anywhere except while you are typing in a box."
    >
      <ul className="divide-y divide-line">
        {SHORTCUTS.map((s) => (
          <li key={s.keys.join('+')} className="flex items-center gap-3 py-2.5">
            <span className="min-w-0 flex-1 text-[13px] text-ink2">{s.what}</span>
            <span className="flex shrink-0 gap-1">
              {s.keys.map((k) => (
                <Kbd key={k}>{k}</Kbd>
              ))}
            </span>
          </li>
        ))}
      </ul>
    </Dialog>
  );
}
