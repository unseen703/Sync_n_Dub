import { Dialog } from '../ui/Dialog';
import { Button } from '../ui/Button';
import { Eyebrow } from '../ui/Card';
import { useStore } from '../../store/useStore';

export function CastConfirmDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (v: boolean) => void;
}) {
  const table = useStore((s) => s.speakers);
  const staged = useStore((s) => s.staged);
  const cast = useStore((s) => s.cast);
  const applying = useStore((s) => s.applying);

  const voiceLabel = (v: string | null | undefined) =>
    (v
      ? (table?.voice_options?.find((o) => o.voice === v)?.display
        ?? v.split('-').pop() ?? v)
      : '—');

  const rows = (table?.speakers ?? []).map((sp) => {
    const voice = staged[sp.id]?.voice !== undefined ? staged[sp.id].voice : sp.voice;
    return { id: sp.id, label: sp.label ?? sp.id, voice };
  });

  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      width="max-w-xl"
      title="Cast these voices and start the dub?"
      description=" Confirming starts the speech, mix and timing stages with the voices below — lip-sync stays a separate step after."
      footer={
        <>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            variant="primary"
            busy={applying}
            onClick={async () => {
              await cast();
              onOpenChange(false);
            }}
          >
            Cast &amp; continue
          </Button>
        </>
      }
    >
      <section>
        <Eyebrow className="mb-2">Voices</Eyebrow>
        <ul className="space-y-1.5">
          {rows.map((r) => (
            <li key={r.id} className="flex items-center gap-2 text-[12.5px]">
              <span className="min-w-0 truncate font-medium text-ink">{r.label}</span>
              <span className="text-ink4" aria-hidden="true">→</span>
              <span className="font-semibold text-ink">
                {voiceLabel(r.voice)}
              </span>
            </li>
          ))}
        </ul>
      </section>
    </Dialog>
  );
}
