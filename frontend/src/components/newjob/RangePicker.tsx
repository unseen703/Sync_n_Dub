import { useId } from 'react';
import { cn } from '../../lib/cn';
import { fmtClock } from '../../lib/format';

function parseClock(s: string): number | null {
  const parts = s.trim().split(':').map((p) => p.trim());
  if (parts.some((p) => p !== '' && !/^\d+$/.test(p))) return null;
  const nums = parts.map((p) => Number(p || 0));
  if (nums.some((n) => Number.isNaN(n))) return null;
  if (nums.length === 1) return nums[0];
  if (nums.length === 2) return nums[0] * 60 + nums[1];
  if (nums.length === 3) return nums[0] * 3600 + nums[1] * 60 + nums[2];
  return null;
}

const TRACK =
  'absolute inset-x-0 top-2 h-4 w-full appearance-none bg-transparent pointer-events-none';

const THUMB =
  '[&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:w-3.5 ' +
  '[&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full ' +
  '[&::-webkit-slider-thumb]:pointer-events-auto [&::-webkit-slider-thumb]:cursor-grab ' +
  '[&::-webkit-slider-thumb]:shadow [&::-webkit-slider-thumb]:ring-2 ' +
  '[&::-webkit-slider-thumb]:ring-canvas ' +
  '[&::-moz-range-thumb]:h-3.5 [&::-moz-range-thumb]:w-3.5 ' +
  '[&::-moz-range-thumb]:appearance-none [&::-moz-range-thumb]:rounded-full ' +
  '[&::-moz-range-thumb]:border-0 [&::-moz-range-thumb]:pointer-events-auto ' +
  '[&::-moz-range-thumb]:cursor-grab';

export function RangePicker({
  duration,
  whole,
  onWholeChange,
  start,
  end,
  onChange,
  capS,
}: {
  duration: number;
  whole: boolean;
  onWholeChange: (v: boolean) => void;
  start: number;
  end: number;
  onChange: (start: number, end: number) => void;
  capS: number;
}) {
  const id = useId();
  const span = Math.max(0, end - start);
  const over = !whole && span > capS;
  const wholeOver = whole && duration > capS;

  const pctL = duration ? (start / duration) * 100 : 0;
  const pctW = duration ? (span / duration) * 100 : 0;

  const preset = (secs: number | null) => {
    if (secs === null) {
      onWholeChange(true);
      return;
    }
    onWholeChange(false);
    onChange(0, Math.min(secs, duration));
  };

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-1.5">
        {[
          ['First minute', 60],
          ['First 5 min', 300],
          ['First 20 min', 1200],
          ['Whole video', null],
        ].map(([label, secs]) => {
          const isActive =
            secs === null ? whole : !whole && start === 0 && end === Math.min(secs as number, duration);
          return (
            <button
              key={label as string}
              type="button"
              onClick={() => preset(secs as number | null)}
              className={cn(
                'rounded-lg border px-2.5 py-1 text-[11.5px] transition-colors',
                isActive
                  ? 'border-brand-ink/50 bg-brand/20 text-ink'
                  : 'border-edge/60 text-ink3 hover:border-edge hover:text-ink',
              )}
            >
              {label}
            </button>
          );
        })}
      </div>

      {!whole ? (
        <>
          <div className="flex items-center gap-2">
            <TimeField
              id={`${id}-start`}
              label="From"
              value={start}
              max={duration}
              onCommit={(v) => onChange(Math.min(v, end - 1), end)}
            />
            <span className="pt-4 text-ink4">–</span>
            <TimeField
              id={`${id}-end`}
              label="To"
              value={end}
              max={duration}
              onCommit={(v) => onChange(start, Math.max(v, start + 1))}
            />
            <div className="ml-auto pt-4 text-right">
              <div className="text-[11.5px] text-ink3">Length</div>
              <div
                className={cn(
                  'font-mono text-[12px]',
                  over ? 'text-danger' : 'text-ink',
                )}
              >
                {fmtClock(span)}
              </div>
            </div>
          </div>

          <div className="relative h-8">
            <div className="absolute inset-x-0 top-3.5 h-1 rounded-full bg-ink/[0.1]" />
            <div
              className="absolute top-3.5 h-1 rounded-full bg-gradient-to-r from-brand to-aqua"
              style={{ left: `${pctL}%`, width: `${pctW}%` }}
            />
            <input
              type="range"
              aria-label="Start of the part to dub"
              min={0}
              max={duration}
              step={1}
              value={start}
              onChange={(e) => onChange(Math.min(Number(e.target.value), end - 1), end)}
              className={cn(TRACK, THUMB,
                "[&::-webkit-slider-thumb]:bg-ink [&::-moz-range-thumb]:bg-ink")}
            />
            <input
              type="range"
              aria-label="End of the part to dub"
              min={0}
              max={duration}
              step={1}
              value={end}
              onChange={(e) => onChange(start, Math.max(Number(e.target.value), start + 1))}
              className={cn(TRACK, THUMB,
                "[&::-webkit-slider-thumb]:bg-aqua [&::-moz-range-thumb]:bg-aqua")}
            />
          </div>
        </>
      ) : (
        <p className="text-[12px] text-ink3">
          The whole video — {fmtClock(duration)}.
        </p>
      )}

      {over || wholeOver ? (
        <p className="rounded-lg border border-danger/45 bg-danger/[0.07] px-2.5 py-2 text-[11.5px] leading-relaxed text-ink2">
          Lip-sync is capped at {fmtClock(capS)} of video. It runs at 3–21× the clip length on
          this machine&rsquo;s GPUs, so {fmtClock(whole ? duration : span)} would be hours.
          {wholeOver
            ? ' Pick a shorter part, or turn lip-sync off to dub the whole thing as audio only.'
            : ' Shorten the range, or turn lip-sync off.'}
        </p>
      ) : null}
    </div>
  );
}

function TimeField({
  id,
  label,
  value,
  max,
  onCommit,
}: {
  id: string;
  label: string;
  value: number;
  max: number;
  onCommit: (v: number) => void;
}) {
  return (
    <div>
      <label htmlFor={id} className="block text-[11.5px] text-ink3">
        {label}
      </label>
      <input
        id={id}
        defaultValue={fmtClock(value)}
        key={value}
        onBlur={(e) => {
          const v = parseClock(e.target.value);
          if (v === null) {
            e.target.value = fmtClock(value);
            return;
          }
          onCommit(Math.max(0, Math.min(v, max)));
        }}
        className="h-8 w-20 rounded-lg border border-edge/70 bg-canvas/60 px-2 text-center
                   font-mono text-[12px] text-ink focus:border-brand-ink focus:outline-none"
      />
    </div>
  );
}
