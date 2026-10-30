/** Display-format helpers. Inputs are raw API values; outputs are short strings for the UI. */
import type { JobRow } from '../api/types';

export function fmtBytes(n: number | null | undefined): string {
  if (n == null || !isFinite(n)) return '—';
  if (n < 1024) return `${n} B`;
  const units = ['KB', 'MB', 'GB', 'TB'];
  let v = n / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v.toFixed(v >= 100 || i === 0 ? 0 : 1)} ${units[i]}`;
}

export function fmtClock(s: number | null | undefined): string {
  if (s == null || !isFinite(s)) return '—';
  const t = Math.max(0, Math.round(s));
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const sec = t % 60;
  const pad = (n: number) => String(n).padStart(2, '0');
  return h ? `${h}:${pad(m)}:${pad(sec)}` : `${m}:${pad(sec)}`;
}

export function fmtDuration(s: number | null | undefined): string {
  if (s == null || !isFinite(s) || s <= 0) return '—';
  if (s < 90) return `${Math.round(s)} seconds`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m} minute${m === 1 ? '' : 's'}`;
  const h = Math.floor(m / 60);
  const rm = m % 60;
  return rm ? `${h} h ${rm} min` : `${h} hour${h === 1 ? '' : 's'}`;
}

export function fmtRelative(iso: string | null | undefined): string {
  if (!iso) return '—';
  const t = Date.parse(iso.endsWith('Z') || /[+-]\d\d:?\d\d$/.test(iso) ? iso : `${iso}Z`);
  if (Number.isNaN(t)) return iso;
  const d = (Date.now() - t) / 1000;
  if (d < 90) return 'just now';
  if (d < 3600) return `${Math.floor(d / 60)} min ago`;
  if (d < 86400) {
    const h = Math.floor(d / 3600);
    return `${h} hour${h === 1 ? '' : 's'} ago`;
  }
  const days = Math.floor(d / 86400);
  if (days < 30) return `${days} day${days === 1 ? '' : 's'} ago`;
  return new Date(t).toISOString().slice(0, 10);
}

export function fmtElapsed(iso: string | null | undefined, now = Date.now()): string {
  if (!iso) return '—';
  const t = Date.parse(iso.endsWith('Z') ? iso : `${iso}Z`);
  if (Number.isNaN(t)) return '—';
  return fmtClock(Math.max(0, (now - t) / 1000));
}

export function jobTitle(
  job: Pick<JobRow, 'title' | 'display_name' | 'slug' | 'job_id'>,
): { text: string; degraded: boolean } {

  if (job.title) return { text: job.title, degraded: false };
  if (job.display_name) return { text: job.display_name, degraded: true };
  if (job.slug && job.slug !== job.job_id) return { text: job.slug, degraded: true };
  return { text: job.job_id, degraded: true };
}

export function jobWhen(timestamp: string): string {
  const d = new Date(timestamp.replace(' ', 'T'));
  if (Number.isNaN(d.getTime())) return timestamp;
  return d.toLocaleString(undefined, {
    day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit',
  });
}

export function normalizeState(raw: string | null | undefined): string {
  const s = (raw ?? '').toLowerCase();
  if (s.startsWith('failed')) return 'failed';
  if (s === 'ok' || s === 'complete' || s === 'completed') return 'done';
  if (s.startsWith('running')) return 'running';
  if (['queued', 'done', 'failed', 'cancelled', 'unknown'].includes(s)) return s;
  return s || 'unknown';
}

export function fmtRationale(r: unknown): string | null {
  if (r == null) return null;
  if (typeof r === 'string') return r.trim() || null;
  if (typeof r !== 'object') return String(r);
  const parts: string[] = [];
  for (const [k, v] of Object.entries(r as Record<string, unknown>)) {
    const name = k.replace(/_/g, ' ');
    if (typeof v === 'number') {
      parts.push(`${name} ${k.includes('match') ? `${Math.round(v * 100)}%` : v.toFixed(2)}`);
    } else if (v != null) {
      parts.push(`${name} ${String(v)}`);
    }
  }
  return parts.length ? `Matched on ${parts.join(', ')}.` : null;
}

export function genderWord(g: string | null | undefined): string | null {
  if (g === 'male') return 'Male';
  if (g === 'female') return 'Female';
  return null;
}

export function toneLabel(
  options: Array<{ value: string; label: string }> | null | undefined,
  value: string | null | undefined,
): string {
  if (!value) return 'the engine default';
  return options?.find((o) => o.value === value)?.label ?? value;
}
