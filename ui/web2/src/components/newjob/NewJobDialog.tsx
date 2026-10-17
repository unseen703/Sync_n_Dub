import { useEffect, useState } from 'react';
import { Dialog } from '../ui/Dialog';
import { Button, DemoChip } from '../ui/Button';
import { Select } from '../ui/Select';
import { Eyebrow } from '../ui/Card';
import { IconAlert, IconCheck, IconSpinner } from '../ui/Icons';
import { RangePicker } from './RangePicker';
import { useStore } from '../../store/useStore';
import * as api from '../../api';
import { MOCKED } from '../../api';
import { ensureScriptFont, langClass } from '../../lib/lang';
import { fmtClock } from '../../lib/format';
import type { Options, ProbeResult } from '../../api/types';

/**
 * Paste a link, pick a language, pick which part of the video to dub.
 *
 * The probe is what makes the rest of the form honest — it is the only
 * source of the duration that bounds the range picker, and it is also how
 * the title gets captured at all (`download_youtube` sets
 * `filename_template="%(id)s.%(ext)s"` and throws the metadata away, and
 * with a pre-cut window the pipeline never calls it).
 */
export function NewJobDialog() {
  const open = useStore((s) => s.newJobOpen);
  const setOpen = useStore((s) => s.setNewJobOpen);
  const selectJob = useStore((s) => s.selectJob);
  const refreshJobs = useStore((s) => s.refreshJobs);
  const pushToast = useStore((s) => s.pushToast);

  const [options, setOptions] = useState<Options | null>(null);
  const [url, setUrl] = useState('');
  const [probe, setProbe] = useState<ProbeResult | null>(null);
  const [probing, setProbing] = useState(false);
  const [probeError, setProbeError] = useState<string | null>(null);

  const [language, setLanguage] = useState('hi');
  const [lipsync, setLipsync] = useState('auto');
  const [whole, setWhole] = useState(false);
  const [start, setStart] = useState(0);
  const [end, setEnd] = useState(60);
  const [creating, setCreating] = useState(false);

  // No engine picker: every job runs on the install's TTS engine (Bodhan
  // indic-speak unless `DUBBING_UI_TTS_MODEL` says otherwise). The language
  // list still comes from the server rather than a constant here, because a
  // language is offered only when that engine can actually speak it — the
  // alternative is a job that runs for minutes and then cannot synthesise a
  // word. `GET /api/options` with no `?tts=` answers for whichever engine the
  // server will really use, so the two cannot disagree.
  useEffect(() => {
    if (!open) return;
    let alive = true;
    void api
      .getOptions()
      .then((o) => {
        if (!alive) return;
        setOptions(o);
        if (!o.languages.some((l) => l.code === language)) {
          setLanguage(o.languages[0]?.code ?? 'hi');
        }
      })
      .catch(() => undefined);
    return () => {
      alive = false;
    };
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    ensureScriptFont(language);
  }, [language]);

  const runProbe = async () => {
    if (!url.trim()) return;
    setProbing(true);
    setProbeError(null);
    try {
      const r = await api.probeYouTube(url.trim());
      setProbe(r);
      setEnd(Math.min(60, r.duration_s ?? 60));
      setStart(0);
      setWhole(false);
    } catch (e) {
      setProbe(null);
      setProbeError((e as Error).message);
    } finally {
      setProbing(false);
    }
  };

  const cap = options?.limits.max_lipsync_s ?? 1200;
  const dur = probe?.duration_s ?? 0;
  const span = whole ? dur : end - start;
  const tooLong = lipsync !== 'off' && span > cap;
  const canCreate = Boolean(probe) && !creating && !tooLong;

  const reset = () => {
    setUrl('');
    setProbe(null);
    setProbeError(null);
    setCreating(false);
  };

  const create = async () => {
    if (!probe) return;
    setCreating(true);
    try {
      const res = await api.createJob({
        url: probe.webpage_url ?? url.trim(),
        target_language: language,
        window: whole ? null : { start_s: start, end_s: end },
        lipsync_model: lipsync,
      });
      await refreshJobs();
      selectJob(res.job_id);
      setOpen(false);
      reset();
      pushToast({
        kind: MOCKED.newJob ? 'warn' : 'success',
        title: MOCKED.newJob
          ? 'Nothing was queued — POST /api/jobs is not built yet'
          : res.position
            ? `Queued at position ${res.position}`
            : 'Job started',
        body: MOCKED.newJob
          ? 'The form and its validation are real; the submit endpoint is still a stub in src/api/mock.ts.'
          : res.estimate?.text,
      });
    } catch (e) {
      pushToast({ kind: 'error', title: 'Could not start the job', body: (e as Error).message });
    } finally {
      setCreating(false);
    }
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(v) => {
        setOpen(v);
        if (!v) reset();
      }}
      width="max-w-2xl"
      title={
        <span className="flex items-center gap-2">
          New dub
          {MOCKED.newJob ? <DemoChip /> : null}
        </span>
      }
      description="Paste a YouTube link, choose the language, and pick how much of it to dub."
      footer={
        <>
          <Button variant="ghost" onClick={() => setOpen(false)}>
            Cancel
          </Button>
          <Button variant="primary" busy={creating} disabled={!canCreate} onClick={() => void create()}>
            Start dubbing
            {probe ? ` · ${fmtClock(span)}` : ''}
          </Button>
        </>
      }
    >
      <div className="space-y-5">
        <section>
          <Eyebrow className="mb-2">YouTube link</Eyebrow>
          <div className="flex gap-2">
            <input
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') void runProbe();
              }}
              onBlur={() => {
                if (url.trim() && !probe) void runProbe();
              }}
              placeholder="https://www.youtube.com/watch?v=…"
              aria-label="YouTube link"
              className="field min-w-0 flex-1"
            />
            <Button onClick={() => void runProbe()} busy={probing} disabled={!url.trim()}>
              Check
            </Button>
          </div>

          {probing ? (
            <p className="mt-2 flex items-center gap-2 text-[12px] text-ink3">
              <IconSpinner width={12} height={12} />
              Reading the video&rsquo;s details…
            </p>
          ) : probeError ? (
            <p className="mt-2 flex items-start gap-2 text-[12px] text-danger">
              <IconAlert width={13} height={13} className="mt-px shrink-0" />
              {probeError}
            </p>
          ) : probe ? (
            <div className="mt-2.5 flex items-start gap-2.5 rounded-xl border border-line bg-surface/70 px-3 py-2.5">
              <IconCheck width={14} height={14} className="mt-0.5 shrink-0 text-ok" />
              <div className="min-w-0">
                <div className="truncate text-[13px] font-medium text-ink">{probe.title}</div>
                <div className="mt-0.5 font-mono text-[11.5px] text-ink3">
                  {[probe.uploader, fmtClock(probe.duration_s), probe.video_id]
                    .filter(Boolean)
                    .join(' · ')}
                </div>
              </div>
            </div>
          ) : null}
        </section>

        <section className="grid gap-4 sm:grid-cols-2">
          <div>
            <Eyebrow className="mb-2">Dub into</Eyebrow>
            <Select
              ariaLabel="Target language"
              value={language}
              onValueChange={setLanguage}
              items={(options?.languages ?? []).map((l) => ({
                value: l.code,
                label: (
                  <span>
                    {l.english}
                    {l.native ? (
                      <span className={`ml-1.5 text-ink3 ${langClass(l.code)}`}>
                        {l.native}
                      </span>
                    ) : null}
                  </span>
                ),
              }))}
            />
          </div>
          <div>
            <Eyebrow className="mb-2">Lip-sync</Eyebrow>
            <Select
              ariaLabel="Lip-sync model"
              value={lipsync}
              onValueChange={setLipsync}
              items={[
                ...(options?.lipsync_models ?? []).map((m) => ({
                  value: m.id,
                  label: m.label,
                  hint: m.note ?? m.speed,
                })),
                { value: 'off', label: 'Off — dubbed audio only', hint: 'Fastest by far' },
              ]}
            />
          </div>
        </section>

        <section>
          <Eyebrow className="mb-2">Which part</Eyebrow>
          {probe ? (
            <RangePicker
              duration={probe.duration_s ?? 0}
              whole={whole}
              onWholeChange={setWhole}
              start={start}
              end={end}
              onChange={(s, e) => {
                setStart(s);
                setEnd(e);
              }}
              capS={lipsync === 'off' ? Number.MAX_SAFE_INTEGER : cap}
            />
          ) : (
            <p className="t-sm">
              Check a link first — we need the video&rsquo;s length before you can pick part
              of it.
            </p>
          )}
        </section>

        {options?.tts_engines?.[0] || options?.limits.note ? (
          <div className="border-t border-line/50 pt-3 text-[11.5px] leading-relaxed text-ink3">
            {/* Stated, not chosen. The engine decides which languages are on
                offer and whether tone edits will do anything later, so the
                operator should be able to see it even though the form no
                longer lets them change it. */}
            {options?.tts_engines?.[0] ? (
              <p>
                Voices: {options.tts_engines[0].label}
                {options.tts_engines[0].note ? ` — ${options.tts_engines[0].note}` : ''}
              </p>
            ) : null}
            {options?.limits.note ? <p className="mt-1">{options.limits.note}</p> : null}
          </div>
        ) : null}
      </div>
    </Dialog>
  );
}
