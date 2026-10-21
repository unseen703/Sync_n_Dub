import { useEffect, useRef, useState } from 'react';
import { Button } from '../ui/Button';
import { Select } from '../ui/Select';
import { Eyebrow } from '../ui/Card';
import { RangePicker } from '../newjob/RangePicker';
import { IconAlert, IconFilm, IconSpinner } from '../ui/Icons';
import { useStore } from '../../store/useStore';
import * as api from '../../api';
import { ensureScriptFont, langClass } from '../../lib/lang';
import { fmtBytes, fmtClock } from '../../lib/format';
import type { Options, ProbeResult, UploadResult } from '../../api/types';

/**
 * Start a dub from the home page: paste a link, or upload a file.
 *
 * The two routes converge deliberately. A YouTube link is probed by
 * `POST /api/youtube/probe` (yt-dlp, no download); a local file is sent to
 * `POST /api/uploads` and probed with ffprobe. Both answer with the same
 * `{title, duration_s, estimates}`, so everything below this point —
 * the language and type pickers, the quote, the submit — has one code
 * path and cannot drift between sources.
 *
 * `Type` is the whole-dub delivery register (`tts_style`). Four formats,
 * no emotions: an emotion is a per-segment mood that stage 9b2 derives
 * from the source speaker's own delivery, so choosing one here would
 * flatten every line of the video into it. Left unset, stage 9b1 reads
 * the register off the video's own metadata instead.
 */
type Source =
  | { kind: 'url'; probe: ProbeResult }
  | { kind: 'upload'; upload: UploadResult };

export function StartPanel() {
  const refreshJobs = useStore((s) => s.refreshJobs);
  const selectJob = useStore((s) => s.selectJob);
  const pushToast = useStore((s) => s.pushToast);

  const [options, setOptions] = useState<Options | null>(null);
  const [url, setUrl] = useState('');
  const [src, setSrc] = useState<Source | null>(null);
  const [busy, setBusy] = useState<'probe' | 'upload' | 'create' | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [language, setLanguage] = useState('hi');
  // NOT '' — Radix <Select.Item> throws on an empty value
  // string, which took the whole app down to a white page.
  const [style, setStyle] = useState('auto');
  // No lip-sync control here any more. A submission dubs the audio and
  // stops; the model is named afterwards, on the review page, against the
  // face size this run measured. Offering the choice here meant committing
  // the most expensive stage in the pipeline before anyone had seen the
  // dub — and "auto" meant committing it from a number nobody saw.
  const [whole, setWhole] = useState(false);
  const [start, setStart] = useState(0);
  const [end, setEnd] = useState(60);
  // Preselected from the server, so the picker and the pipeline agree.
  const [spoken, setSpoken] = useState('en');
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    void api.getOptions().then((o) => {
      setOptions(o);
      setSpoken(o.default_source_language ?? 'auto');
      if (!o.languages.some((l) => l.code === language)) {
        setLanguage(o.languages[0]?.code ?? 'hi');
      }
    }).catch(() => undefined);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { ensureScriptFont(language); }, [language]);

  const title = src?.kind === 'url' ? src.probe.title : src?.upload.title;
  const duration = src?.kind === 'url' ? src.probe.duration_s : src?.upload.duration_s;

  // The cap is on the LIP-SYNCED portion, and nothing is lip-synced at
  // submit any more — so this is a heads-up, not a gate. Blocking here
  // would refuse a 90-minute audio dub for compute it will never spend;
  // the real refusal lives at `POST .../lipsync`, where the model is known.
  const cap = options?.limits.max_lipsync_s ?? 1200;
  const span = whole ? (duration ?? 0) : end - start;
  const overCapForLater = span > cap;

  const check = async () => {
    if (!url.trim()) return;
    setBusy('probe'); setError(null);
    try {
      const p = await api.probeYouTube(url.trim());
      setSrc({ kind: 'url', probe: p });
      setStart(0); setEnd(Math.min(60, p.duration_s ?? 60)); setWhole(false);
    } catch (e) {
      setSrc(null); setError((e as Error).message);
    } finally { setBusy(null); }
  };

  const upload = async (file: File) => {
    setBusy('upload'); setError(null); setUrl('');
    try {
      const u = await api.uploadVideo(file);
      setSrc({ kind: 'upload', upload: u });
      setStart(0); setEnd(Math.min(60, u.duration_s ?? 60)); setWhole(false);
    } catch (e) {
      setSrc(null); setError((e as Error).message);
    } finally { setBusy(null); }
  };

  const create = async () => {
    if (!src) return;
    setBusy('create'); setError(null);
    try {
      const res = await api.createJob({
        ...(src.kind === 'url'
          ? { url: src.probe.webpage_url ?? url.trim() }
          : { upload_id: src.upload.upload_id }),
        target_language: language,
        style: style === 'auto' ? null : style || null,
        source_language: spoken || null,
        window: whole ? null : { start_s: start, end_s: end },
      });
      await refreshJobs();
      selectJob(res.job_id);
      pushToast({ kind: 'success', title: 'Job started', body: res.estimate?.text });
    } catch (e) {
      setError((e as Error).message);
    } finally { setBusy(null); }
  };

  return (
    <section className="card p-4 sm:p-5">
      <div className="text-[13.5px] font-semibold text-ink">Start a dub</div>

      <div className="mt-3 flex flex-col gap-2 sm:flex-row">
        <input
          id="start-url"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') void check(); }}
          placeholder="Paste a YouTube link"
          aria-label="YouTube link"
          className="min-w-0 flex-1 rounded-lg border border-edge/70 bg-canvas/60 px-3 py-2
                     text-[13px] text-ink placeholder:text-ink4"
        />
        <Button onClick={() => void check()} disabled={!url.trim() || busy !== null}>
          {busy === 'probe' ? <IconSpinner /> : null} Check
        </Button>
        {/* A real <input type=file>, kept offscreen and driven by the
            button, so the control is keyboard-reachable and the browser's
            own picker does the work. */}
        <input
          ref={fileRef}
          type="file"
          accept="video/mp4,video/quicktime,video/x-matroska,video/webm,.mp4,.mov,.mkv,.webm,.m4v,.avi"
          className="sr-only"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) void upload(f);
            e.target.value = '';
          }}
        />
        <Button
          variant="outline"
          onClick={() => fileRef.current?.click()}
          disabled={busy !== null}
        >
          {busy === 'upload' ? <IconSpinner /> : null} Upload video
        </Button>
      </div>

      {error ? (
        <p className="mt-2 flex items-start gap-1.5 text-[12.5px] text-warn">
          <IconAlert width={13} height={13} className="mt-0.5 flex-none" />
          <span>{error}</span>
        </p>
      ) : null}

      {/* The three pickers do NOT wait for a source. None of them depends
          on it, and hiding them until a link resolves made the only two
          language controls in the product invisible on first load — the
          question "where do I choose the languages" answers itself when
          they are simply on screen. */}
      <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <div>
              {/* Stating the spoken language beats detection. Each
                  diarization turn is its own ASR call, and on 1-5 s of
                  audio the detector is unreliable — one English lecture
                  came back in four different Indic scripts. The adapter
                  now pins one detection for the whole file, but a wrong
                  pin is then consistently wrong. */}
              <Eyebrow className="mb-2">Spoken</Eyebrow>
              <Select
                ariaLabel="Spoken language in the source"
                value={spoken}
                onValueChange={setSpoken}
                items={[
                  { value: 'auto', label: 'Detect it',
                    hint: 'Detected once from the longest turn, then pinned' },
                  { value: 'en', label: 'English' },
                  ...(options?.languages ?? []).map((l) => ({
                    value: l.code,
                    label: l.english,
                  })),
                ]}
              />
            </div>
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
                        <span className={`ml-1.5 text-ink3 ${langClass(l.code)}`}>{l.native}</span>
                      ) : null}
                    </span>
                  ),
                }))}
              />
            </div>
            <div>
              <Eyebrow className="mb-2">Type</Eyebrow>
              <Select
                ariaLabel="Delivery type"
                value={style}
                onValueChange={setStyle}
                items={[
                  { value: 'auto', label: 'Work it out from the video',
                    hint: 'Stage 9b1 reads the register off the video’s metadata' },
                  ...(options?.styles ?? []).map((s) => ({ value: s.value, label: s.label })),
                ]}
              />
            </div>
      </div>

      <p className="mt-3 rounded-lg border border-line bg-surface/60 px-3 py-2
                    text-[12px] text-ink3">
        This dubs the audio and stops. You review the cast, the script and the
        measured face size, then choose a lip-sync model and start it from the
        job page — so no GPU time is committed before you have heard the dub.
      </p>

      {src ? (
        <>
          <div className="mt-3 flex items-center gap-2 rounded-lg border border-line
                          bg-surface/60 px-3 py-2 text-[12.5px]">
            <IconFilm width={14} height={14} className="flex-none text-ink3" />
            <span className="min-w-0 flex-1 truncate text-ink">{title}</span>
            <span className="flex-none font-mono text-[11.5px] text-ink4">
              {duration ? fmtClock(duration) : '—'}
              {src.kind === 'upload' ? ` · ${fmtBytes(src.upload.size_bytes)}` : ''}
            </span>
          </div>

          <div className="mt-3">
            <Eyebrow className="mb-2">Which part</Eyebrow>
            <RangePicker
              duration={duration ?? 0}
              whole={whole}
              onWholeChange={setWhole}
              start={start}
              end={end}
              onChange={(a, b) => { setStart(a); setEnd(b); }}
              capS={Number.MAX_SAFE_INTEGER}
            />
          </div>

          {overCapForLater ? (
            <p className="mt-2 flex items-start gap-1.5 text-[12.5px] text-warn">
              <IconAlert width={13} height={13} className="mt-0.5 flex-none" />
              <span>
                The dub will run fine, but {Math.round(span / 60)} minutes is
                over the {Math.round(cap / 60)}-minute lip-sync cap — you will
                not be able to lip-sync this one afterwards. Pick a shorter
                part if you need that.
              </span>
            </p>
          ) : null}

          <div className="mt-3 flex flex-wrap items-center gap-3">
            <Button variant="primary" onClick={() => void create()}
                    disabled={busy !== null}>
              {busy === 'create' ? <IconSpinner /> : null} Start the dub
            </Button>
            <span className="text-[12px] text-ink3">
              {/* By VALUE, not by preset name. This read `estimates.standard`
                  and silently rendered nothing the moment the presets
                  collapsed to one — an estimate that quietly disappears is
                  worse than one that is wrong. */}
              {Object.values(
                (src.kind === 'url' ? src.probe.estimates : src.upload.estimates) ?? {},
              )[0]?.text}
            </span>
          </div>
        </>
      ) : null}
    </section>
  );
}
