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

type Source =
  | { kind: 'url'; probe: ProbeResult }
  | { kind: 'upload'; upload: UploadResult };

/** New-dub form. It collects a source and settings from the user, then creates a job. */
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

  const [style, setStyle] = useState('auto');
  const [targetScript, setTargetScript] = useState('codemix');

  const [whole, setWhole] = useState(false);
  const [start, setStart] = useState(0);
  const [end, setEnd] = useState(60);

  const [spoken, setSpoken] = useState('auto');
  const [detectedLang, setDetectedLang] = useState<{ code: string; language?: string; method?: string } | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    void api.getOptions().then((o) => {
      setOptions(o);
      setSpoken(o.default_source_language ?? 'auto');
      setDetectedLang(null);
      if (!o.languages.some((l) => l.code === language)) {
        setLanguage(o.languages[0]?.code ?? 'hi');
      }
    }).catch(() => undefined);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { ensureScriptFont(language); }, [language]);

  const title = src?.kind === 'url' ? src.probe.title : src?.upload.title;
  const duration = src?.kind === 'url' ? src.probe.duration_s : src?.upload.duration_s;

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
      if (p.detected_source_language) {
        setDetectedLang(p.detected_source_language);
        if (spoken === 'auto') setSpoken(p.detected_source_language.code);
      } else {
        setDetectedLang(null);
      }
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
      if (u.detected_source_language) {
        setDetectedLang(u.detected_source_language);
        if (spoken === 'auto') setSpoken(u.detected_source_language.code);
      } else {
        setDetectedLang(null);
      }
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
        code_mix: targetScript === 'codemix',
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
      <div className="text-[13.5px] font-semibold text-ink">Add your source video</div>

      <div className="mt-3 flex flex-col gap-2 sm:flex-row">
        <input
          id="start-url"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') void check(); }}
          placeholder="Paste a YouTube or video link"
          aria-label="Video link"
          className="min-w-0 flex-1 rounded-lg border border-edge/70 bg-canvas/60 px-3 py-2
                     text-[13px] text-ink placeholder:text-ink4"
        />
        <Button onClick={() => void check()} disabled={!url.trim() || busy !== null}>
          {busy === 'probe' ? <IconSpinner /> : null} Check
        </Button>

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
          {busy === 'upload' ? <IconSpinner /> : null} Upload a file
        </Button>
      </div>

      <p className="mt-1.5 text-[11.5px] text-ink4">
        Accepts MP4, MOV, MKV, WebM, M4V or AVI up to 4 GB.
      </p>

      {error ? (
        <p className="mt-2 flex items-start gap-1.5 text-[12.5px] text-warn">
          <IconAlert width={13} height={13} className="mt-0.5 flex-none" />
          <span>{error}</span>
        </p>
      ) : null}

      <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <div>

              <Eyebrow className="mb-2">Original language</Eyebrow>
              <Select
                ariaLabel="Original language"
                value={spoken}
                onValueChange={setSpoken}
                items={[
                  { value: 'auto', label: detectedLang
                    ? `Automatic detection — ${detectedLang.language || detectedLang.code}`
                    : 'Automatic detection',
                    hint: detectedLang
                      ? `Detected via ${detectedLang.method === 'yt_metadata' ? 'video metadata' : 'audio analysis'}`
                      : 'Detected from video metadata or a short audio sample' },
                  { value: 'en', label: 'English' },
                  ...(options?.languages ?? []).map((l) => ({
                    value: l.code,
                    label: l.english,
                  })),
                ]}
              />
            </div>
            <div>
              <Eyebrow className="mb-2">Indian language to dub into</Eyebrow>
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
              <Eyebrow className="mb-2">Delivery style</Eyebrow>
              <Select
                ariaLabel="Delivery style"
                value={style}
                onValueChange={setStyle}
                items={[
                  { value: 'auto', label: 'Work it out from the video',
                    hint: 'Automatically detected from the video' },
                  ...(options?.styles ?? []).map((s) => ({ value: s.value, label: s.label })),
                ]}
              />
            </div>
            <div>
              <Eyebrow className="mb-2">Code-mix</Eyebrow>
              <Select
                ariaLabel="Code-mix"
                value={targetScript}
                onValueChange={setTargetScript}
                items={[
                  { value: 'codemix', label: 'Code-mix' },
                  { value: 'native', label: 'Native' },
                ]}
              />
            </div>
      </div>

      <p className="mt-3 rounded-lg border border-line bg-surface/60 px-3 py-2
                    text-[12px] text-ink3">
        Audio dubbing runs first. You review the result, adjust voices and — when
        you're ready — add lip-sync from the job page.
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
              {busy === 'create' ? <IconSpinner /> : null} Create dubbed video
            </Button>
            <span className="text-[12px] text-ink3">

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
