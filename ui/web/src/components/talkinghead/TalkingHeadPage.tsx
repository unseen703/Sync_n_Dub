import { useCallback, useEffect, useRef, useState } from 'react';
import { Button } from '../ui/Button';
import { IconAlert, IconSparkle } from '../ui/Icons';
import { IconSpinner } from '../ui/Icons';
import {
  thDialogues, thOptions, thRender, thRenderStatus, thUploadPhoto,
  type THCategory, type THDialogue, type THOptions, type THPhoto,
  type THRender,
} from '../../api/talkingHead';

/**
 * Make a still portrait speak.
 *
 * Three steps, gated in order, because each one is useless without the one
 * before it: a photo WITH A FACE, a language, then a line.
 *
 * The face check is server-side and happens on upload rather than at render
 * time — finding out a picture has no face after a GPU job is queued is the
 * expensive way to learn it.
 *
 * What this page must not do is oversell. MuseTalk animates the MOUTH and
 * copies the rest of the frame through (measured: 0.87% of the frame changes
 * between output frames). A user expecting a nodding, blinking avatar would
 * read a correct render as a broken one, so the note from the server is
 * shown up front, not buried.
 *
 * A line with no `audio_url` is listed and MARKED rather than hidden: an
 * empty page would suggest the feature is missing, when the only missing
 * piece is the clip. (Every line shipped that way until the 50 clips were
 * supplied; the rule stays, for lines added later.)
 *
 * The render is a background job, so this page POSTs once and then polls.
 * It is 8-21x the clip length on this host — minutes — which is long enough
 * that a synchronous request would die to a browser timeout while the GPU
 * work carried on.
 */
export function TalkingHeadPage() {
  const [opts, setOpts] = useState<THOptions | null>(null);
  const [lang, setLang] = useState('hi');
  const [cat, setCat] = useState<string>('all');
  const [lines, setLines] = useState<THDialogue[]>([]);
  const [msg, setMsg] = useState<string | null>(null);
  const [photo, setPhoto] = useState<THPhoto | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [picked, setPicked] = useState<string | null>(null);
  const [render, setRender] = useState<THRender | null>(null);
  const [rendering, setRendering] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    thOptions().then(setOpts).catch((e) => setErr(String(e.message ?? e)));
  }, []);

  useEffect(() => {
    let live = true;
    thDialogues(lang)
      .then((d) => {
        if (!live) return;
        setLines(d.dialogues);
        setMsg(d.message);
        setPicked(null);
      })
      .catch((e) => live && setErr(String(e.message ?? e)));
    return () => {
      live = false;
    };
  }, [lang]);

  const onFile = useCallback(async (f: File | undefined) => {
    if (!f) return;
    setBusy(true);
    setErr(null);
    try {
      setPhoto(await thUploadPhoto(f));
    } catch (e) {
      setErr(String((e as Error).message ?? e));
      setPhoto(null);
    } finally {
      setBusy(false);
    }
  }, []);

  /**
   * Poll one render until it stops being `running`.
   *
   * Driven off `render.state` rather than a boolean of our own, because the
   * server recomputes that state from the file on every call — a render
   * whose process was killed reports `failed`, not `running`, so this stops
   * instead of polling until the tab closes.
   */
  useEffect(() => {
    if (!render || render.state !== 'running') return;
    let live = true;
    const t = setInterval(() => {
      thRenderStatus(render.render_id)
        .then((r) => {
          if (!live) return;
          setRender(r);
          if (r.state !== 'running') setRendering(false);
        })
        .catch(() => {
          /* a dropped poll is not a failed render; the next tick retries */
        });
    }, 3000);
    return () => {
      live = false;
      clearInterval(t);
    };
  }, [render]);

  const startRender = useCallback(async () => {
    if (!photo?.photo_id || !picked) return;
    setRendering(true);
    setErr(null);
    setRender(null);
    try {
      setRender(await thRender({
        photo_id: photo.photo_id, dialogue_id: picked, language: lang,
      }));
    } catch (e) {
      // The server's refusals are the useful ones and they are specific:
      // the clip is missing from disk, the photo is no longer staged, no
      // GPU is free. Show the sentence rather than "render failed".
      setErr(String((e as Error).message ?? e));
      setRendering(false);
    }
  }, [photo?.photo_id, picked, lang]);

  const cats: THCategory[] = opts?.categories ?? [];
  const shown = cat === 'all' ? lines : lines.filter((l) => l.category === cat);
  const faceOk = photo?.ok === true;
  const pickedLine = lines.find((l) => l.id === picked) ?? null;
  // Every gate, in the order the page presents them, so the button can say
  // which one is not met instead of being inertly grey.
  const blocker = !faceOk
    ? 'Upload a portrait with a clear face first.'
    : !picked
      ? 'Choose a line for it to say.'
      : !pickedLine?.playable
        ? 'That line has no audio clip yet, so there is nothing to animate to.'
        : null;

  return (
    <div className="mx-auto w-full max-w-3xl px-1 py-6">
      <div className="eyebrow mb-1">Talking head</div>
      <h1 className="text-[22px] font-semibold tracking-tight text-ink">
        Make a photo speak
      </h1>
      <p className="mt-1.5 max-w-prose text-[13px] leading-relaxed text-ink3">
        Upload a portrait, pick a language, choose a line. {opts?.note}
      </p>

      {err ? (
        <div
          role="alert"
          className="mt-4 flex items-start gap-2 rounded-lg border border-amber-500/40
                     bg-amber-500/10 px-3 py-2 text-[12.5px] text-ink2"
        >
          <IconAlert className="mt-0.5 h-4 w-4 shrink-0 text-amber-500" />
          <span>{err}</span>
        </div>
      ) : null}

      {/* 1 — the photo, and the face gate */}
      <section className="mt-6">
        <div className="eyebrow mb-2">1 · Photo</div>
        <div className="card flex flex-wrap items-center gap-3 px-3 py-3">
          <input
            ref={fileRef}
            type="file"
            accept=".jpg,.jpeg,.png,.bmp,.webp"
            className="sr-only"
            onChange={(e) => void onFile(e.target.files?.[0])}
          />
          <Button onClick={() => fileRef.current?.click()} disabled={busy}>
            {busy ? 'Checking for a face…' : photo ? 'Choose another' : 'Choose a photo'}
          </Button>
          <span className="text-[12px] text-ink4">
            JPG, PNG, BMP or WebP{opts ? ` · up to ${opts.max_image_mb} MB` : ''}
          </span>
        </div>

        {photo ? (
          <div className="mt-3 flex flex-wrap items-start gap-4">
            {photo.url ? (
              <img
                src={photo.url}
                alt="The portrait you uploaded"
                className="h-28 w-28 rounded-lg border border-line object-cover"
              />
            ) : null}
            <div className="min-w-[14rem] flex-1 text-[12.5px]">
              {faceOk ? (
                <p className="font-medium text-emerald-500">
                  Face found — {photo.width}×{photo.height}
                  {photo.n_faces && photo.n_faces > 1
                    ? ` · ${photo.n_faces} faces` : ''}
                </p>
              ) : (
                <p className="font-medium text-amber-500">{photo.message}</p>
              )}
              {(photo.warnings ?? []).map((w) => (
                <p key={w} className="mt-1 text-ink4">{w}</p>
              ))}
            </div>
          </div>
        ) : null}
      </section>

      {/* 2 — language. Disabled until there is a face, so the order of the
          steps is visible rather than merely implied. */}
      <section className={`mt-6 ${faceOk ? '' : 'opacity-45'}`}>
        <div className="eyebrow mb-2">2 · Language</div>
        <select
          value={lang}
          disabled={!faceOk}
          onChange={(e) => setLang(e.target.value)}
          className="w-full max-w-xs rounded-lg border border-edge bg-surface px-2.5
                     py-1.5 text-[13px] text-ink disabled:cursor-not-allowed"
        >
          {(opts?.languages ?? []).map((l) => (
            <option key={l.code} value={l.code}>
              {l.english}
              {l.native ? ` — ${l.native}` : ''}
              {l.has_dialogues ? ` (${l.n_dialogues})` : ' — no lines yet'}
            </option>
          ))}
        </select>
      </section>

      {/* 3 — the line */}
      <section className={`mt-6 ${faceOk ? '' : 'opacity-45'}`}>
        <div className="eyebrow mb-2">3 · Pick a line</div>
        <div className="mb-3 flex flex-wrap gap-1.5">
          {[{ id: 'all', label: 'All' }, ...cats].map((c) => (
            <button
              key={c.id}
              type="button"
              disabled={!faceOk}
              onClick={() => setCat(c.id)}
              className={`rounded-lg border px-2.5 py-1 text-[12px] transition-colors
                ${cat === c.id
                  ? 'border-accent/60 bg-accent/10 text-ink'
                  : 'border-line text-ink3 hover:border-edge/70'}`}
            >
              {c.label}
            </button>
          ))}
        </div>

        {msg ? (
          <p className="text-[12.5px] text-ink4">{msg}</p>
        ) : (
          <ul className="grid gap-2">
            {shown.map((d) => (
              <li key={d.id}>
                <button
                  type="button"
                  disabled={!faceOk}
                  onClick={() => setPicked(d.id)}
                  className={`w-full rounded-lg border px-3 py-2.5 text-left transition-colors
                    ${picked === d.id
                      ? 'border-accent/60 bg-accent/10'
                      : 'border-line hover:border-edge/70 hover:bg-surface'}`}
                >
                  <div className="flex items-center gap-2">
                    <span className="eyebrow shrink-0">{d.category}</span>
                    {!d.playable ? (
                      <span
                        className="rounded border border-amber-500/40 bg-amber-500/10
                                   px-1.5 py-0.5 text-[10.5px] text-amber-500"
                        title="The line is in the catalogue; its audio clip has not been supplied yet."
                      >
                        no audio yet
                      </span>
                    ) : null}
                  </div>
                  <div className="mt-1 text-[13.5px] leading-snug text-ink">{d.text}</div>
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="mt-7 border-t border-line pt-4">
        <div className="flex flex-wrap items-center gap-3">
          <Button
            variant="primary"
            onClick={() => void startRender()}
            disabled={Boolean(blocker) || rendering}
            title={blocker ?? 'Animate the photo against the chosen line'}
          >
            {rendering ? (
              <IconSpinner className="mr-1.5 h-4 w-4" />
            ) : (
              <IconSparkle className="mr-1.5 h-4 w-4" />
            )}
            {rendering ? 'Generating…' : 'Generate talking head'}
          </Button>
          {/* The unmet gate, in words. A greyed-out button with no reason
              beside it is the shape this page shipped with, and it read as
              "broken" rather than "not yet". */}
          {blocker ? (
            <span className="text-[12px] text-ink4">{blocker}</span>
          ) : rendering ? (
            <span className="text-[12px] text-ink4">
              This runs on the GPU and takes a few minutes. You can leave the
              page — the render carries on.
            </span>
          ) : null}
        </div>

        {render?.state === 'failed' ? (
          <p
            role="alert"
            className="mt-3 flex items-start gap-1.5 text-[12.5px] text-warn"
          >
            <IconAlert width={13} height={13} className="mt-0.5 flex-none" />
            <span>{render.error ?? 'The render did not produce a video.'}</span>
          </p>
        ) : null}

        {render?.ready && render.url ? (
          <div className="mt-4 space-y-2">
            <div className="eyebrow">Result</div>
            <video
              key={render.url}
              src={render.url}
              controls
              autoPlay
              className="w-full max-w-xl rounded-xl bg-black"
            />
            <div className="flex flex-wrap items-center gap-3 text-[12px] text-ink4">
              <a
                href={render.url}
                download
                className="text-ink2 underline underline-offset-2 hover:text-ink"
              >
                Download
              </a>
              {render.duration_s ? (
                <span>
                  {render.duration_s.toFixed(1)}s
                  {render.fps ? ` · ${render.fps.toFixed(0)} fps` : ''}
                </span>
              ) : null}
              {render.model ? <span>{render.model}</span> : null}
            </div>
            {/* Repeated HERE, next to the result, and not only at the top of
                the page. Mouth-only is measured behaviour (0.87% of the
                frame changes between output frames), and the moment someone
                is most likely to read a correct render as a broken one is
                the moment they first watch it. */}
            <p className="max-w-prose text-[11.5px] text-ink4">{opts?.note}</p>
          </div>
        ) : null}
      </section>
    </div>
  );
}
