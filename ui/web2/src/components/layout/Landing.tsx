import { useMemo } from 'react';
import { Button } from '../ui/Button';
import { IconFilm, IconGlobe, IconSparkle, IconWave } from '../ui/Icons';
import { useStore } from '../../store/useStore';
import { LANGUAGES } from '../../lib/lang';
import { StartPanel } from './StartPanel';

/**
 * What the user sees before they have picked a dub — and the first thing a
 * newcomer sees at all.
 *
 * Every claim on this page maps to something the pipeline actually does, so
 * it stays true as the product grows: the language list is the same table
 * `ui/langs.py` serves, the four steps are the seven pipeline phases folded
 * in half, and the lip-sync line names the three models `GET /api/options`
 * actually offers. Nothing here promises a capability the server does not
 * have — a landing page that oversells is the first place a tool starts
 * lying about itself.
 */
const STEPS = [
  {
    Icon: IconFilm,
    title: 'Point it at a video',
    body: 'Paste a YouTube link and choose the whole thing or just a section.',
  },
  {
    Icon: IconWave,
    title: 'It works out who is speaking',
    body: 'Speakers are separated, transcribed and timed, so each line keeps its own voice.',
  },
  {
    Icon: IconGlobe,
    title: 'Translated and re-voiced',
    body: 'The script is translated, cast to a voice per speaker and mixed over the original audio.',
  },
  {
    Icon: IconSparkle,
    title: 'Mouths matched to the new audio',
    body: 'Optional lip-sync re-renders the speaker’s mouth with Wav2Lip, MuseTalk or LatentSync.',
  },
];

export function Landing() {
  const jobs = useStore((s) => s.jobs);
  const loaded = useStore((s) => s.jobsLoaded);
  const selectJob = useStore((s) => s.selectJob);

  const langs = useMemo(() => {
    const all = Object.values(LANGUAGES).filter((l) => l.script !== 'latn');
    return { count: all.length, sample: all.slice(0, 9) };
  }, []);

  const recent = jobs.slice(0, 3);

  return (
    <div className="mx-auto w-full max-w-4xl px-1 py-8 sm:py-12">
      <div className="flex flex-col items-start gap-5">
        <span
          className="inline-flex items-center gap-2 rounded-full border border-line
                     bg-surface/70 px-3 py-1 text-[11.5px] font-medium text-brand-ink"
        >
          <span className="h-1.5 w-1.5 rounded-full bg-aqua" aria-hidden="true" />
          Indic video dubbing, end to end
        </span>

        <h1 className="max-w-2xl text-[30px] font-semibold leading-[1.15] tracking-tight text-ink sm:text-[38px]">
          Dub a video into an Indian language
          <span className="bg-gradient-to-r from-brand-ink to-aqua-ink bg-clip-text text-transparent">
            {' '}
            without re-shooting it
          </span>
          .
        </h1>

        <p className="max-w-2xl text-[15px] leading-relaxed text-ink2">
          Dubbing Studio takes the video you already have, works out who is speaking, translates
          what they say, gives each speaker a voice in the target language and — when you want it
          — moves their mouth to match. You watch the result here, compare it against the
          original, and recast a voice without re-running the whole thing.
        </p>

        <div className="flex flex-wrap items-center gap-3">
          <Button
            variant="primary"
            size="lg"
            onClick={() => document.querySelector<HTMLInputElement>('#start-url')?.focus()}
          >
            Start a dub
          </Button>
          <span className="text-[12.5px] text-ink3">
            paste a link below — or press <Kbd>N</Kbd> from anywhere
          </span>
        </div>
      </div>

      {/* The home page starts a dub itself. The dialog is still there for
          the window picker and the lip-sync model; this covers the common
          case in one screen. */}
      <div className="mt-8">
        <StartPanel />
      </div>

      {/* The other thing this tool can do. Kept to one line and one link:
          it is a different job from dubbing a video, and burying it in the
          dub form would make both harder to find. */}
      <button
        type="button"
        onClick={() => {
          location.hash = '#/talking-head';
        }}
        className="card mt-3 flex w-full items-center gap-3 p-4 text-left transition-colors
                   hover:border-edge/70 hover:bg-surface"
      >
        <span
          className="grid h-9 w-9 flex-none place-items-center rounded-xl border border-line
                     bg-gradient-to-br from-brand/25 to-aqua/15 text-brand-ink"
          aria-hidden="true"
        >
          <IconSparkle width={17} height={17} />
        </span>
        <span className="min-w-0">
          <span className="block text-[13.5px] font-semibold text-ink">
            Talking head — make a photo speak
          </span>
          <span className="mt-1 block text-[12.5px] leading-relaxed text-ink3">
            Upload a portrait, pick a language and a line. No video needed.
          </span>
        </span>
      </button>

      <ol className="mt-10 grid gap-3 sm:grid-cols-2">
        {STEPS.map(({ Icon, title, body }, i) => (
          <li key={title} className="card flex gap-3 p-4">
            <span
              className="grid h-9 w-9 flex-none place-items-center rounded-xl border border-line
                         bg-gradient-to-br from-brand/25 to-aqua/15 text-brand-ink"
              aria-hidden="true"
            >
              <Icon width={17} height={17} />
            </span>
            <div className="min-w-0">
              <div className="text-[13.5px] font-semibold text-ink">
                <span className="mr-1.5 font-mono text-[12px] text-ink4">{i + 1}</span>
                {title}
              </div>
              <p className="mt-1 text-[12.5px] leading-relaxed text-ink3">{body}</p>
            </div>
          </li>
        ))}
      </ol>

      <section className="card mt-3 p-4">
        <div className="text-[13.5px] font-semibold text-ink">
          {langs.count} Indian languages, in their own script
        </div>
        <p className="mt-1 text-[12.5px] leading-relaxed text-ink3">
          Technical terms can stay in English where that is how people actually say them.
        </p>
        <ul className="mt-3 flex flex-wrap gap-1.5">
          {langs.sample.map((l) => (
            <li
              key={l.native}
              className={`rounded-lg border border-line bg-canvas/50 px-2.5 py-1 text-[12.5px] text-ink2 lang-${l.script}`}
            >
              {l.native}
            </li>
          ))}
          <li className="rounded-lg border border-line bg-canvas/50 px-2.5 py-1 text-[12.5px] text-ink4">
            +{langs.count - langs.sample.length} more
          </li>
        </ul>
      </section>

      {loaded && recent.length > 0 ? (
        <section className="mt-8">
          <div className="eyebrow mb-2">Pick up where you left off</div>
          <div className="grid gap-2 sm:grid-cols-3">
            {recent.map((j) => (
              <button
                key={j.job_id}
                type="button"
                onClick={() => selectJob(j.job_id)}
                className="card px-3 py-2.5 text-left transition-colors hover:border-edge/70 hover:bg-surface"
              >
                <div className="truncate text-[12.5px] font-medium text-ink">
                  {j.title ?? j.slug}
                </div>
                <div className="mt-0.5 truncate text-[11.5px] text-ink4">{j.job_id}</div>
              </button>
            ))}
          </div>
        </section>
      ) : null}
    </div>
  );
}

export function Kbd({ children }: { children: React.ReactNode }) {
  return (
    <kbd
      className="rounded-md border border-edge/70 bg-surface px-1.5 py-0.5 font-mono
                 text-[11px] font-medium text-ink2"
    >
      {children}
    </kbd>
  );
}
