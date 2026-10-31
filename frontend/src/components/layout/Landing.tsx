import { useMemo } from 'react';
import { IconFilm, IconGlobe, IconSparkle } from '../frontend/Icons';
import { useStore } from '../../store/useStore';
import { LANGUAGES } from '../../lib/lang';
import { StartPanel } from './StartPanel';

const STEPS = [
  {
    Icon: IconFilm,
    title: 'Identify',
    body: 'Detect speakers and dialogue.',
  },
  {
    Icon: IconGlobe,
    title: 'Dub',
    body: 'Translate and generate voices.',
  },
  {
    Icon: IconSparkle,
    title: 'Lip-sync',
    body: 'Lip-sync and resolution enhancement.',
  },
];

export function Landing() {
  const jobs = useStore((s) => s.jobs);
  const loaded = useStore((s) => s.jobsLoaded);
  const selectJob = useStore((s) => s.selectJob);

  const langs = useMemo(() => {
    const all = Object.values(LANGUAGES).filter((l) => l.script !== 'latn');
    return { count: all.length, all };
  }, []);

  const recent = jobs.slice(0, 3);

  return (
    <div className="mx-auto w-full max-w-5xl px-3 py-4 sm:px-6 sm:py-8 lg:px-10" >
      <div className="flex flex-col items-start gap-5">
        <span
          className="inline-flex items-center gap-2 rounded-full border border-line
                     bg-surface/70 px-3 py-1 text-[11.5px] font-medium text-brand-ink"
        >
          <span className="h-1.5 w-1.5 rounded-full bg-aqua" aria-hidden="true" />
          Indic video dubbing, end to end
        </span>

        <h1 className="max-w-2xl text-[30px] font-semibold leading-[1.15] tracking-tight text-ink sm:text-[38px]">
          Bring your videos to more
          <span className="bg-gradient-to-r from-brand-ink to-aqua-ink bg-clip-text text-transparent">
            {' '}
            Indian-language audiences
          </span>
          .
        </h1>

        <p className="max-w-2xl text-[15px] leading-relaxed text-ink2">
          Transcribe, translate and re-voice your video in any supported Indian language.
        </p>

      </div>

      <div className="mt-5">
        <StartPanel />
      </div>

      <ol className="mt-6 grid gap-3 sm:grid-cols-3">
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
          {langs.all.map((l) => (
            <li
              key={l.native}
              className={`rounded-lg border border-line bg-canvas/50 px-2.5 py-1 text-[12.5px] text-ink2 lang-${l.script}`}
            >
              {l.native}
            </li>
          ))}
        </ul>
      </section>

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
            Create a talking photo
          </span>
          <span className="mt-1 block text-[12.5px] leading-relaxed text-ink3">
            Upload a portrait, pick a language and a line. No video needed.
          </span>
        </span>
      </button>

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
