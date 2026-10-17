/**
 * Language -> script -> font, and the lazy loader that makes it real.
 *
 * `ui/web/styles.css` has defined `.lang-hi` and `.lang-ta` since the first
 * commit and applies them NOWHERE, so every bit of Devanagari and Tamil in
 * the old inspector renders in whatever the browser falls back to. This
 * module is the fix: `langClass(code)` goes on every element that renders
 * target-language text, and `ensureScriptFont(code)` injects the matching
 * Noto subset the first time that script appears on screen.
 *
 * Lazy, not eager, because there are 13 scripts here and the subsets run
 * 5–57 kB each. Shipping all of them would roughly double the eager budget
 * for text the user will never see — a job dubs into one language.
 *
 * Self-hosted under `public/fonts/`. No CDN at runtime: the comment in
 * `ui/web/index.html` makes that a repo convention, and these files are
 * served by the same FastAPI process as everything else.
 */

export interface LangMeta {
  english: string;
  native: string;
  script: string;
}

/** Transcribed from `ui/langs.py::LANGUAGES`. Keep the two in step. */
export const LANGUAGES: Record<string, LangMeta> = {
  as: { english: 'Assamese', native: 'অসমীয়া', script: 'beng' },
  bn: { english: 'Bengali', native: 'বাংলা', script: 'beng' },
  brx: { english: 'Bodo', native: 'बड़ो', script: 'deva' },
  doi: { english: 'Dogri', native: 'डोगरी', script: 'deva' },
  en: { english: 'English', native: 'English', script: 'latn' },
  gu: { english: 'Gujarati', native: 'ગુજરાતી', script: 'gujr' },
  hi: { english: 'Hindi', native: 'हिन्दी', script: 'deva' },
  kn: { english: 'Kannada', native: 'ಕನ್ನಡ', script: 'knda' },
  kok: { english: 'Konkani', native: 'कोंकणी', script: 'deva' },
  ks: { english: 'Kashmiri', native: 'کٲشُر', script: 'arab' },
  mai: { english: 'Maithili', native: 'मैथिली', script: 'deva' },
  ml: { english: 'Malayalam', native: 'മലയാളം', script: 'mlym' },
  mni: { english: 'Manipuri', native: 'ꯃꯤꯇꯩꯂꯣꯟ', script: 'mtei' },
  mr: { english: 'Marathi', native: 'मराठी', script: 'deva' },
  ne: { english: 'Nepali', native: 'नेपाली', script: 'deva' },
  or: { english: 'Odia', native: 'ଓଡ଼ିଆ', script: 'orya' },
  pa: { english: 'Punjabi', native: 'ਪੰਜਾਬੀ', script: 'guru' },
  sa: { english: 'Sanskrit', native: 'संस्कृतम्', script: 'deva' },
  sat: { english: 'Santali', native: 'ᱥᱟᱱᱛᱟᱲᱤ', script: 'olck' },
  sd: { english: 'Sindhi', native: 'سنڌي', script: 'arab' },
  ta: { english: 'Tamil', native: 'தமிழ்', script: 'taml' },
  te: { english: 'Telugu', native: 'తెలుగు', script: 'telu' },
  ur: { english: 'Urdu', native: 'اردو', script: 'arab' },
};

/**
 * Script -> font family, matching `ui/langs.py::SCRIPT_FONTS`.
 *
 * The files are `public/fonts/<script>-400.woff2` and `-600.woff2`, each the
 * script-only subset from Google Fonts (not the full face — the Latin
 * ranges are already covered by the UI font).
 */
export const SCRIPT_FONTS: Record<string, string> = {
  deva: 'Noto Sans Devanagari',
  beng: 'Noto Sans Bengali',
  gujr: 'Noto Sans Gujarati',
  guru: 'Noto Sans Gurmukhi',
  knda: 'Noto Sans Kannada',
  mlym: 'Noto Sans Malayalam',
  orya: 'Noto Sans Oriya',
  taml: 'Noto Sans Tamil',
  telu: 'Noto Sans Telugu',
  arab: 'Noto Naskh Arabic',
  olck: 'Noto Sans Ol Chiki',
  mtei: 'Noto Sans Meetei Mayek',
  latn: '',
};

/** Scripts written right-to-left. Urdu, Kashmiri and Sindhi all use Arabic. */
const RTL_SCRIPTS = new Set(['arab']);

export function scriptOf(code: string | null | undefined): string {
  if (!code) return 'latn';
  const base = code.split('-')[0].toLowerCase();
  return LANGUAGES[base]?.script ?? 'latn';
}

/**
 * The class to put on any element rendering target-language text.
 * Returns '' for Latin, so English text is never given a pointless class.
 */
export function langClass(code: string | null | undefined): string {
  const s = scriptOf(code);
  return s === 'latn' ? '' : `lang-${s}`;
}

export function isRtl(code: string | null | undefined): boolean {
  return RTL_SCRIPTS.has(scriptOf(code));
}

export function describeLang(code: string | null | undefined): string {
  if (!code) return '—';
  const meta = LANGUAGES[code.split('-')[0].toLowerCase()];
  if (!meta) return code;
  return meta.native && meta.native !== meta.english
    ? `${meta.english} (${meta.native})`
    : meta.english;
}

// ---------------------------------------------------------------------------
// lazy font injection
// ---------------------------------------------------------------------------

const loaded = new Set<string>();

/**
 * Inject the @font-face pair and the `.lang-<script>` rule for one script.
 *
 * Idempotent and safe to call on every render. Until it runs, `.lang-deva`
 * simply has no rule and the text renders in the UI font — degraded, never
 * broken, which is why this can be lazy at all.
 *
 * TODO if the woff2 files are ever missing from `public/fonts/`: the loader
 * still runs and the browser silently falls back. The files expected are
 * `<script>-400.woff2` and `<script>-600.woff2` for each key of
 * SCRIPT_FONTS except `latn`, i.e. deva, beng, gujr, guru, knda, mlym,
 * orya, taml, telu, arab, olck, mtei — the script-subset woff2 from the
 * matching Google Fonts family named above.
 */
export function ensureScriptFont(code: string | null | undefined): void {
  const script = scriptOf(code);
  const family = SCRIPT_FONTS[script];
  if (!family || loaded.has(script)) return;
  loaded.add(script);

  const el = document.createElement('style');
  el.dataset.script = script;
  el.textContent = [400, 600]
    .map(
      (w) => `@font-face{font-family:'${family}';font-style:normal;font-weight:${w};`
        + `font-display:swap;src:url('/fonts/${script}-${w}.woff2') format('woff2')}`,
    )
    .concat([
      `.lang-${script}{font-family:'${family}','Inter',system-ui,sans-serif;`
      + `font-feature-settings:normal}`,
    ])
    .join('\n');
  document.head.appendChild(el);

  // Nudge the 400 weight so the first Indic glyph on screen is not the
  // thing that triggers the download.
  const pre = document.createElement('link');
  pre.rel = 'preload';
  pre.as = 'font';
  pre.type = 'font/woff2';
  pre.crossOrigin = 'anonymous';
  pre.href = `/fonts/${script}-400.woff2`;
  document.head.appendChild(pre);
}
