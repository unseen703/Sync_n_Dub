
/** Language metadata helpers. Inputs are language codes; outputs are labels, CSS classes, or loaded script fonts. */
export interface LangMeta {
  english: string;
  native: string;
  script: string;
}

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

const RTL_SCRIPTS = new Set(['arab']);

export function scriptOf(code: string | null | undefined): string {
  if (!code) return 'latn';
  const base = code.split('-')[0].toLowerCase();
  return LANGUAGES[base]?.script ?? 'latn';
}

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

const loaded = new Set<string>();

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

  const pre = document.createElement('link');
  pre.rel = 'preload';
  pre.as = 'font';
  pre.type = 'font/woff2';
  pre.crossOrigin = 'anonymous';
  pre.href = `/fonts/${script}-400.woff2`;
  document.head.appendChild(pre);
}
