/**
 * Contrast checker for the palette in `src/index.css`.
 *
 *     conda run -n dubui npm --prefix frontend/ run contrast
 *
 * (Never bare `npm` on this host — system node is v12 and the shim dies with
 * `Cannot find module 'node:path'`. See CLAUDE.md §1.)
 *
 * It reads the `--c-*` RGB triplets straight out of the stylesheet, so it
 * checks what the app actually ships rather than a table someone typed. The
 * backgrounds it tests are the real composites: the canvas, the canvas under
 * the brightest point of the body's brand-tinted radial gradient, the
 * `.card` surface at its 70 % opacity over that, and the opaque raised
 * surface used by popovers and dialogs.
 *
 * Exits non-zero when any text token drops below WCAG AA (4.5:1) on any of
 * them, or when the control-boundary token drops below the 3:1 that WCAG
 * 1.4.11 asks of a UI component's edge. The property worth protecting is
 * that there is NO combination of ink and background in this app that fails
 * AA — the theme this replaced had #475569 body text on #0b0f17, which is
 * 2.4:1.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const css = readFileSync(join(here, '..', 'src', 'index.css'), 'utf8');

const tokens = {};
for (const m of css.matchAll(/(--c-[\w-]+):\s*([\d ]+);/g)) {
  tokens[m[1]] = m[2].trim().split(/\s+/).map(Number);
}

const lin = (c) => {
  const v = c / 255;
  return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
};
const L = ([r, g, b]) => 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
const ratio = (a, b) => {
  const [hi, lo] = [L(a), L(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
};
const over = (fg, bg, a) => fg.map((c, i) => Math.round(c * a + bg[i] * (1 - a)));

const canvas = tokens['--c-canvas'];
// The body gradient's brightest point: rgb(79 70 229 / .16) over the canvas.
const glow = over([79, 70, 229], canvas, 0.16);

const BACKGROUNDS = {
  canvas,
  'canvas + glow': glow,
  'card (surface/70)': over(tokens['--c-surface'], glow, 0.7),
  'raised (popover)': tokens['--c-raised'],
};

const INK = [
  '--c-ink', '--c-ink-2', '--c-ink-3', '--c-ink-4',
  '--c-brand-ink', '--c-aqua-ink', '--c-blue-ink',
  '--c-ok', '--c-warn', '--c-danger',
];

const AA = 4.5;
const UI = 3.0;
const failures = [];
let worst = Infinity;

const pad = (s, n) => String(s).padEnd(n);
const col = (s, n = 20) => String(s).padStart(n);

console.log(pad('', 16) + Object.keys(BACKGROUNDS).map((b) => col(b)).join(''));
for (const name of INK) {
  let row = pad(name.replace('--c-', ''), 16);
  for (const [bn, bg] of Object.entries(BACKGROUNDS)) {
    const r = ratio(tokens[name], bg);
    worst = Math.min(worst, r);
    if (r < AA) failures.push(`${name} on ${bn}: ${r.toFixed(2)} (needs ${AA})`);
    row += col(r.toFixed(2));
  }
  console.log(row);
}

console.log('\ncontrol boundary (WCAG 1.4.11, needs 3:1)');
for (const [bn, bg] of Object.entries(BACKGROUNDS)) {
  const r = ratio(tokens['--c-edge'], bg);
  if (r < UI) failures.push(`--c-edge on ${bn}: ${r.toFixed(2)} (needs ${UI})`);
  console.log(`  edge on ${pad(bn, 20)} ${r.toFixed(2)}`);
}

console.log('\nsolid fills carrying white text');
for (const name of ['--c-brand', '--c-danger-fill']) {
  const r = ratio([255, 255, 255], tokens[name]);
  if (r < AA) failures.push(`white on ${name}: ${r.toFixed(2)} (needs ${AA})`);
  console.log(`  white on ${pad(name.replace('--c-', ''), 18)} ${r.toFixed(2)}`);
}

console.log(`\nworst text ratio anywhere: ${worst.toFixed(2)}:1`);
if (failures.length) {
  console.error('\nFAIL');
  for (const f of failures) console.error('  ' + f);
  process.exit(1);
}
console.log('OK — every ink token clears AA on every surface.');
