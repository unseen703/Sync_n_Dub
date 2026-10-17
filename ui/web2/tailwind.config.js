/** @type {import('tailwindcss').Config} */

/*
  One palette, defined once as RGB triplets in `src/index.css` and surfaced
  here as semantic names. Nothing in `src/` is allowed to reach for a raw
  Tailwind grey again — `text-slate-600` on a near-black canvas measured
  2.4:1, which is why the old UI was hard to read.

  The four ink steps are the whole contrast story. Measured against the three
  background tokens (see `npm run contrast`):

      token    hex       canvas   card   popover
      ink      #F2F5F9    17.78   16.26    14.73
      ink2     #C7D1DF    12.61   11.53    10.44
      ink3     #9FACBF     8.45    7.73     7.00
      ink4     #7E8CA0     5.69    5.20     4.71

  The DIMMEST text token still clears WCAG AA (4.5:1) on the LIGHTEST
  surface, so there is no combination of ink and background in this app that
  fails AA. That is the property worth protecting; keep it when editing, and
  re-check with `npm run contrast`, which reads the live tokens and exits
  non-zero if any pair slips.
*/
const ink = (v) => `rgb(var(${v}) / <alpha-value>)`;

export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        canvas: ink('--c-canvas'),
        surface: ink('--c-surface'),
        raised: ink('--c-raised'),

        /** Decorative hairline. Never carries meaning, never carries text. */
        line: ink('--c-line'),
        /** Interactive control boundary — >=3.3:1 everywhere (WCAG 1.4.11). */
        edge: ink('--c-edge'),

        ink: ink('--c-ink'),
        ink2: ink('--c-ink-2'),
        ink3: ink('--c-ink-3'),
        ink4: ink('--c-ink-4'),

        brand: {
          DEFAULT: ink('--c-brand'),
          /** Solid fill that carries WHITE text at 6.3:1. */
          fill: ink('--c-brand'),
          soft: ink('--c-brand-soft'),
          /** Brand colour as TEXT on a dark background — 9.8:1. */
          ink: ink('--c-brand-ink'),
        },
        aqua: {
          DEFAULT: ink('--c-aqua'),
          ink: ink('--c-aqua-ink'),
        },
        ok: ink('--c-ok'),
        warn: ink('--c-warn'),
        danger: {
          DEFAULT: ink('--c-danger'),
          fill: ink('--c-danger-fill'),
        },
      },
      fontFamily: {
        sans: [
          'Inter var',
          'Inter',
          'system-ui',
          '-apple-system',
          'Segoe UI',
          'Roboto',
          'Helvetica Neue',
          'Arial',
          'sans-serif',
        ],
        mono: [
          'ui-monospace',
          'SFMono-Regular',
          'Menlo',
          'Consolas',
          'Noto Sans Mono',
          'monospace',
        ],
      },
      borderRadius: {
        '4xl': '1.75rem',
      },
      keyframes: {
        'fade-in': { from: { opacity: '0' }, to: { opacity: '1' } },
        'slide-up': {
          from: { opacity: '0', transform: 'translateY(6px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        shimmer: { '100%': { transform: 'translateX(100%)' } },
        'pulse-soft': {
          '0%,100%': { opacity: '1' },
          '50%': { opacity: '0.45' },
        },
      },
      animation: {
        'fade-in': 'fade-in 140ms ease-out',
        'slide-up': 'slide-up 160ms cubic-bezier(0.22,1,0.36,1)',
        'pulse-soft': 'pulse-soft 1.6s ease-in-out infinite',
      },
    },
  },
  plugins: [],
};
