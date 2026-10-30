

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

        line: ink('--c-line'),

        edge: ink('--c-edge'),

        ink: ink('--c-ink'),
        ink2: ink('--c-ink-2'),
        ink3: ink('--c-ink-3'),
        ink4: ink('--c-ink-4'),

        brand: {
          DEFAULT: ink('--c-brand'),

          fill: ink('--c-brand'),
          soft: ink('--c-brand-soft'),

          ink: ink('--c-brand-ink'),
        },
        aqua: {
          DEFAULT: ink('--c-aqua'),
          ink: ink('--c-aqua-ink'),
        },

        blue: {
          ink: ink('--c-blue-ink'),
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
