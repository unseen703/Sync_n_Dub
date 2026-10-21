import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Build output lands in `ui/web-dist/`. `ui/server.py` already prefers that
// directory over the legacy `ui/web/` inspector whenever `index.html` exists
// there, so a broken build falls back instead of taking the UI down. Do not
// change the path without changing WEB_DIST in server.py.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../web-dist',
    emptyOutDir: true,
    target: 'es2020',
    assetsInlineLimit: 2048,
    // Four chunks so React and Radix stay cached across app deploys, and so
    // the eager budget in the plan (~115 kB gz) is measurable per chunk in
    // DevTools rather than being one opaque blob.
    rollupOptions: {
      output: {
        manualChunks(id: string) {
          if (!id.includes('node_modules')) return undefined;
          // framer-motion is deliberately NOT grouped. Forcing it into one
          // named chunk defeats the LazyMotion split: `App.tsx` statically
          // imports LazyMotion/MotionConfig, so the whole group — including
          // the animation features we go to the trouble of loading lazily —
          // gets modulepreloaded into the critical path. Left alone, Rollup
          // puts the small static part with the app and the feature bundle
          // in its own async chunk.
          // react-dialog is reached ONLY from lazily-imported screens (the
          // new-dub form, the apply-cost dialog, the shortcut list). Left in
          // the shared `radix` chunk it rides along on first paint anyway,
          // because Tooltip/Select/Popover pull that chunk eagerly — so it
          // gets its own, and the ~4 kB gz leaves the critical path. Its
          // shared primitives (dismissable-layer, focus-scope, portal) stay
          // in `radix`, which popover needs regardless.
          if (id.includes('@radix-ui/react-dialog')) {
            return 'radix-dialog';
          }
          if (id.includes('@radix-ui') || id.includes('@floating-ui') || id.includes('aria-hidden')) {
            return 'radix';
          }
          if (id.includes('react-dom') || id.includes('scheduler') || id.includes('/react/')) {
            return 'react';
          }
          return 'vendor';
        },
      },
    },
  },
  server: {
    port: 5173,
    // Dev runs against the live FastAPI server so the mock is opt-in, not
    // the default. `npm run dev:mock` sets VITE_MOCK=1 for the endpoints
    // that do not exist yet.
    proxy: {
      '/api': { target: 'http://127.0.0.1:7861', changeOrigin: false },
      '/files': { target: 'http://127.0.0.1:7861', changeOrigin: false },
    },
  },
});
