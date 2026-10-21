import { lazy, Suspense, useEffect } from 'react';
import { LazyMotion, MotionConfig } from 'framer-motion';
import { TooltipProvider } from './components/ui/Tooltip';
import { AppShell } from './components/layout/AppShell';
import { useStore } from './store/useStore';
import { useHashRoute } from './hooks/useHashRoute';
import { useJobProgress, useJobsPoll } from './hooks/useJobProgress';
import { useShortcuts } from './hooks/useShortcuts';
import { anyMocked } from './api';

/** Fetched after first paint — see `lib/motionFeatures.ts`. */
const loadMotionFeatures = () =>
  import('./lib/motionFeatures').then((mod) => mod.default);

/**
 * The shortcuts dialog is loaded on demand — it is not on the path to
 * first paint. The new-dub form is no longer a dialog at all: it lives on
 * the landing page, so there is one place to start a dub instead of two
 * that drifted apart.
 */
const ShortcutsDialog = lazy(() =>
  import('./components/ui/ShortcutsDialog').then((m) => ({ default: m.ShortcutsDialog })),
);

export default function App() {
  const bootstrap = useStore((s) => s.bootstrap);
  const selected = useStore((s) => s.selected);
  const specOpen = useStore((s) => s.specOpen);
  const shortcutsOpen = useStore((s) => s.shortcutsOpen);
  const setShortcutsOpen = useStore((s) => s.setShortcutsOpen);

  useEffect(() => {
    void bootstrap();
  }, [bootstrap]);

  // Keep the <html> class the inline boot script set in step with the store,
  // so the skeleton's column count and the app's agree.
  useEffect(() => {
    document.documentElement.classList.toggle('spec-open', specOpen);
  }, [specOpen]);

  useHashRoute();
  useShortcuts();
  useJobsPoll();
  useJobProgress(selected);

  return (
    // `reducedMotion="user"` once, at the root: every animation in the app
    // respects prefers-reduced-motion without each component remembering to.
    // `LazyMotion` + `m.*` instead of `motion.*`, so only the animation
    // FEATURES move to an async chunk. `strict` makes a stray `motion.x`
    // throw rather than silently linking the full bundle back in, which is
    // the only way this stays true.
    <LazyMotion features={loadMotionFeatures} strict>
      <MotionConfig reducedMotion="user">
        <TooltipProvider delayDuration={300} skipDelayDuration={200}>
          <AppShell />
          {shortcutsOpen ? (
            <Suspense fallback={null}>
              <ShortcutsDialog open onOpenChange={setShortcutsOpen} />
            </Suspense>
          ) : null}
          {anyMocked ? <MockBanner /> : null}
        </TooltipProvider>
      </MotionConfig>
    </LazyMotion>
  );
}

/**
 * A permanent, unmissable statement that some endpoints are stubs.
 *
 * This is not decoration. The alternative is a UI that accepts a new job
 * and reports success while nothing was queued, which is exactly the
 * "reported on what it meant to do" failure the repo's guardrails exist to
 * prevent. `VITE_MOCK` defaults to `off`, so it never shows in a real build.
 */
function MockBanner() {
  return (
    <div
      className="pointer-events-none fixed left-1/2 top-2 z-[80] -translate-x-1/2 rounded-full
                 border border-warn/45 bg-raised px-3 py-1 text-[11px]
                 font-semibold uppercase tracking-wider text-warn"
    >
      Mock mode — submit / cancel / apply do nothing
    </div>
  );
}
