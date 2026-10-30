import { lazy, Suspense, useEffect } from 'react';
import { LazyMotion, MotionConfig } from 'framer-motion';
import { TooltipProvider } from './components/ui/Tooltip';
import { AppShell } from './components/layout/AppShell';
import { useStore } from './store/useStore';
import { useHashRoute } from './hooks/useHashRoute';
import { useJobProgress, useJobsPoll } from './hooks/useJobProgress';
import { useShortcuts } from './hooks/useShortcuts';
import { anyMocked } from './api';

const loadMotionFeatures = () =>
  import('./lib/motionFeatures').then((mod) => mod.default);

const ShortcutsDialog = lazy(() =>
  import('./components/ui/ShortcutsDialog').then((m) => ({ default: m.ShortcutsDialog })),
);

/** Root application shell. It reads route and store state, then renders the active page. */
export default function App() {
  const bootstrap = useStore((s) => s.bootstrap);
  const selected = useStore((s) => s.selected);
  const shortcutsOpen = useStore((s) => s.shortcutsOpen);
  const setShortcutsOpen = useStore((s) => s.setShortcutsOpen);

  useEffect(() => {
    void bootstrap();
  }, [bootstrap]);

  useHashRoute();
  useShortcuts();
  useJobsPoll();
  useJobProgress(selected);

  return (

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
