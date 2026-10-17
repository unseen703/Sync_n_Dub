import { useEffect } from 'react';
import { useStore } from '../store/useStore';

/**
 * `#/job/<id>` is the whole router.
 *
 * The hash never reaches the server, so `ui/server.py` needs no SPA
 * catch-all route and the StaticFiles mount at `/` keeps working exactly as
 * it does today. react-router would be ~25 kB for this.
 */
export function useHashRoute(): void {
  useEffect(() => {
    const onHash = () => {
      const m = location.hash.match(/^#\/job\/(.+)$/);
      const id = m ? decodeURIComponent(m[1]) : null;
      if (id && id !== useStore.getState().selected) {
        useStore.getState().selectJob(id);
      }
    };
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);
}
