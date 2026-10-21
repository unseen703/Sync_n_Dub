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
      // `#/job/<id>/edit` is the agent page for that job. Matched BEFORE
      // the plain job route, because `(.+)` would otherwise swallow the
      // `/edit` suffix into the id — the same greedy-match mistake the
      // server made with its `{job_id:path}` converter.
      // `#/new` is kept as an ALIAS for home: the new-dub form moved onto
      // the landing page, and the old link should not 'do nothing'.
      if (location.hash === '#/new') {
        location.replace('#/');
        return;
      }
      // `#/talking-head` is a standalone page with no job behind it, so it
      // clears the selection the way `#/` does — otherwise `bootstrap()`'s
      // auto-selected job would keep rendering underneath it.
      if (location.hash === '#/talking-head') {
        useStore.getState().setTalkingHead(true);
        useStore.getState().setEditing(false);
        if (useStore.getState().selected) useStore.getState().selectJob(null);
        return;
      }
      useStore.getState().setTalkingHead(false);
      const edit = location.hash.match(/^#\/job\/(.+)\/edit$/);
      const m = edit ?? location.hash.match(/^#\/job\/(.+)$/);
      const id = m ? decodeURIComponent(m[1]) : null;
      useStore.getState().setEditing(Boolean(edit));
      if (id) {
        if (id !== useStore.getState().selected) useStore.getState().selectJob(id);
      } else if (useStore.getState().selected) {
        // `#/` is HOME, and it has to clear the selection to get there.
        // The wordmark has always linked here, but this handler only ever
        // *selected* — it never deselected — so clicking the logo changed
        // the hash and nothing else, and the landing page was unreachable
        // once `bootstrap()` had auto-selected the newest job (which it
        // does on every load).
        useStore.getState().selectJob(null);
      }
    };
    onHash();
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);
}
