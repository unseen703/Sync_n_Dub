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
      // `#/new` is kept as an ALIAS for home: the new-dub form moved onto
      // the landing page, and the old link should not 'do nothing'.
      if (location.hash === '#/new') {
        location.replace('#/');
        return;
      }
      // `#/job/<id>/edit` was the agent's own page until 2026-10-21. The
      // agent now lives on the job page itself, below the video, so this
      // REDIRECTS rather than 404s — an old bookmark or a link in a chat
      // still lands somewhere, and lands on the thing it was pointing at.
      //
      // Matched BEFORE the plain job route, because `(.+)` would otherwise
      // swallow the `/edit` suffix into the id — the same greedy-match
      // mistake the server made with its `{job_id:path}` converter, which
      // is how a job called `<id>/edit` would be looked up and not found.
      const legacyEdit = location.hash.match(/^#\/job\/(.+)\/edit$/);
      if (legacyEdit) {
        location.replace(`#/job/${legacyEdit[1]}`);
        return;
      }
      // `#/talking-head` is a standalone page with no job behind it, so it
      // clears the selection the way `#/` does — otherwise `bootstrap()`'s
      // auto-selected job would keep rendering underneath it.
      if (location.hash === '#/talking-head') {
        useStore.getState().setTalkingHead(true);
        if (useStore.getState().selected) useStore.getState().selectJob(null);
        return;
      }
      useStore.getState().setTalkingHead(false);
      const m = location.hash.match(/^#\/job\/(.+)$/);
      const id = m ? decodeURIComponent(m[1]) : null;
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
