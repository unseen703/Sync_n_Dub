import { useEffect } from 'react';
import { useStore } from '../store/useStore';

/** Syncs the hash route with selection state. It reads the URL hash and updates the shared store. */
export function useHashRoute(): void {
  useEffect(() => {
    const onHash = () => {

      if (location.hash === '#/new') {
        location.replace('#/');
        return;
      }

      const legacyEdit = location.hash.match(/^#\/job\/(.+)\/edit$/);
      if (legacyEdit) {
        location.replace(`#/job/${legacyEdit[1]}`);
        return;
      }

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

        useStore.getState().selectJob(null);
      }
    };
    onHash();
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);
}
