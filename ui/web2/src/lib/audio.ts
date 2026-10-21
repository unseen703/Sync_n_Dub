/**
 * One `HTMLAudioElement` for every preview in the app.
 *
 * The voice picker can list 44 voices and the speaker table one reference
 * clip per speaker. Giving each row its own element means 44 media elements
 * competing for decoders, and means two previews can play at once — which,
 * when you are trying to tell two voices apart, is the one thing that must
 * not happen.
 */
type Listener = (src: string | null) => void;

let el: HTMLAudioElement | null = null;
let current: string | null = null;
const listeners = new Set<Listener>();

function node(): HTMLAudioElement {
  if (!el) {
    el = new Audio();
    el.preload = 'none';
    el.addEventListener('ended', () => set(null));
    el.addEventListener('error', () => set(null));
    el.addEventListener('pause', () => {
      if (el && el.currentTime === 0) set(null);
    });
  }
  return el;
}

function set(src: string | null) {
  current = src;
  listeners.forEach((fn) => fn(src));
}

/** Play `src`, or stop if it is already the one playing. Returns the new state. */
export function toggle(src: string | null | undefined): boolean {
  if (!src) return false;
  const a = node();
  if (current === src && !a.paused) {
    a.pause();
    a.currentTime = 0;
    set(null);
    return false;
  }
  a.pause();
  a.src = src;
  a.currentTime = 0;
  set(src);
  void a.play().catch(() => set(null));
  return true;
}

export function stop(): void {
  if (!el) return;
  el.pause();
  el.currentTime = 0;
  set(null);
}

export function playing(): string | null {
  return current;
}

/** The shared element, for callers that need `currentTime`/`duration`.
 *  Returns null before the first play so nothing is constructed eagerly. */
export function element(): HTMLAudioElement | null {
  return el;
}

/** Seek within whatever is currently playing. No-op otherwise. */
export function seek(src: string, fraction: number): void {
  if (!el || current !== src || !isFinite(el.duration)) return;
  el.currentTime = Math.max(0, Math.min(1, fraction)) * el.duration;
}

export function subscribe(fn: Listener): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}
