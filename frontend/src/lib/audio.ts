/** Shared audio-preview controller. Inputs identify an audio source; outputs report or update the active preview. */
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

export function element(): HTMLAudioElement | null {
  return el;
}

export function seek(src: string, fraction: number): void {
  if (!el || current !== src || !isFinite(el.duration)) return;
  el.currentTime = Math.max(0, Math.min(1, fraction)) * el.duration;
}

export function subscribe(fn: Listener): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}
