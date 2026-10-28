import { useEffect, useState } from 'react';
import * as shared from '../lib/audio';

/** Subscribes to the single shared preview element. Returns the src that is
 *  currently playing, or null. */
export function useAudioPreview(): string | null {
  const [src, setSrc] = useState<string | null>(() => shared.playing());
  useEffect(() => shared.subscribe(setSrc), []);
  return src;
}

export const togglePreview = shared.toggle;
export const stopPreview = shared.stop;
