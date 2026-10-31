import { useEffect, useState } from 'react';
import * as shared from '../lib/audio';

/** Returns the source currently playing in the shared audio preview, or `null`. */
export function useAudioPreview(): string | null {
  const [src, setSrc] = useState<string | null>(() => shared.playing());
  useEffect(() => shared.subscribe(setSrc), []);
  return src;
}

export const togglePreview = shared.toggle;
export const stopPreview = shared.stop;
