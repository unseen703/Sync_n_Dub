import clsx, { type ClassValue } from 'clsx';

/** Class joiner. No tailwind-merge: 1.5 kB to fix conflicts we can avoid
 *  by not writing them. */
export const cn = (...parts: ClassValue[]) => clsx(parts);
