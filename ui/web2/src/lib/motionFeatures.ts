/**
 * framer-motion's DOM feature set, in its own module so it can be loaded
 * AFTER first paint.
 *
 * `<LazyMotion features={domAnimation}>` with a direct import links the
 * feature bundle into the eager graph. Handing `LazyMotion` a function that
 * returns a dynamic import moves it to its own chunk, fetched once the app
 * is interactive. Nothing on screen needs it for the first frame — the
 * skeleton and the shell are plain CSS.
 *
 * This only works while `vite.config.ts` does NOT group framer-motion into
 * a manual chunk; grouping it re-links the whole thing into the preloaded
 * critical path. There is a comment there saying so.
 */
import { domAnimation } from 'framer-motion';

export default domAnimation;
