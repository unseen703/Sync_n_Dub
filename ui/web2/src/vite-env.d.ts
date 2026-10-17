/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** 'all' | 'writes' | 'off' — see src/api/index.ts */
  readonly VITE_MOCK?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

interface Window {
  /** Set by the inline <script> in index.html, awaited by api.listJobs(). */
  __BOOT__?: Promise<unknown>;
}
