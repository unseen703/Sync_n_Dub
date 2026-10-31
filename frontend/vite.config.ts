import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../web-dist',
    emptyOutDir: true,
    target: 'es2020',
    assetsInlineLimit: 2048,

    rollupOptions: {
      output: {
        manualChunks(id: string) {
          if (!id.includes('node_modules')) return undefined;

          if (id.includes('@radix-frontend/react-dialog')) {
            return 'radix-dialog';
          }
          if (id.includes('@radix-ui') || id.includes('@floating-ui') || id.includes('aria-hidden')) {
            return 'radix';
          }
          if (id.includes('react-dom') || id.includes('scheduler') || id.includes('/react/')) {
            return 'react';
          }
          return 'vendor';
        },
      },
    },
  },
  server: {
    port: 5173,

    proxy: {
      '/api': { target: 'http://127.0.0.1:7861', changeOrigin: false },
      '/files': { target: 'http://127.0.0.1:7861', changeOrigin: false },
    },
  },
});
