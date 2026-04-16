import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// base: '/'          → local dev + Docker / Vercel / Netlify / HF Spaces
// base: '/repo-name/' → GitHub Pages only (update before GH Pages deploy)
export default defineConfig({
  plugins: [react()],
  base: '/',
  server: {
    port: 5173,
    strictPort: true,
  },
  preview: {
    // `vite preview` serves the production build locally on this port
    port: 4173,
    strictPort: true,
  },
  build: {
    // Generate a manifest so the Nginx cache-busting headers know hashed filenames
    manifest: true,
    // Raise the chunk-size warning threshold (react-markdown pulls in remark/unified)
    chunkSizeWarningLimit: 700,
  },
})
