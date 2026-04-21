import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

<<<<<<< HEAD
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
=======
// Change base to your repo name for GitHub Pages deployment
// e.g. base: '/Group-9-DS-and-AI-Lab-Project/'
export default defineConfig({
  plugins: [react()],
  base: '/',   // update to '/your-repo-name/' when deploying to GH Pages
>>>>>>> 6cc04f6 (Restructuring project files and adding backend-api)
})
