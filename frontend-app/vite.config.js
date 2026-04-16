import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Change base to your repo name for GitHub Pages deployment
// e.g. base: '/Group-9-DS-and-AI-Lab-Project/'
export default defineConfig({
  plugins: [react()],
  base: '/',   // update to '/your-repo-name/' when deploying to GH Pages
})
