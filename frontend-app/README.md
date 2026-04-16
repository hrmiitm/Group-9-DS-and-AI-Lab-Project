# FraudGuard Frontend App

This is the React/Vite frontend for the FraudGuard platform. It provides a dynamic dashboard to extract job details, run 13 different investigative tools, and generate comprehensive fraud assessment reports.

## Local Development

1. **Install Dependencies:**
   ```bash
   cd frontend-app
   npm install
   ```

2. **Run the Development Server:**
   ```bash
   npm run dev
   ```
   The app will be accessible at `http://localhost:5173`.

3. **Connect to Backend:**
   - By default, the frontend points to `http://localhost:8000` for local development.
   - You can change this temporarily in the UI by clicking the **Settings (⚙️)** button and updating the "Backend API URL".

## Deployment (GitHub Pages / Vercel)

### Option A: Deploy to GitHub Pages

1. **Configure Base Path:**
   - Open `vite.config.js`.
   - Update the `base` property to match your GitHub repository name. For example, if your repo is `username/fraudguard-ui`, set it to:
     ```javascript
     export default defineConfig({
       plugins: [react()],
       base: '/fraudguard-ui/', // CHANGE THIS!
     })
     ```

2. **Update Default Backend URL:**
   - Open `src/contexts/SettingsContext.jsx`.
   - Change the `backendUrl` from `http://localhost:8000` to your actual deployed HuggingFace Space URL (e.g., `https://[your-username]-[space-name].hf.space`).

3. **Build the App:**
   ```bash
   npm run build
   ```
   This will generate a `dist` folder.

4. **Deploy:**
   - You can use the `gh-pages` npm package or simply push the contents of the `dist` folder to a `gh-pages` branch on your repository.

### Option B: Deploy to Vercel (Recommended)

Vercel is the easiest way to deploy Vite React apps.

1. **Update Default Backend URL:**
   - Open `src/contexts/SettingsContext.jsx`.
   - Change the `backendUrl` to your deployed HuggingFace Space URL.
   - Ensure the `base` in `vite.config.js` is set to `/` (default).

2. **Deploy via Vercel Dashboard:**
   - Push your code to a GitHub repository.
   - Go to [Vercel](https://vercel.com) and import your repository.
   - Vercel will automatically detect Vite and set the build command to `npm run build` and the output directory to `dist`.
   - Click **Deploy**.

## Features

- **Drag & Drop JD Input:** Supports `.txt`, `.md`, `.pdf`, and `.docx` parsing.
- **Dynamic Tool Grid:** Automatically fetches available tools from the backend registry.
- **Editable Tool Inputs:** Pre-filled from LLM extraction, but user-editable prior to running individual checks.
- **Local Storage Settings:** API keys and model preferences are saved securely in the browser.
