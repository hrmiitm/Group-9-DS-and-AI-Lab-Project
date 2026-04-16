/**
 * components/SettingsModal.jsx
 * Modal for configuring LLM API keys, models, base URL.
 * All values persist to localStorage via SettingsContext.
 */
import { useState } from 'react'
import { useSettings } from '../contexts/SettingsContext'
import styles from './SettingsModal.module.css'

export default function SettingsModal({ onClose }) {
  const { settings, updateSettings, resetSettings } = useSettings()
  const [local, setLocal] = useState({ ...settings })
  const [saved, setSaved]  = useState(false)

  const set = (key, val) => setLocal(prev => ({ ...prev, [key]: val }))

  const handleSave = () => {
    updateSettings(local)
    setSaved(true)
    setTimeout(() => { setSaved(false); onClose() }, 800)
  }

  return (
    <div className={styles.overlay} onClick={e => e.target === e.currentTarget && onClose()}>
      <div className={`${styles.modal} fade-in`} role="dialog" aria-modal="true" aria-label="API Settings">
        <div className={styles.modalHeader}>
          <h2 className={styles.title}>⚙️ API Settings</h2>
          <button className="btn btn-ghost btn-sm" onClick={onClose} aria-label="Close">✕</button>
        </div>

        <div className={styles.body}>
          <p className={styles.hint}>
            Settings are saved in your browser. If the backend has env vars set, these are optional
            (only needed when the server has no API key configured).
          </p>

          <section className={styles.section}>
            <h3 className={styles.sectionTitle}>Backend</h3>
            <div className={styles.field}>
              <label htmlFor="s-backend-url">Backend API URL</label>
              <input id="s-backend-url" className="input" value={local.backendUrl}
                onChange={e => set('backendUrl', e.target.value)}
                placeholder="https://your-space.hf.space" />
            </div>
          </section>

          <section className={styles.section}>
            <h3 className={styles.sectionTitle}>LLM API</h3>
            <div className={styles.field}>
              <label htmlFor="s-api-key">API Key</label>
              <input id="s-api-key" className="input" type="password" value={local.apiKey}
                onChange={e => set('apiKey', e.target.value)}
                placeholder="sk-... or aipipe token" />
            </div>
            <div className={styles.field}>
              <label htmlFor="s-base-url">Base URL</label>
              <input id="s-base-url" className="input" value={local.llmBaseUrl}
                onChange={e => set('llmBaseUrl', e.target.value)}
                placeholder="https://aipipe.org/openrouter/v1" />
            </div>
          </section>

          <section className={styles.section}>
            <h3 className={styles.sectionTitle}>Models</h3>
            <div className={styles.grid2}>
              <div className={styles.field}>
                <label htmlFor="s-extract-model">Extract JD Model</label>
                <input id="s-extract-model" className="input" value={local.extractModel}
                  onChange={e => set('extractModel', e.target.value)}
                  placeholder="openai/gpt-4.1-mini" />
              </div>
              <div className={styles.field}>
                <label htmlFor="s-research-model">Deep Research Model</label>
                <input id="s-research-model" className="input" value={local.deepResearchModel}
                  onChange={e => set('deepResearchModel', e.target.value)}
                  placeholder="openai/gpt-4.1-mini" />
              </div>
              <div className={styles.field}>
                <label htmlFor="s-inference-model">Tool Inference Model</label>
                <input id="s-inference-model" className="input" value={local.toolInferenceModel}
                  onChange={e => set('toolInferenceModel', e.target.value)}
                  placeholder="openai/gpt-4.1-mini" />
              </div>
              <div className={styles.field}>
                <label htmlFor="s-summary-model">Final Summary Model</label>
                <input id="s-summary-model" className="input" value={local.finalSummaryModel}
                  onChange={e => set('finalSummaryModel', e.target.value)}
                  placeholder="openai/gpt-4.1-mini" />
              </div>
            </div>
          </section>
        </div>

        <div className={styles.footer}>
          <button className="btn btn-ghost btn-sm" onClick={() => { resetSettings(); setLocal({...settings}) }}>
            Reset to defaults
          </button>
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn btn-secondary" onClick={onClose}>Cancel</button>
            <button id="settings-save-btn" className="btn btn-primary" onClick={handleSave}>
              {saved ? '✓ Saved!' : 'Save Settings'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
