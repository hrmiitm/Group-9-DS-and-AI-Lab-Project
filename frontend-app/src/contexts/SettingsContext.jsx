/**
 * contexts/SettingsContext.jsx
 * Stores all user-configurable settings in localStorage.
 * Provides a hook for any component to read/write settings.
 */
import { createContext, useContext, useState, useCallback } from 'react'

const STORAGE_KEY = 'fraudguard_settings'

// const DEFAULTS = {
//   backendUrl:            'https://hrmhrmhrm-company-backend-api.hf.space',  // change to HF Space URL after deployment
//   apiKey:                '',
//   extractModel:          'gpt-4.1-mini',
//   deepResearchModel:     'gpt-4.1-mini',
//   toolInferenceModel:    'gpt-4.1-mini',
//   finalSummaryModel:     'gpt-4.1-mini',
//   llmBaseUrl:            'https://aipipe.org/openai/v1',
// }

const DEFAULTS = {
  backendUrl:            'https://huggingface.co/spaces/arun-1988/Fraud_Guard',  // change to HF Space URL after deployment
  apiKey:                '',
  extractModel:          'gpt-4.1-nano',
  deepResearchModel:     'gpt-4.1-nano',
  toolInferenceModel:    'gpt-4.1-nano',
  finalSummaryModel:     'gpt-4.1-nano',
  llmBaseUrl:            'https://aipipe.org/openai/v1',
}

function loadSettings() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) return { ...DEFAULTS, ...JSON.parse(raw) }
  } catch (_) {}
  return { ...DEFAULTS }
}

const SettingsContext = createContext(null)

export function SettingsProvider({ children }) {
  const [settings, setSettings] = useState(loadSettings)

  const updateSettings = useCallback((patch) => {
    setSettings(prev => {
      const next = { ...prev, ...patch }
      try { localStorage.setItem(STORAGE_KEY, JSON.stringify(next)) } catch (_) {}
      return next
    })
  }, [])

  const resetSettings = useCallback(() => {
    setSettings({ ...DEFAULTS })
    try { localStorage.removeItem(STORAGE_KEY) } catch (_) {}
  }, [])

  /** Build the llm_config block sent with every LLM request */
  const getLLMConfig = useCallback((modelKey = 'extractModel') => {
    return settings.apiKey ? {
      api_key:  settings.apiKey,
      base_url: settings.llmBaseUrl,
      model:    settings[modelKey] || DEFAULTS[modelKey],
    } : null
  }, [settings])

  return (
    <SettingsContext.Provider value={{ settings, updateSettings, resetSettings, getLLMConfig }}>
      {children}
    </SettingsContext.Provider>
  )
}

export function useSettings() {
  const ctx = useContext(SettingsContext)
  if (!ctx) throw new Error('useSettings must be used inside SettingsProvider')
  return ctx
}
