/**
 * components/ToolCard.jsx
 * Individual tool card with:
 *  - Editable input fields (pre-filled from extracted JD)
 *  - Independent "Run" button
 *  - Loading spinner per card
 *  - LLM inference bullet points
 *  - Collapsible raw JSON output
 */
import { useState } from 'react'
import { runTool, llmToolInference } from '../services/api'
import { useSettings } from '../contexts/SettingsContext'
import styles from './ToolCard.module.css'

const STATUS_COLORS = {
  idle:    '',
  loading: styles.statusLoading,
  done:    styles.statusDone,
  error:   styles.statusError,
}

function RiskBadge({ value }) {
  if (!value) return null
  const cls = value === 'high' ? 'badge-high' : value === 'medium' ? 'badge-medium' : 'badge-low'
  return <span className={`badge ${cls}`}>{value} risk</span>
}

function BulletList({ bullets }) {
  if (!bullets?.length) return null
  return (
    <ul className={styles.bullets}>
      {bullets.map((b, i) => <li key={i}>{b}</li>)}
    </ul>
  )
}

export default function ToolCard({ toolName, meta, initialParams, jobDict }) {
  const { settings, getLLMConfig } = useSettings()
  const [params,    setParams]    = useState({ ...initialParams })
  const [status,    setStatus]    = useState('idle')  // idle|loading|done|error
  const [result,    setResult]    = useState(null)
  const [inference, setInference] = useState(null)
  const [infLoading,setInfLoading]= useState(false)
  const [showRaw,   setShowRaw]   = useState(false)
  const [error,     setError]     = useState(null)

  const handleRun = async () => {
    setStatus('loading')
    setError(null)
    setResult(null)
    setInference(null)

    // 1. Run the tool
    let toolRes
    try {
      toolRes = await runTool(settings.backendUrl, toolName, params)
      setResult(toolRes)
      setStatus('done')
    } catch (err) {
      setError(err.message)
      setStatus('error')
      return
    }

    // 2. Get LLM inference automatically
    setInfLoading(true)
    try {
      const infCfg = getLLMConfig('toolInferenceModel')
      const infRes = await llmToolInference(
        settings.backendUrl,
        toolName,
        meta.label,
        toolRes.result || toolRes,
        jobDict || {},
        infCfg
      )
      setInference(infRes)
    } catch (_) {
      // inference is optional — tool result still shown
    }
    setInfLoading(false)
  }

  const riskLevel = result?.result?.data?.risk_level
  const isStub    = meta.is_stub
  const canRun    = !isStub

  return (
    <div className={`${styles.card} ${status === 'done' ? styles.cardDone : status === 'error' ? styles.cardError : ''} fade-in`}>
      {/* ── Header ── */}
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <span className={styles.icon}>{meta.icon}</span>
          <div>
            <div className={styles.label}>{meta.label}</div>
            <div className={styles.category}>{meta.category?.replace('_', ' ')}</div>
          </div>
        </div>
        <div className={styles.headerRight}>
          {status !== 'idle' && (
            <span className={`${styles.statusDot} ${STATUS_COLORS[status]}`}>
              {status === 'loading' ? <span className="spinner" /> :
               status === 'done'    ? '✓' :
               status === 'error'   ? '✗' : ''}
            </span>
          )}
          {riskLevel && <RiskBadge value={riskLevel} />}
        </div>
      </div>

      <p className={styles.description}>{meta.description}</p>

      {/* ── Editable inputs ── */}
      <div className={styles.inputs}>
        {Object.entries(meta.input_schema || {}).map(([key, schema]) => (
          <div key={key} className={styles.inputRow}>
            <label htmlFor={`${toolName}-${key}`}>
              {key.replace(/_/g, ' ')}
              {schema.required && <span className={styles.required}> *</span>}
            </label>
            {key === 'job_text' ? (
              <textarea
                id={`${toolName}-${key}`}
                className="textarea"
                style={{ minHeight: 80, fontSize: 12 }}
                value={params[key] ?? ''}
                onChange={e => setParams(p => ({ ...p, [key]: e.target.value }))}
                placeholder={schema.description}
              />
            ) : (
              <input
                id={`${toolName}-${key}`}
                className="input"
                value={params[key] ?? ''}
                onChange={e => setParams(p => ({ ...p, [key]: e.target.value }))}
                placeholder={schema.description}
              />
            )}
          </div>
        ))}
      </div>

      {/* ── Actions ── */}
      <div className={styles.actions}>
        <button
          id={`run-${toolName}`}
          className="btn btn-primary btn-sm"
          onClick={handleRun}
          disabled={status === 'loading' || !canRun}
          title={isStub ? 'This tool is not yet implemented' : 'Run this tool'}
        >
          {status === 'loading' ? <><span className="spinner" /> Running…</> : '▶ Run'}
        </button>
        {isStub && <span className={styles.stubNote}>Not yet implemented</span>}
      </div>

      {/* ── Error ── */}
      {status === 'error' && error && (
        <div className={styles.errorBox}>⚠ {error}</div>
      )}

      {/* ── LLM Inference bullets ── */}
      {(infLoading || inference) && (
        <div className={styles.inferenceBox}>
          <div className={styles.inferenceHeader}>
            <span>🧠 Analysis</span>
            {infLoading && <span className="spinner" />}
          </div>
          {inference?.bullets?.length > 0 && (
            <BulletList bullets={inference.bullets} />
          )}
          {inference?.ok === false && (
            <p className={styles.infError}>LLM inference unavailable</p>
          )}
        </div>
      )}

      {/* ── Raw result ── */}
      {result && (
        <div className={styles.rawSection}>
          <button
            className={`btn btn-ghost btn-sm ${styles.rawToggle}`}
            onClick={() => setShowRaw(v => !v)}
          >
            {showRaw ? '▲ Hide' : '▼ Raw JSON'}
          </button>
          {showRaw && (
            <pre className="code-block">{JSON.stringify(result, null, 2)}</pre>
          )}
        </div>
      )}
    </div>
  )
}
