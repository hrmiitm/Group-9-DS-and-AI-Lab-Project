/**
 * App.jsx
 * Root component — orchestrates the entire analysis pipeline:
 *   1. Fetch tool registry
 *   2. Extract JD via LLM
 *   3. Deep research missing fields
 *   4. Run all tools concurrently (with progress tracking)
 *   5. Run LLM inference per tool
 *   6. Generate final summary + verdict
 *
 * All pipeline state lives here so components are pure presentational.
 */
import { useState, useCallback, useEffect } from 'react'
import { SettingsProvider, useSettings } from './contexts/SettingsContext'
import Header from './components/Header'
import JDInput from './components/JDInput'
import PipelineProgress from './components/PipelineProgress'
import ExtractedInfo from './components/ExtractedInfo'
import ToolGrid from './components/ToolGrid'
import FinalReport from './components/FinalReport'
import {
  fetchTools,
  llmExtract,
  llmDeepResearch,
  runTool,
  llmToolInference,
  llmFinalSummary,
  buildToolDefaults,
} from './services/api'
import styles from './App.module.css'

// ── Inner app (needs SettingsContext) ───────────────────────────────────────
function AppInner() {
  const { settings, getLLMConfig } = useSettings()

  // ── state ──────────────────────────────────────────────────────────────────
  const [tools,        setTools]        = useState({})
  const [toolsError,   setToolsError]   = useState(null)
  const [toolsLoading, setToolsLoading] = useState(false)

  // Pipeline
  const [pipelineStep, setPipelineStep] = useState(null)  // null|'extract'|'research'|'tools'|'inference'|'report'
  const [toolProgress, setToolProgress] = useState(null)  // {done, total, currentTool}

  // Results
  const [jobDict,       setJobDict]      = useState(null)
  const [deepResearch,  setDeepResearch] = useState(null)
  const [toolParams,    setToolParams]   = useState({})   // pre-filled params for each card
  const [toolResults,   setToolResults]  = useState({})
  const [toolInferences,setToolInferences] = useState({})
  const [finalVerdict,  setFinalVerdict] = useState(null)
  const [finalReport,   setFinalReport]  = useState(null)
  const [reportLoading, setReportLoading]= useState(false)

  const [globalError,  setGlobalError]  = useState(null)
  const [rawText,      setRawText]      = useState('')

  const isAnalysing = pipelineStep !== null

  // ── fetch tool registry on mount ───────────────────────────────────────────
  useEffect(() => {
    loadTools()
  }, [settings.backendUrl])

  async function loadTools() {
    setToolsLoading(true)
    setToolsError(null)
    try {
      const res = await fetchTools(settings.backendUrl)
      setTools(res.tools || {})
    } catch (err) {
      setToolsError(`Cannot reach backend: ${err.message}. Check Settings → Backend URL.`)
    }
    setToolsLoading(false)
  }

  // ── Reset everything for a new analysis ────────────────────────────────────
  function resetAnalysis() {
    setPipelineStep(null)
    setToolProgress(null)
    setJobDict(null)
    setDeepResearch(null)
    setToolParams({})
    setToolResults({})
    setToolInferences({})
    setFinalVerdict(null)
    setFinalReport(null)
    setGlobalError(null)
    setReportLoading(false)
  }

  // ── Main pipeline ──────────────────────────────────────────────────────────
  const handleAnalyze = useCallback(async (text) => {
    resetAnalysis()
    setRawText(text)

    // ── Step 1: Extract JD ─────────────────────────────────────────────────
    setPipelineStep('extract')
    let extracted = {}
    try {
      const extractCfg = getLLMConfig('extractModel')
      const res = await llmExtract(settings.backendUrl, text, extractCfg)
      if (!res.ok) throw new Error(res.error || 'Extraction failed')
      extracted = res.data || {}
      setJobDict(extracted)
    } catch (err) {
      setGlobalError(`Extract failed: ${err.message}`)
      setPipelineStep(null)
      return
    }

    // ── Step 2: Deep Research ──────────────────────────────────────────────
    setPipelineStep('research')
    let researchResult = null
    try {
      const resCfg = getLLMConfig('deepResearchModel')
      researchResult = await llmDeepResearch(settings.backendUrl, extracted, text, resCfg)
      setDeepResearch(researchResult)
      // Merge overrides into jobDict for tool params
      if (researchResult?.data?.applied_overrides) {
        extracted = { ...extracted, ...researchResult.data.applied_overrides }
        setJobDict(extracted)
      }
    } catch (_) {
      // Deep research is optional — continue
    }

    // ── Build tool params from extracted data ──────────────────────────────
    const params = {}
    for (const [name, meta] of Object.entries(tools)) {
      params[name] = buildToolDefaults(meta.input_schema, extracted)
    }
    setToolParams(params)

    // ── Step 3: Run all tools ──────────────────────────────────────────────
    setPipelineStep('tools')
    const toolNames = Object.keys(tools).filter(n => !tools[n].is_stub)
    const total = toolNames.length
    let done = 0
    const results = {}

    setToolProgress({ done: 0, total, currentTool: toolNames[0] || '' })

    // Run tools concurrently in batches of 4
    const BATCH = 4
    for (let i = 0; i < toolNames.length; i += BATCH) {
      const batch = toolNames.slice(i, i + BATCH)
      await Promise.allSettled(
        batch.map(async (name) => {
          setToolProgress(p => ({ ...p, currentTool: name }))
          try {
            const res = await runTool(settings.backendUrl, name, params[name] || {})
            results[name] = res
          } catch (err) {
            results[name] = { ok: false, tool: name, error: err.message }
          }
          done++
          setToolResults({ ...results })
          setToolProgress(p => ({ ...p, done }))
        })
      )
    }

    // ── Step 4: LLM Inference per tool ─────────────────────────────────────
    setPipelineStep('inference')
    const inferences = {}
    const infCfg = getLLMConfig('toolInferenceModel')

    await Promise.allSettled(
      Object.entries(results).map(async ([name, result]) => {
        if (!result?.ok) return
        try {
          const inf = await llmToolInference(
            settings.backendUrl, name,
            tools[name]?.label || name,
            result?.result || result,
            extracted, infCfg
          )
          if (inf?.ok) inferences[name] = inf.bullets?.join(' | ') || inf.inference || ''
        } catch (_) {}
        setToolInferences({ ...inferences })
      })
    )

    // ── Step 5: Final report ───────────────────────────────────────────────
    setPipelineStep('report')
    setReportLoading(true)
    try {
      const sumCfg = getLLMConfig('finalSummaryModel')
      const sumRes = await llmFinalSummary(
        settings.backendUrl, extracted, results, inferences, sumCfg
      )
      setFinalVerdict(sumRes.verdict || 'SUSPICIOUS')
      setFinalReport(sumRes.report || '')
    } catch (err) {
      setGlobalError(`Final report failed: ${err.message}`)
    }
    setReportLoading(false)
    setPipelineStep('done')

  }, [settings, tools, getLLMConfig])

  // ── render ─────────────────────────────────────────────────────────────────
  return (
    <div className={styles.page}>
      <Header />

      <main className={`container ${styles.main}`}>

        {/* Hero */}
        <div className={styles.hero}>
          <h1 className={styles.heroTitle}>
            Detect Fake Job Postings with AI
          </h1>
          <p className={styles.heroSub}>
            Paste any job description — we'll run 13 investigative tools and a fine-tuned
            RoBERTa classifier to assess fraud risk.
          </p>
        </div>

        {/* Backend status */}
        {toolsError && (
          <div className={styles.errorBanner}>
            ⚠️ {toolsError}
            <button className="btn btn-ghost btn-sm" onClick={loadTools}>Retry</button>
          </div>
        )}

        {toolsLoading && (
          <div className={styles.loadingBar}>
            <div className="progress-bar progress-indeterminate"><div className="progress-fill" /></div>
            <span>Connecting to backend…</span>
          </div>
        )}

        {/* Input */}
        {!isAnalysing && pipelineStep !== 'done' && (
          <JDInput onAnalyze={handleAnalyze} isLoading={isAnalysing} />
        )}

        {/* Reset button once done */}
        {pipelineStep === 'done' && (
          <div className={styles.resetRow}>
            <button className="btn btn-secondary" onClick={resetAnalysis}>
              ← Analyse Another JD
            </button>
          </div>
        )}

        {/* Global error */}
        {globalError && (
          <div className={styles.errorBanner}>⚠️ {globalError}</div>
        )}

        {/* Pipeline progress */}
        {pipelineStep && pipelineStep !== 'done' && (
          <PipelineProgress
            currentStep={pipelineStep}
            toolProgress={toolProgress}
          />
        )}

        {/* Final Report (shown first for prominence) */}
        {(finalVerdict || reportLoading) && (
          <FinalReport
            verdict={finalVerdict}
            report={finalReport}
            isLoading={reportLoading}
          />
        )}

        {/* Extracted Info + Deep Research */}
        {jobDict && (
          <div className={styles.twoCol}>
            <ExtractedInfo jobDict={jobDict} deepResearch={deepResearch} />
          </div>
        )}

        {/* Tool Grid — shown as soon as toolParams are ready */}
        {Object.keys(toolParams).length > 0 && (
          <div className={styles.toolSection}>
            <div className={styles.sectionHeader}>
              <div className={styles.sectionTitle}>🛠️ Investigative Tools</div>
              <span className={styles.sectionHint}>
                Inputs are pre-filled from the JD. You can edit any field and run a tool individually.
              </span>
            </div>
            <ToolGrid
              tools={tools}
              toolParams={toolParams}
              jobDict={jobDict}
            />
          </div>
        )}

        {/* If no analysis yet — show tool registry preview */}
        {!pipelineStep && Object.keys(tools).length > 0 && (
          <div className={styles.toolPreview}>
            <div className={styles.sectionTitle}>Available Tools ({Object.keys(tools).length})</div>
            <div className={styles.previewGrid}>
              {Object.entries(tools).map(([name, meta]) => (
                <div key={name} className={styles.previewCard}>
                  <span className={styles.previewIcon}>{meta.icon}</span>
                  <div>
                    <div className={styles.previewLabel}>{meta.label}</div>
                    <div className={styles.previewCat}>{meta.category?.replace('_', ' ')}</div>
                  </div>
                  {meta.is_stub && <span className="badge badge-neutral" style={{fontSize:9}}>stub</span>}
                </div>
              ))}
            </div>
          </div>
        )}

      </main>

      <footer className={styles.footer}>
        <div className="container">
          FraudGuard · Built with RoBERTa + LangChain · Group 9 DS &amp; AI Lab
        </div>
      </footer>
    </div>
  )
}

// ── Root export with provider ────────────────────────────────────────────────
export default function App() {
  return (
    <SettingsProvider>
      <AppInner />
    </SettingsProvider>
  )
}
