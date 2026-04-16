/**
 * components/PipelineProgress.jsx
 * Visual step-by-step progress tracker for the analysis pipeline.
 */
import styles from './PipelineProgress.module.css'

const STEPS = [
  { id: 'extract',   label: 'Extract JD',       icon: '📋' },
  { id: 'research',  label: 'Deep Research',     icon: '🔎' },
  { id: 'tools',     label: 'Run Tools',         icon: '🛠️' },
  { id: 'inference', label: 'LLM Inference',     icon: '🧠' },
  { id: 'report',    label: 'Final Report',      icon: '📊' },
]

export default function PipelineProgress({ currentStep, toolProgress }) {
  const currentIdx = STEPS.findIndex(s => s.id === currentStep)

  return (
    <div className={styles.wrapper}>
      <div className={styles.steps}>
        {STEPS.map((step, idx) => {
          const done    = idx < currentIdx
          const active  = idx === currentIdx
          const pending = idx > currentIdx

          return (
            <div key={step.id} className={styles.stepItem}>
              {idx > 0 && (
                <div className={`${styles.connector} ${done ? styles.connectorDone : ''}`} />
              )}
              <div className={`${styles.step} ${done ? styles.done : active ? styles.active : styles.pending}`}>
                <div className={styles.stepIcon}>
                  {done ? '✓' : active ? <span className="spinner" /> : step.icon}
                </div>
                <span className={styles.stepLabel}>{step.label}</span>
              </div>
            </div>
          )
        })}
      </div>

      {currentStep === 'tools' && toolProgress && (
        <div className={styles.toolProg}>
          <div className={styles.toolProgLabel}>
            <span>Running tools</span>
            <span className={styles.toolProgCount}>
              {toolProgress.done} / {toolProgress.total}
            </span>
          </div>
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{ width: `${toolProgress.total ? (toolProgress.done / toolProgress.total) * 100 : 0}%` }}
            />
          </div>
          {toolProgress.currentTool && (
            <span className={styles.currentTool}>Running: {toolProgress.currentTool}</span>
          )}
        </div>
      )}
    </div>
  )
}
