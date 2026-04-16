/**
 * components/FinalReport.jsx
 * Displays the final fraud verdict and markdown report.
 */
import ReactMarkdown from 'react-markdown'
import styles from './FinalReport.module.css'

const VERDICT_CONFIG = {
  SAFE:        { icon: '✅', label: 'Safe',        cls: styles.safe,        badge: 'badge-safe' },
  SUSPICIOUS:  { icon: '⚠️', label: 'Suspicious',  cls: styles.suspicious,  badge: 'badge-suspicious' },
  LIKELY_FAKE: { icon: '❌', label: 'Likely Fake', cls: styles.fake,        badge: 'badge-fake' },
}

export default function FinalReport({ verdict, report, isLoading }) {
  if (!verdict && !isLoading) return null

  const cfg = VERDICT_CONFIG[verdict] || VERDICT_CONFIG['SUSPICIOUS']

  return (
    <div className={`${styles.wrapper} card fade-in`}>
      {isLoading ? (
        <div className={styles.loading}>
          <span className="spinner spinner-lg" />
          <span>Generating final report…</span>
        </div>
      ) : (
        <>
          <div className={`${styles.verdictBanner} ${cfg.cls}`}>
            <div className={styles.verdictLeft}>
              <span className={styles.verdictIcon}>{cfg.icon}</span>
              <div>
                <div className={styles.verdictTitle}>Fraud Risk Assessment</div>
                <div className={styles.verdictLabel}>{cfg.label}</div>
              </div>
            </div>
            <span className={`badge ${cfg.badge} ${styles.verdictBadge}`}>{verdict}</span>
          </div>

          {report && (
            <div className={styles.reportBody}>
              <ReactMarkdown
                components={{
                  h1: ({children}) => <h1 className={styles.h1}>{children}</h1>,
                  h2: ({children}) => <h2 className={styles.h2}>{children}</h2>,
                  h3: ({children}) => <h3 className={styles.h3}>{children}</h3>,
                  p:  ({children}) => <p  className={styles.p}>{children}</p>,
                  ul: ({children}) => <ul className={styles.ul}>{children}</ul>,
                  li: ({children}) => <li className={styles.li}>{children}</li>,
                  strong: ({children}) => <strong className={styles.strong}>{children}</strong>,
                  hr: () => <hr />,
                }}
              >
                {report}
              </ReactMarkdown>
            </div>
          )}
        </>
      )}
    </div>
  )
}
