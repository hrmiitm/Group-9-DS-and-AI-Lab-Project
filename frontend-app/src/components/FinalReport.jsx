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

const PLATFORM_ICONS = {
  linkedin:  '💼',
  twitter_x: '🐦',
  facebook:  '📘',
  instagram: '📷',
  github:    '🐙',
  youtube:   '▶️',
  glassdoor: '🪟',
}

export default function FinalReport({ verdict, report, isLoading, socialLinks = null, recentPosts = null }) {
  if (!verdict && !isLoading) return null

  const cfg = VERDICT_CONFIG[verdict] || VERDICT_CONFIG['SUSPICIOUS']

  // Defensively strip any leading VERDICT: line the backend might return
  // (backend now strips it, but this guards against cached/older responses)
  const strippedReport = (() => {
    if (!report) return ''
    const lines = report.split('\n')
    if (!lines[0]?.toUpperCase().startsWith('VERDICT:')) return report
    const rest = lines.slice(1)
    return (rest[0]?.trim() === '' ? rest.slice(1) : rest).join('\n')
  })()

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

          {strippedReport && (
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
                  a: ({href, children}) => (
                    <a href={href} target="_blank" rel="noopener noreferrer" className={styles.reportLink}>
                      {children}
                    </a>
                  ),
                }}
              >
                {strippedReport}
              </ReactMarkdown>
            </div>
          )}

          {/* Social media profiles discovered during deep research */}
          {socialLinks && Object.keys(socialLinks).length > 0 && (
            <div className={styles.socialSection}>
              <div className={styles.socialTitle}>🔗 Verified Social Profiles</div>
              <div className={styles.socialLinks}>
                {Object.entries(socialLinks).map(([platform, url]) =>
                  url ? (
                    <a
                      key={platform}
                      href={url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className={styles.socialChip}
                    >
                      {PLATFORM_ICONS[platform] || '🔗'} {platform.replace('_', '/')}
                    </a>
                  ) : null
                )}
              </div>
            </div>
          )}

          {/* Recent posts / mentions to review */}
          {recentPosts?.length > 0 && (
            <div className={styles.postsSection}>
              <div className={styles.socialTitle}>📰 Recent Posts &amp; Mentions to Review</div>
              <div className={styles.postsList}>
                {recentPosts.map((post, i) => (
                  <a
                    key={i}
                    href={post.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className={styles.postCard}
                  >
                    <span className={styles.postPlatform}>
                      {PLATFORM_ICONS[post.platform] || '🔗'} {post.platform}
                    </span>
                    <span className={styles.postTitle}>{post.title}</span>
                    {post.snippet && (
                      <span className={styles.postSnippet}>{post.snippet.slice(0, 150)}…</span>
                    )}
                  </a>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
