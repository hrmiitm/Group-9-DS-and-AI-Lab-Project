/**
 * components/Header.jsx
 * Top navigation bar with logo and settings button.
 */
import { useState } from 'react'
import SettingsModal from './SettingsModal'
import styles from './Header.module.css'

export default function Header() {
  const [showSettings, setShowSettings] = useState(false)

  return (
    <>
      <header className={styles.header}>
        <div className={`container ${styles.inner}`}>
          <div className={styles.logo}>
            <span className={styles.logoIcon}>🛡️</span>
            <div>
              <span className={styles.logoText}>FraudGuard</span>
              <span className={styles.logoSub}>Fake Job Detector</span>
            </div>
          </div>

          <nav className={styles.nav}>
            <button
              id="settings-btn"
              className="btn btn-ghost btn-sm"
              onClick={() => setShowSettings(true)}
              title="Configure LLM API settings"
            >
              ⚙️ Settings
            </button>
          </nav>
        </div>
      </header>

      {showSettings && <SettingsModal onClose={() => setShowSettings(false)} />}
    </>
  )
}
