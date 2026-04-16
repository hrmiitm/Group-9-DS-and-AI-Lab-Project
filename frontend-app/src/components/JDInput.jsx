/**
 * components/JDInput.jsx
 * Drag-and-drop + text area input for job descriptions.
 * Accepts txt/pdf/docx/md files and sends text to parent.
 */
import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import styles from './JDInput.module.css'

const SAMPLE_JD = `Job Title: Senior Data Entry Specialist (Work From Home)
Company: Global Solutions Inc.

We are urgently hiring Data Entry Specialists! 
Earn $500/day working from home — no experience needed!

Requirements:
- Basic computer skills
- WhatsApp only contact: +1-555-0198
- Apply immediately — only 10 seats left!
- Send registration fee of $50 to join

Contact: globalsolns@gmail.com
Website: http://globalsolutions-careers.net

Guaranteed income! Easy money from home. Work 2 hours daily.`

export default function JDInput({ onAnalyze, isLoading }) {
  const [tab, setTab]       = useState('text')   // 'text' | 'file'
  const [text, setText]     = useState('')
  const [fileName, setFileName] = useState(null)
  const [fileText, setFileText] = useState('')

  const onDrop = useCallback(acceptedFiles => {
    const file = acceptedFiles[0]
    if (!file) return
    setFileName(file.name)
    setTab('file')
    const reader = new FileReader()
    reader.onload = e => setFileText(e.target.result || '')
    reader.readAsText(file)
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/plain': ['.txt'],
      'text/markdown': ['.md'],
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
    },
    multiple: false,
  })

  const activeText = tab === 'file' ? fileText : text
  const canSubmit  = activeText.trim().length > 30

  const handleAnalyze = () => {
    if (canSubmit && !isLoading) onAnalyze(activeText.trim())
  }

  const handleSample = () => {
    setText(SAMPLE_JD)
    setTab('text')
  }

  return (
    <div className={styles.wrapper}>
      <div className={styles.tabBar}>
        <button
          className={`${styles.tab} ${tab === 'text' ? styles.tabActive : ''}`}
          onClick={() => setTab('text')}
          id="tab-text"
        >
          📝 Paste Text
        </button>
        <button
          className={`${styles.tab} ${tab === 'file' ? styles.tabActive : ''}`}
          onClick={() => setTab('file')}
          id="tab-file"
        >
          📁 Upload File
        </button>
      </div>

      {tab === 'text' ? (
        <div className={styles.textPanel}>
          <textarea
            id="jd-textarea"
            className={`textarea ${styles.jdTextarea}`}
            placeholder="Paste the full job description here…"
            value={text}
            onChange={e => setText(e.target.value)}
            rows={12}
          />
          <button className={`btn btn-ghost btn-sm ${styles.sampleBtn}`} onClick={handleSample}>
            Load sample fraudulent JD
          </button>
        </div>
      ) : (
        <div className={styles.filePanel}>
          <div
            {...getRootProps()}
            className={`${styles.dropzone} ${isDragActive ? styles.dropzoneActive : ''}`}
            id="file-dropzone"
          >
            <input {...getInputProps()} id="file-input" />
            {fileName ? (
              <div className={styles.fileInfo}>
                <span className={styles.fileIcon}>📄</span>
                <span className={styles.fileName}>{fileName}</span>
                <button
                  className="btn btn-ghost btn-sm"
                  onClick={e => { e.stopPropagation(); setFileName(null); setFileText('') }}
                >
                  Remove
                </button>
              </div>
            ) : (
              <div className={styles.dropPrompt}>
                <span className={styles.dropIcon}>{isDragActive ? '📂' : '📁'}</span>
                <p>{isDragActive ? 'Drop it here!' : 'Drag & drop a file, or click to browse'}</p>
                <span className={styles.dropHint}>Supports: .txt  .md  .pdf  .docx</span>
              </div>
            )}
          </div>
          {fileText && (
            <div className={styles.filePreview}>
              <label>Preview</label>
              <div className="code-block" style={{ maxHeight: 140 }}>{fileText.slice(0, 800)}{fileText.length > 800 ? '…' : ''}</div>
            </div>
          )}
        </div>
      )}

      <div className={styles.actions}>
        <span className={styles.charCount}>
          {activeText.trim().length} chars {activeText.trim().length < 30 && activeText.trim().length > 0 && '(too short)'}
        </span>
        <button
          id="analyze-btn"
          className="btn btn-primary btn-lg"
          onClick={handleAnalyze}
          disabled={!canSubmit || isLoading}
        >
          {isLoading ? (
            <><span className="spinner" /> Analysing…</>
          ) : (
            <>🔍 Analyse Job Posting</>
          )}
        </button>
      </div>
    </div>
  )
}
