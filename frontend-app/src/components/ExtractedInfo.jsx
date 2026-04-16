/**
 * components/ExtractedInfo.jsx
 * Shows the structured JD fields extracted by LLM.
 * Also shows deep research applied overrides.
 */
import styles from './ExtractedInfo.module.css'

const FIELD_LABELS = {
  title:              'Job Title',
  company_name:       'Company',
  location:           'Location',
  employment_type:    'Employment Type',
  required_experience:'Experience',
  required_education: 'Education',
  salary_range:       'Salary',
  industry:           'Industry',
  department:         'Department',
  contact_email:      'Contact Email',
  contact_phone:      'Contact Phone',
  company_website:    'Website',
  telecommuting:      'Remote',
  has_company_logo:   'Has Logo',
}

function FieldRow({ label, value, source }) {
  if (value === null || value === undefined || value === '') return null
  const display = typeof value === 'number' ? (value === 1 ? 'Yes' : 'No') : String(value)

  return (
    <div className={styles.row}>
      <span className={styles.fieldLabel}>{label}</span>
      <div className={styles.fieldValue}>
        <span className={styles.fieldValueText}>{display}</span>
        {source && (
          <span className={`badge badge-info ${styles.sourceBadge}`}>{source}</span>
        )}
      </div>
    </div>
  )
}

export default function ExtractedInfo({ jobDict, deepResearch }) {
  if (!jobDict) return null

  const overrides = deepResearch?.data?.applied_overrides || {}

  return (
    <div className={`card ${styles.wrapper}`}>
      <div className={styles.header}>
        <span className={styles.title}>📋 Extracted Job Info</span>
        {jobDict.title && (
          <span className={styles.jobTitle}>{jobDict.title}</span>
        )}
      </div>

      <div className={styles.grid}>
        {Object.entries(FIELD_LABELS).map(([key, label]) => {
          const val    = overrides[key] ?? jobDict[key]
          const source = overrides[key] ? 'Deep Research' : null
          return <FieldRow key={key} label={label} value={val} source={source} />
        })}
      </div>

      {deepResearch?.data && (
        <div className={styles.research}>
          <div className={styles.researchHeader}>
            <span>🔎 Deep Research</span>
            {deepResearch.data.missing_from_jd?.length > 0 && (
              <span className={styles.missing}>
                Missing from JD: {deepResearch.data.missing_from_jd.join(', ')}
              </span>
            )}
          </div>
          {deepResearch.data.summary && (
            <p className={styles.researchSummary}>{deepResearch.data.summary}</p>
          )}
          {deepResearch.data.sources?.length > 0 && (
            <div className={styles.sources}>
              {deepResearch.data.sources.slice(0, 3).map((src, i) => (
                <a key={i} href={src} target="_blank" rel="noopener noreferrer" className={styles.source}>
                  🔗 {src.slice(0, 50)}…
                </a>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
