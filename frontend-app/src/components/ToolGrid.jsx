/**
 * components/ToolGrid.jsx
 * Responsive grid of all ToolCards.
 * Filters by category.
 */
import { useState } from 'react'
import ToolCard from './ToolCard'
import styles from './ToolGrid.module.css'

const CATEGORIES = [
  { id: 'all',           label: 'All Tools' },
  { id: 'text_analysis', label: 'Text Analysis' },
  { id: 'contact',       label: 'Contact' },
  { id: 'website',       label: 'Website' },
  { id: 'company',       label: 'Company' },
  { id: 'ml_model',      label: 'ML Model' },
]

export default function ToolGrid({ tools, toolParams, jobDict }) {
  const [filter, setFilter] = useState('all')

  if (!tools || Object.keys(tools).length === 0) return null

  const filtered = Object.entries(tools).filter(([, meta]) =>
    filter === 'all' || meta.category === filter
  )

  return (
    <div className={styles.wrapper}>
      <div className={styles.filterBar}>
        {CATEGORIES.map(cat => (
          <button
            key={cat.id}
            className={`btn btn-ghost btn-sm ${filter === cat.id ? styles.filterActive : ''}`}
            onClick={() => setFilter(cat.id)}
          >
            {cat.label}
            {cat.id !== 'all' && (
              <span className={styles.count}>
                {Object.values(tools).filter(m => m.category === cat.id).length}
              </span>
            )}
          </button>
        ))}
      </div>

      <div className={styles.grid}>
        {filtered.map(([toolName, meta]) => (
          <ToolCard
            key={toolName}
            toolName={toolName}
            meta={meta}
            initialParams={toolParams[toolName] || {}}
            jobDict={jobDict}
          />
        ))}
      </div>
    </div>
  )
}
