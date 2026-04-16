/**
 * services/api.js
 * All fetch calls to the backend-api.
 * Every function accepts a `baseUrl` as first arg so it can be overridden.
 */

const DEFAULT_TIMEOUT = 60000 // 60s

function apiFetch(url, options = {}, timeout = DEFAULT_TIMEOUT) {
  const controller = new AbortController()
  const id = setTimeout(() => controller.abort(), timeout)
  return fetch(url, { ...options, signal: controller.signal })
    .then(async res => {
      clearTimeout(id)
      if (!res.ok) {
        const text = await res.text().catch(() => '')
        throw new Error(`HTTP ${res.status}: ${text.slice(0, 200)}`)
      }
      return res.json()
    })
    .catch(err => {
      clearTimeout(id)
      if (err.name === 'AbortError') throw new Error('Request timed out')
      throw err
    })
}

// ── Meta ──────────────────────────────────────────────────────
export async function fetchTools(baseUrl) {
  return apiFetch(`${baseUrl}/api/v1/tools`)
}

// ── Run a single tool ───────────────────────────────────────────
export async function runTool(baseUrl, toolName, params) {
  return apiFetch(`${baseUrl}/api/v1/run/${toolName}`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(params),
  }, 90000) // tools can be slow (WHOIS, etc.)
}

// ── LLM: Extract JD ────────────────────────────────────────────
export async function llmExtract(baseUrl, rawText, llmConfig = null) {
  return apiFetch(`${baseUrl}/api/v1/llm/extract`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ raw_text: rawText, llm_config: llmConfig }),
  }, 120000)
}

// ── LLM: Deep Research ─────────────────────────────────────────
export async function llmDeepResearch(baseUrl, jobDict, rawText, llmConfig = null) {
  return apiFetch(`${baseUrl}/api/v1/llm/deep-research`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ job_dict: jobDict, raw_text: rawText, llm_config: llmConfig }),
  }, 120000)
}

// ── LLM: Tool Inference ────────────────────────────────────────
export async function llmToolInference(baseUrl, toolName, toolLabel, toolResult, jobDict, llmConfig = null) {
  return apiFetch(`${baseUrl}/api/v1/llm/tool-inference`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({
      tool_name:   toolName,
      tool_label:  toolLabel,
      tool_result: toolResult,
      job_dict:    jobDict,
      llm_config:  llmConfig,
    }),
  }, 60000)
}

// ── LLM: Final Summary ─────────────────────────────────────────
export async function llmFinalSummary(baseUrl, jobDict, toolResults, toolInferences, llmConfig = null) {
  return apiFetch(`${baseUrl}/api/v1/llm/final-summary`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({
      job_dict:        jobDict,
      tool_results:    toolResults,
      tool_inferences: toolInferences,
      llm_config:      llmConfig,
    }),
  }, 120000)
}

// ── Helpers ────────────────────────────────────────────────────

/** Read a File object as text (for PDF, txt, etc. sent as multipart if needed) */
export async function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload  = e => resolve(e.target.result)
    reader.onerror = reject
    reader.readAsText(file) // works for txt, md; PDF will be raw bytes but server handles
  })
}

/** Convert tool registry input_schema into default param values for UI */
export function buildToolDefaults(inputSchema, jobDict) {
  const MAP = {
    job_text:        () => [jobDict.description, jobDict.title, jobDict.requirements].filter(Boolean).join('\n'),
    email:           () => jobDict.contact_email || '',
    domain_or_email: () => jobDict.contact_email || jobDict.company_website || '',
    company_name:    () => jobDict.company_name   || '',
    url:             () => jobDict.company_website || '',
    job_title:       () => jobDict.title           || '',
    location:        () => jobDict.location        || '',
    phone:           () => jobDict.contact_phone   || '',
  }

  const result = {}
  for (const [key, meta] of Object.entries(inputSchema || {})) {
    const fn = MAP[key]
    result[key] = fn ? fn() : (meta.default ?? '')
  }
  return result
}
