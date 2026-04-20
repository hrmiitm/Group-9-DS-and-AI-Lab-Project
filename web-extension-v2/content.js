// ============================================================
// FraudGuard v2 — Content Script (all-in-one)
// Scrapes LinkedIn DOM, calls backend API, renders results.
// All API calls happen here — no service worker needed.
// ============================================================

(function () {
    "use strict";
    if (window.__fraudguardV2) return;
    window.__fraudguardV2 = true;

    const DEFAULT_BACKEND = "https://hrmhrmhrm-company-backend-api.hf.space";

    // ── Config ────────────────────────────────────────────────
    function getConfig() {
        return new Promise((resolve) => {
            chrome.storage.local.get(
                ["backendUrl", "llmApiKey", "llmBaseUrl", "llmModel"],
                (result) => {
                    let url = result.backendUrl || DEFAULT_BACKEND;
                    // Auto-migrate stale localhost default
                    if (!url || url.includes("localhost") || url.includes("127.0.0.1")) {
                        url = DEFAULT_BACKEND;
                        chrome.storage.local.set({ backendUrl: DEFAULT_BACKEND });
                    }
                    resolve({
                        backendUrl:  url.replace(/\/$/, ""),
                        llmApiKey:   result.llmApiKey  || null,
                        llmBaseUrl:  result.llmBaseUrl || null,
                        llmModel:    result.llmModel   || null,
                    });
                }
            );
        });
    }

    function buildLlmConfig(config) {
        const cfg = {};
        if (config.llmApiKey)  cfg.api_key  = config.llmApiKey;
        if (config.llmBaseUrl) cfg.base_url = config.llmBaseUrl;
        if (config.llmModel)   cfg.model    = config.llmModel;
        return Object.keys(cfg).length > 0 ? cfg : undefined;
    }

    // ── Fetch with timeout ────────────────────────────────────
    async function apiPost(baseUrl, path, body, timeoutMs = 30000) {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), timeoutMs);
        try {
            const resp = await fetch(`${baseUrl}${path}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
                signal: controller.signal,
            });
            clearTimeout(timer);
            if (!resp.ok) {
                const text = await resp.text().catch(() => "");
                throw new Error(`HTTP ${resp.status}: ${text.substring(0, 150)}`);
            }
            return resp.json();
        } catch (err) {
            clearTimeout(timer);
            if (err.name === "AbortError") throw new Error(`Request timed out after ${timeoutMs / 1000}s`);
            throw err;
        }
    }

    async function apiGet(baseUrl, path, timeoutMs = 15000) {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), timeoutMs);
        try {
            const resp = await fetch(`${baseUrl}${path}`, { signal: controller.signal });
            clearTimeout(timer);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            return resp.json();
        } catch (err) {
            clearTimeout(timer);
            if (err.name === "AbortError") throw new Error(`Health check timed out after ${timeoutMs / 1000}s`);
            throw err;
        }
    }

    // ── DOM Scraping ──────────────────────────────────────────
    function getTextContent(selector) {
        try {
            const el = document.querySelector(selector);
            const text = el?.textContent?.trim();
            return text?.length > 0 ? text : null;
        } catch { return null; }
    }

    function getTextFromSelectors(selectors) {
        for (const sel of selectors) {
            const r = getTextContent(sel);
            if (r) return r;
        }
        return null;
    }

    function isVisible(el) {
        if (!el) return false;
        if (el.getAttribute("aria-hidden") === "true") return false;
        if (el.closest('[aria-hidden="true"]')) return false;
        const cls = (el.className || "") + " " + (el.closest("[class]")?.className || "");
        if (/visually.?hidden|sr-only|screen-reader/i.test(cls)) return false;
        if (el.offsetParent === null && getComputedStyle(el).position !== "fixed") return false;
        return true;
    }

    function isValidJobTitle(t) {
        if (!t || t.length < 2 || t.length > 200) return false;
        return !/^\d+\s+notification|^(home|menu|nav|search|messaging|connections|jobs)$/i.test(t.trim());
    }

    function parseDocumentTitle() {
        const raw = document.title || "";
        const m = raw.match(/^(.+?)\s+at\s+(.+?)\s*[\|·]/);
        if (m) return { title: m[1].trim(), company: m[2].trim() };
        const cleaned = raw.replace(/\s*[\|·].*$/, "").trim();
        return cleaned.length > 2 ? { title: cleaned, company: "" } : null;
    }

    function getTextByAttrContains(attrValue, tag = "*") {
        try {
            const el = document.querySelector(`${tag}[class*="${attrValue}"]`);
            const text = el?.textContent?.trim();
            return text?.length > 0 ? text : null;
        } catch { return null; }
    }

    function getDescriptionFallback() {
        const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
            acceptNode: (n) => {
                const t = n.textContent.trim().toLowerCase();
                return (t === "about the job" || t === "about this role" || t === "about the role")
                    ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_SKIP;
            },
        });
        let node;
        while ((node = walker.nextNode())) {
            let sib = node.parentElement?.nextElementSibling;
            while (sib) {
                if (sib.textContent.trim().length > 150) return sib.textContent.trim();
                sib = sib.nextElementSibling;
            }
        }
        return null;
    }

    function scrapeJobData() {
        const docParsed = parseDocumentTitle();
        const data = {};

        const domTitle = getTextFromSelectors([
            ".job-details-jobs-unified-top-card__job-title h1",
            ".job-details-jobs-unified-top-card__job-title",
            ".jobs-unified-top-card__job-title",
            ".t-24.t-bold.inline",
            ".top-card-layout__title",
        ]) || getTextByAttrContains("job-title", "h1") || getTextByAttrContains("job-title")
          || (() => {
              for (const h1 of document.querySelectorAll("h1")) {
                  const t = h1.textContent.trim();
                  if (isVisible(h1) && isValidJobTitle(t)) return t;
              }
              return null;
          })();

        data.title   = (isValidJobTitle(domTitle) ? domTitle : null) || docParsed?.title || "";
        data.company = getTextFromSelectors([
            ".job-details-jobs-unified-top-card__company-name a",
            ".job-details-jobs-unified-top-card__company-name",
            ".jobs-unified-top-card__company-name a",
            ".jobs-unified-top-card__company-name",
            ".topcard__org-name-link",
        ]) || getTextByAttrContains("company-name", "a") || getTextByAttrContains("company-name")
          || (() => {
              for (const a of document.querySelectorAll("a[href*='/company/']")) {
                  const t = a.textContent.trim();
                  if (t.length > 1 && t.length < 100 && isVisible(a)) return t;
              }
              return null;
          })() || docParsed?.company || "";

        data.location     = getTextFromSelectors([".job-details-jobs-unified-top-card__bullet", ".jobs-unified-top-card__bullet", ".topcard__flavor--bullet"]) || getTextByAttrContains("bullet") || "";
        data.workplaceType = getTextFromSelectors([".job-details-jobs-unified-top-card__workplace-type", ".jobs-unified-top-card__workplace-type"]) || getTextByAttrContains("workplace-type") || "";
        data.salary        = getTextFromSelectors([".job-details-jobs-unified-top-card__job-insight--highlight", ".salary-main-rail__data-body", ".compensation__salary"]) || getTextByAttrContains("salary") || "";

        document.querySelectorAll('[class*="job-insight"], [class*="job-criteria"], .description__job-criteria-item').forEach((item) => {
            const label = (item.querySelector('[class*="subtitle"], h3, .t-black--light')?.textContent || "").trim().toLowerCase();
            const value = (item.querySelector('span:last-child, [class*="criteria-text"]')?.textContent || "").trim();
            const text  = item.textContent.trim();
            if (label.includes("seniority"))   data.seniorityLevel  = value || text;
            if (label.includes("employment") || /full-time|part-time|internship|contract/i.test(text)) data.employmentType = value || text;
            if (label.includes("function"))     data.jobFunction     = value || text;
            if (label.includes("industr"))      data.industries      = value || text;
        });

        data.description = getTextFromSelectors([
            "#job-details",
            ".jobs-description-content__text",
            ".jobs-description__content",
            ".jobs-box__html-content",
            ".show-more-less-html__markup",
            "article.jobs-description",
            "[class*='jobs-description']",
        ]) || getTextByAttrContains("description-content") || getDescriptionFallback()
          || (() => {
              const JOB_KW = /responsibilities|requirements|qualifications|experience|skills|benefits/i;
              let best = null, bestLen = 0;
              for (const el of document.querySelectorAll("div, section, article")) {
                  const t = el.textContent.trim();
                  if (t.length > bestLen && t.length < 15000 && JOB_KW.test(t)) { best = t; bestLen = t.length; }
              }
              return best;
          })() || "";

        data.companyDescription = getTextFromSelectors([".jobs-company__company-description"]) || getTextByAttrContains("company-description") || "";

        Object.keys(data).forEach(k => { if (typeof data[k] === "string") data[k] = data[k].replace(/\s+/g, " ").trim(); });
        return data;
    }

    async function waitForJobContent() {
        for (let i = 0; i < 6; i++) {
            const d = scrapeJobData();
            if (d.title && d.description?.length > 80) return d;
            if (i < 5) await sleep(900);
        }
        return scrapeJobData();
    }

    function buildJobText(d) {
        const parts = [];
        if (d.title)              parts.push(`Job Title: ${d.title}`);
        if (d.company)            parts.push(`Company: ${d.company}`);
        if (d.location)           parts.push(`Location: ${d.location}`);
        if (d.workplaceType)      parts.push(`Workplace Type: ${d.workplaceType}`);
        if (d.employmentType)     parts.push(`Employment Type: ${d.employmentType}`);
        if (d.seniorityLevel)     parts.push(`Seniority Level: ${d.seniorityLevel}`);
        if (d.salary)             parts.push(`Salary: ${d.salary}`);
        if (d.industries)         parts.push(`Industry: ${d.industries}`);
        if (d.description)        parts.push(`\nJob Description:\n${d.description}`);
        if (d.companyDescription) parts.push(`\nCompany Profile:\n${d.companyDescription}`);
        return parts.join("\n");
    }

    function buildBatch(jobText, jobData) {
        const batch = [];
        const company = jobData.company || "";
        const title   = jobData.title   || "";

        batch.push({ tool_name: "scam_signals",       job_text: jobText });
        batch.push({ tool_name: "roberta_classifier",  job_text: jobText });

        if (company) {
            batch.push({ tool_name: "company_wikipedia",  company_name: company });
            batch.push({ tool_name: "company_web_search", company_name: company });
            batch.push({ tool_name: "company_news",       company_name: company });
            batch.push({ tool_name: "social_profiles",    company_name: company });
        }
        if (title && company) {
            batch.push({ tool_name: "job_boards", job_title: title, company_name: company });
        }
        return batch;
    }

    function batchToDict(results) {
        const d = {};
        for (const r of (results || [])) {
            if (r.ok && r.result?.ok) d[r.tool] = r.result.data;
            else d[r.tool] = { error: true };
        }
        return d;
    }

    function heuristicVerdict(toolResults) {
        const rob = toolResults.roberta_classifier;
        if (rob?.is_fraud || rob?.fraud_probability >= 0.87) return "LIKELY_FAKE";
        if (rob?.fraud_probability >= 0.5) return "SUSPICIOUS";
        const scam = toolResults.scam_signals;
        if (scam?.risk_level === "HIGH")   return "LIKELY_FAKE";
        if (scam?.risk_level === "MEDIUM") return "SUSPICIOUS";
        return "SAFE";
    }

    function heuristicReport(jobData, toolResults, verdict) {
        const lines = [`## Fraud Risk Assessment: ${verdict}`, ""];
        const rob = toolResults.roberta_classifier;
        if (rob) {
            lines.push("### ML Model (RoBERTa)");
            lines.push(`- Fraud probability: **${Math.round((rob.fraud_probability || 0) * 100)}%** (threshold: 87%)`);
            lines.push(`- Verdict: **${rob.label || "N/A"}** — ${rob.confidence || "N/A"} confidence`);
            lines.push("");
        }
        const scam = toolResults.scam_signals;
        if (scam) {
            lines.push("### Scam Signal Scanner");
            lines.push(`- Risk level: **${scam.risk_level || "N/A"}** (score: ${scam.scam_score || 0}/100)`);
            if (scam.signals_found?.length) lines.push(`- Signals: ${scam.signals_found.slice(0, 5).join(", ")}`);
            lines.push("");
        }
        const wiki = toolResults.company_wikipedia;
        if (wiki && !wiki.error && wiki.extract) {
            lines.push("### Company Wikipedia");
            lines.push(`- ${wiki.extract.substring(0, 200)}...`);
            lines.push("");
        }
        const news = toolResults.company_news;
        if (news && !news.error && news.total_articles > 0) {
            lines.push(`### Company News — ${news.total_articles} articles found`);
            lines.push("");
        }
        lines.push("---");
        lines.push("*Note: LLM report skipped — showing raw tool analysis.*");
        return lines.join("\n");
    }

    // ── Main analysis pipeline (runs in content script) ───────
    async function runAnalysis(jobData) {
        try {
            const config    = await getConfig();
            const { backendUrl } = config;
            const llmConfig = buildLlmConfig(config);
            const jobText   = buildJobText(jobData);

            if (jobText.length < 30) {
                showError("Not enough job content found. Scroll to load the full job description and try again.");
                return;
            }

            // Step 1: Health check (wake up HF Space if needed)
            updateProgress(1, 4, "Connecting to FraudGuard backend...");
            try {
                await apiGet(backendUrl, "/health", 12000);
            } catch (_) {
                updateProgress(1, 4, "Backend waking up (HF Space cold start ~30s)...");
                try {
                    await apiGet(backendUrl, "/health", 50000);
                } catch (e) {
                    throw new Error(`Cannot reach backend at ${backendUrl}.\n\n${e.message}\n\nPlease click the 🛡️ icon and verify the backend URL.`);
                }
            }

            // Step 2: Run batch tools
            updateProgress(2, 4, "Running 13 investigation tools...");
            const batch = buildBatch(jobText, jobData);
            let toolResults = {};
            try {
                const batchResp = await apiPost(backendUrl, "/api/v1/run-batch", batch, 90000);
                if (batchResp.ok && Array.isArray(batchResp.results)) {
                    toolResults = batchToDict(batchResp.results);
                }
            } catch (err) {
                console.warn("[FG v2] Batch failed, trying core tools:", err.message);
                updateProgress(2, 4, "Running core tools individually...");
                const [s, r] = await Promise.allSettled([
                    apiPost(backendUrl, "/api/v1/run/scam_signals",      { job_text: jobText }, 30000),
                    apiPost(backendUrl, "/api/v1/run/roberta_classifier", { job_text: jobText }, 60000),
                ]);
                if (s.status === "fulfilled" && s.value?.ok) toolResults.scam_signals      = s.value.result?.data;
                if (r.status === "fulfilled" && r.value?.ok) toolResults.roberta_classifier = r.value.result?.data;
            }

            // Step 3: LLM final summary
            updateProgress(3, 4, "Generating AI fraud report...");
            let verdict = heuristicVerdict(toolResults);
            let report  = "";
            try {
                const summaryBody = {
                    job_dict: {
                        title:           jobData.title    || "",
                        company_name:    jobData.company  || "",
                        location:        jobData.location || "",
                        salary_range:    jobData.salary   || "",
                        employment_type: jobData.employmentType || "",
                        description:     jobData.description || "",
                    },
                    tool_results: toolResults,
                };
                if (llmConfig) summaryBody.llm_config = llmConfig;
                const summaryResp = await apiPost(backendUrl, "/api/v1/llm/final-summary", summaryBody, 50000);
                if (summaryResp.ok) {
                    verdict = summaryResp.verdict || verdict;
                    report  = summaryResp.report  || "";
                }
            } catch (err) {
                console.warn("[FG v2] LLM summary skipped:", err.message);
                report = heuristicReport(jobData, toolResults, verdict);
            }

            updateProgress(4, 4, "Complete!");
            renderResults(verdict, report, toolResults);

        } catch (err) {
            console.error("[FG v2]", err);
            showError(err.message || "Unknown error.");
        }
    }

    // ── UI: Button ────────────────────────────────────────────
    function injectButton() {
        if (document.getElementById("fg-analyze-btn")) return;
        const btn = document.createElement("button");
        btn.id = "fg-analyze-btn";
        btn.innerHTML = `<span class="fg-btn-icon">🛡️</span><span class="fg-btn-text">FraudGuard</span>`;
        btn.addEventListener("click", handleClick);
        document.body.appendChild(btn);
    }

    async function handleClick() {
        const btn = document.getElementById("fg-analyze-btn");
        if (btn) { btn.disabled = true; btn.innerHTML = `<span class="fg-btn-icon">⏳</span><span class="fg-btn-text">Analyzing...</span>`; }
        removeOverlay();
        showProgressModal("Scraping job details...", 0, 4);

        const jobData = await waitForJobContent();
        if (!jobData.title && !jobData.description) {
            showError("No job listing detected. Open a LinkedIn job detail page first.");
            resetButton();
            return;
        }
        await runAnalysis(jobData);
        resetButton();
    }

    function resetButton() {
        const btn = document.getElementById("fg-analyze-btn");
        if (btn) { btn.disabled = false; btn.innerHTML = `<span class="fg-btn-icon">🛡️</span><span class="fg-btn-text">FraudGuard</span>`; }
    }

    // ── UI: Progress modal ────────────────────────────────────
    function showProgressModal(label, step, total) {
        let modal = document.getElementById("fg-progress-modal");
        if (!modal) {
            modal = document.createElement("div");
            modal.id = "fg-progress-modal";
            modal.innerHTML = `
        <div class="fg-progress-inner">
          <div class="fg-progress-logo">🛡️ FraudGuard</div>
          <div class="fg-progress-label" id="fg-progress-label"></div>
          <div class="fg-progress-bar-track"><div class="fg-progress-bar-fill" id="fg-progress-bar-fill"></div></div>
          <div class="fg-progress-steps" id="fg-progress-steps"></div>
        </div>`;
            document.body.appendChild(modal);
        }
        document.getElementById("fg-progress-label").textContent = label;
        const pct = total > 0 ? Math.round((step / total) * 100) : 5;
        document.getElementById("fg-progress-bar-fill").style.width = Math.max(5, pct) + "%";
        document.getElementById("fg-progress-steps").textContent = step > 0 ? `Step ${step} of ${total}` : "";
    }

    function updateProgress(step, total, label) {
        showProgressModal(label, step, total);
    }

    function hideProgressModal() {
        document.getElementById("fg-progress-modal")?.remove();
    }

    function removeOverlay() {
        document.getElementById("fg-overlay")?.remove();
        document.getElementById("fg-progress-modal")?.remove();
    }

    // ── UI: Error display ─────────────────────────────────────
    function showError(msg) {
        hideProgressModal();
        removeOverlay();
        const overlay = document.createElement("div");
        overlay.id = "fg-overlay";
        overlay.innerHTML = `
      <div class="fg-overlay-header fg-error">
        <div class="fg-verdict-left">
          <span class="fg-verdict-icon">⚠️</span>
          <div>
            <div class="fg-verdict-text" style="color:#fcd34d">Analysis Failed</div>
            <div class="fg-verdict-sub">FraudGuard v2</div>
          </div>
        </div>
        <button class="fg-close-btn" id="fg-close-btn">✕</button>
      </div>
      <div class="fg-error-msg">${escapeHtml(msg)}</div>
      <div class="fg-config-hint">💡 Click the 🛡️ icon in your browser toolbar to verify the Backend URL is set to:<br><strong>https://hrmhrmhrm-company-backend-api.hf.space</strong></div>`;
        document.body.appendChild(overlay);
        document.getElementById("fg-close-btn").addEventListener("click", () => { removeOverlay(); });
    }

    // ── UI: Results overlay ───────────────────────────────────
    function renderResults(verdict, report, toolResults) {
        hideProgressModal();
        removeOverlay();

        const vc = {
            SAFE:        { icon: "✅", label: "SAFE",        cls: "fg-safe",  bg: "fg-safe-bg" },
            SUSPICIOUS:  { icon: "⚠️",  label: "SUSPICIOUS",  cls: "fg-suspicious", bg: "fg-suspicious-bg" },
            LIKELY_FAKE: { icon: "❌", label: "LIKELY FAKE", cls: "fg-fake",  bg: "fg-fake-bg" },
        }[verdict] || { icon: "⚠️", label: verdict, cls: "fg-suspicious", bg: "fg-suspicious-bg" };

        const toolMeta = {
            scam_signals:       { icon: "🚨", label: "Scam Signals" },
            roberta_classifier: { icon: "🤖", label: "ML Model" },
            email_verify:       { icon: "📧", label: "Email" },
            domain_reputation:  { icon: "🌐", label: "Domain" },
            website_verify:     { icon: "🔗", label: "Website" },
            website_content:    { icon: "📄", label: "Site Content" },
            company_wikipedia:  { icon: "📖", label: "Wikipedia" },
            company_web_search: { icon: "🔍", label: "Web Search" },
            company_news:       { icon: "📰", label: "News" },
            social_profiles:    { icon: "👥", label: "Social" },
            job_boards:         { icon: "📋", label: "Job Boards" },
            phone_check:        { icon: "📞", label: "Phone" },
        };

        const toolCardsHtml = Object.entries(toolResults)
            .filter(([n]) => toolMeta[n])
            .map(([n, r]) => {
                const m = toolMeta[n];
                const isErr   = r?.error;
                const isBad   = r?.is_fraud || r?.risk_level === "HIGH" || r?.verdict === "FRAUDULENT";
                const cls     = isErr ? "fg-tool-skip" : isBad ? "fg-tool-fail" : "fg-tool-pass";
                const icon    = isErr ? "—" : isBad ? "✕" : "✓";
                return `<div class="fg-tool-card ${cls}" title="${escapeHtml(n)}"><span class="fg-tool-icon">${m.icon}</span><span class="fg-tool-name">${m.label}</span><span class="fg-tool-status">${icon}</span></div>`;
            }).join("");

        const rob  = toolResults.roberta_classifier;
        const scam = toolResults.scam_signals;
        const robHtml  = rob  ? `<div class="fg-stat"><span class="fg-stat-label">ML Fraud Score</span><span class="fg-stat-value ${rob.is_fraud ? "fg-stat-danger" : "fg-stat-ok"}">${Math.round((rob.fraud_probability || 0) * 100)}%</span></div>` : "";
        const scamHtml = scam ? `<div class="fg-stat"><span class="fg-stat-label">Scam Signals</span><span class="fg-stat-value fg-stat-${scam.risk_level === "HIGH" ? "danger" : scam.risk_level === "MEDIUM" ? "warn" : "ok"}">${scam.risk_level || "N/A"} (${scam.scam_score || 0})</span></div>` : "";

        const summary = (report || "").split("\n").filter(l => l.trim() && !l.startsWith("#")).slice(0, 3).join(" ").substring(0, 400);

        const overlay = document.createElement("div");
        overlay.id = "fg-overlay";
        overlay.innerHTML = `
      <div class="fg-overlay-header ${vc.bg}">
        <div class="fg-verdict-left">
          <span class="fg-verdict-icon">${vc.icon}</span>
          <div>
            <div class="fg-verdict-text ${vc.cls}">${vc.label}</div>
            <div class="fg-verdict-sub">FraudGuard v2 Analysis</div>
          </div>
        </div>
        <button class="fg-close-btn" id="fg-close-btn">✕</button>
      </div>

      ${(robHtml || scamHtml) ? `<div class="fg-stats-row">${robHtml}${scamHtml}</div>` : ""}

      ${toolCardsHtml ? `<div class="fg-section"><div class="fg-section-title">Investigation Tools (${Object.keys(toolResults).length} ran)</div><div class="fg-tool-grid">${toolCardsHtml}</div></div>` : ""}

      ${summary ? `<div class="fg-section"><div class="fg-section-title">Summary</div><div class="fg-summary-text">${escapeHtml(summary)}</div></div>` : ""}

      ${report ? `<div class="fg-section">
        <button class="fg-toggle-btn" id="fg-report-toggle">
          <span id="fg-report-label">📋 Show Full Report</span>
          <span id="fg-toggle-arrow">▼</span>
        </button>
        <div id="fg-report-content" style="display:none">
          <div class="fg-report-text" id="fg-report-text"></div>
        </div>
      </div>` : ""}`;

        document.body.appendChild(overlay);

        document.getElementById("fg-close-btn").addEventListener("click", () => removeOverlay());

        if (report) {
            document.getElementById("fg-report-toggle").addEventListener("click", () => {
                const content = document.getElementById("fg-report-content");
                const arrow   = document.getElementById("fg-toggle-arrow");
                const lbl     = document.getElementById("fg-report-label");
                const open = content.style.display === "none";
                content.style.display = open ? "block" : "none";
                if (open) document.getElementById("fg-report-text").innerHTML = mdToHtml(report);
                arrow.textContent = open ? "▲" : "▼";
                lbl.textContent   = open ? "📋 Hide Full Report" : "📋 Show Full Report";
            });
        }
    }

    // ── Helpers ───────────────────────────────────────────────
    function mdToHtml(md) {
        return md.split("\n").map(line => {
            if (line.startsWith("#### ")) return `<h4>${escapeHtml(line.slice(5))}</h4>`;
            if (line.startsWith("### "))  return `<h3>${escapeHtml(line.slice(4))}</h3>`;
            if (line.startsWith("## "))   return `<h2>${escapeHtml(line.slice(3))}</h2>`;
            if (line.startsWith("# "))    return `<h1>${escapeHtml(line.slice(2))}</h1>`;
            if (line.startsWith("- ") || line.startsWith("* ")) return `<li>${inlineMd(line.slice(2))}</li>`;
            if (/^\d+\.\s/.test(line)) return `<li>${inlineMd(line.replace(/^\d+\.\s/, ""))}</li>`;
            if (!line.trim()) return "<br>";
            return `<p>${inlineMd(line)}</p>`;
        }).join("\n");
    }
    function inlineMd(t) {
        return escapeHtml(t)
            .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
            .replace(/\*(.+?)\*/g, "<em>$1</em>")
            .replace(/`(.+?)`/g, "<code>$1</code>");
    }
    function escapeHtml(s) {
        return String(s || "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
    }
    function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

    // ── SPA navigation handler ────────────────────────────────
    let lastUrl = location.href;
    new MutationObserver(() => {
        if (location.href !== lastUrl) {
            lastUrl = location.href;
            removeOverlay();
            setTimeout(injectButton, 1500);
        }
    }).observe(document.body, { subtree: true, childList: true });

    // ── Init ──────────────────────────────────────────────────
    injectButton();
})();
