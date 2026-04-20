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
            renderResults(verdict, report, toolResults, jobData.company || "");

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

    // ── Sassy insight line generator ─────────────────────────
    function buildInsight(verdict, company, rob, scam) {
        const co   = company ? `<strong class="fg-insight-company">${escapeHtml(company)}</strong>` : "This job posting";
        const pct  = rob  ? Math.round((rob.fraud_probability || 0) * 100) : null;
        const risk = scam ? (scam.risk_level || "").toLowerCase() : null;

        if (verdict === "SAFE") {
            const lines = [
                `${co} looks legitimate. Our AI ran ${pct !== null ? `a <strong>${pct}% fraud probability</strong> through RoBERTa` : "13 deep checks"} and found no significant red flags — you're good to apply.`,
                `${co} passes our 13-tool verification. ML fraud score is a clean <strong>${pct ?? "low"}%</strong> with ${risk || "low"} scam signals. Looks like a real opportunity.`,
                `All clear on ${co}. Scam signals are <strong>${risk || "low"}</strong> and the ML model gave it a fraud score of just <strong>${pct ?? "—"}%</strong>. Go ahead — this one checks out.`,
            ];
            return lines[Math.floor(Math.random() * lines.length)];
        }
        if (verdict === "SUSPICIOUS") {
            const lines = [
                `${co} raised a few eyebrows. Fraud probability sits at <strong>${pct ?? "moderate"}%</strong> with <strong>${risk || "medium"}</strong> scam signals — not a definitive red flag, but dig deeper before applying.`,
                `Something feels off about ${co}. Our ML model scored it <strong>${pct ?? "moderate"}%</strong> fraud likelihood. We'd recommend verifying the company before sharing any personal info.`,
                `${co} is a maybe. Scam signals are <strong>${risk || "medium"}</strong> and fraud probability is <strong>${pct ?? "moderate"}%</strong>. Research the company independently before proceeding.`,
            ];
            return lines[Math.floor(Math.random() * lines.length)];
        }
        // LIKELY_FAKE
        const lines = [
            `🚨 Stay away from ${co}. Fraud probability is <strong>${pct ?? "high"}%</strong> with <strong>${risk || "high"}</strong> scam signals. Multiple red flags detected — do not share personal information.`,
            `${co} is almost certainly a scam. Our RoBERTa model flagged it at <strong>${pct ?? "high"}%</strong> fraud probability. The scam pattern score is <strong>${risk || "high"}</strong>. Proceed with extreme caution.`,
            `This is a red alert for ${co}. <strong>${pct ?? "high"}% fraud probability</strong> + <strong>${risk || "high"}</strong> scam signals = classic fraudulent job posting. Don't apply.`,
        ];
        return lines[Math.floor(Math.random() * lines.length)];
    }

    // ── UI: Error display ─────────────────────────────────────
    function showError(msg) {
        hideProgressModal();
        removeOverlay();
        const overlay = document.createElement("div");
        overlay.id = "fg-overlay";
        overlay.innerHTML = `
      <div class="fg-hero fg-hero-error">
        <div class="fg-hero-top">
          <span class="fg-badge" style="background:rgba(239,68,68,0.15);border-color:rgba(248,113,113,0.4);color:#f87171">
            <span class="fg-badge-icon">⚠️</span> Analysis Failed
          </span>
          <button class="fg-close-btn" id="fg-close-btn">✕</button>
        </div>
        <div class="fg-insight" style="font-size:13px;color:#94a3b8">
          Something went wrong while analyzing this job posting.
        </div>
      </div>
      <div class="fg-error-body">
        <div class="fg-error-msg">${escapeHtml(msg)}</div>
        <div class="fg-config-hint">💡 Click the <strong>🛡️</strong> icon in the browser toolbar and verify the Backend URL is:<br><strong>https://hrmhrmhrm-company-backend-api.hf.space</strong></div>
      </div>
      <div class="fg-footer"><span class="fg-footer-text">FraudGuard v2</span><span class="fg-footer-dot"></span><span class="fg-footer-text">Powered by RoBERTa + GPT</span></div>`;
        document.body.appendChild(overlay);
        document.getElementById("fg-close-btn").addEventListener("click", () => removeOverlay());
    }

    // ── UI: Results overlay ───────────────────────────────────
    function renderResults(verdict, report, toolResults, company) {
        hideProgressModal();
        removeOverlay();

        const rob  = toolResults.roberta_classifier;
        const scam = toolResults.scam_signals;

        // Verdict config
        const vMap = {
            SAFE:        { badge: "fg-badge-safe",  hero: "fg-hero-safe",  icon: "✅", label: "SAFE"        },
            SUSPICIOUS:  { badge: "fg-badge-warn",  hero: "fg-hero-warn",  icon: "⚠️", label: "SUSPICIOUS"  },
            LIKELY_FAKE: { badge: "fg-badge-fake",  hero: "fg-hero-fake",  icon: "🚨", label: "LIKELY FAKE" },
        };
        const vc = vMap[verdict] || vMap["SUSPICIOUS"];

        // Insight class for company colour
        const insightCls = verdict === "SAFE" ? "fg-insight-safe" : verdict === "LIKELY_FAKE" ? "fg-insight-fake" : "fg-insight-warn";

        // Score colors
        const fraudPct  = rob  ? Math.round((rob.fraud_probability  || 0) * 100) : null;
        const scamScore = scam ? (scam.scam_score || 0) : null;
        const robBarCls  = fraudPct === null ? "" : fraudPct >= 70 ? "fg-meter-bar-danger" : fraudPct >= 40 ? "fg-meter-bar-warn" : "fg-meter-bar-safe";
        const robValCls  = fraudPct === null ? "" : fraudPct >= 70 ? "fg-color-danger"     : fraudPct >= 40 ? "fg-color-warn"     : "fg-color-safe";
        const scamBarCls = scam?.risk_level === "HIGH" ? "fg-meter-bar-danger" : scam?.risk_level === "MEDIUM" ? "fg-meter-bar-warn" : "fg-meter-bar-safe";
        const scamValCls = scam?.risk_level === "HIGH" ? "fg-color-danger"     : scam?.risk_level === "MEDIUM" ? "fg-color-warn"     : "fg-color-safe";

        // Meters HTML
        const metersHtml = (fraudPct !== null || scamScore !== null) ? `
        <div class="fg-meters">
          ${fraudPct !== null ? `
          <div class="fg-meter">
            <div class="fg-meter-label">ML Fraud Score</div>
            <div class="fg-meter-bar-track"><div class="fg-meter-bar-fill ${robBarCls}" style="width:${fraudPct}%"></div></div>
            <div class="fg-meter-value ${robValCls}">${fraudPct}%</div>
            <div class="fg-meter-desc">${fraudPct >= 70 ? "High risk" : fraudPct >= 40 ? "Moderate risk" : "Low risk"}</div>
          </div>` : ""}
          ${scamScore !== null ? `
          <div class="fg-meter">
            <div class="fg-meter-label">Scam Signals</div>
            <div class="fg-meter-bar-track"><div class="fg-meter-bar-fill ${scamBarCls}" style="width:${Math.min(scamScore, 100)}%"></div></div>
            <div class="fg-meter-value ${scamValCls}">${scam.risk_level || "LOW"}</div>
            <div class="fg-meter-desc">Score: ${scamScore}/100</div>
          </div>` : ""}
        </div>` : "";

        // Tool grid
        const toolMeta = {
            scam_signals:       { icon: "🚨", label: "Scam Scan" },
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
        const toolEntries = Object.entries(toolResults).filter(([n]) => toolMeta[n]);
        const toolCardsHtml = toolEntries.map(([n, r]) => {
            const m    = toolMeta[n];
            const isErr = r?.error;
            const isBad = r?.is_fraud || r?.risk_level === "HIGH" || r?.verdict === "FRAUDULENT";
            const cls   = isErr ? "fg-tool-skip" : isBad ? "fg-tool-fail" : "fg-tool-pass";
            const lbl   = isErr ? "—" : isBad ? "ALERT" : "CLEAR";
            return `<div class="fg-tool-card ${cls}" title="${escapeHtml(n)}">
              <span class="fg-tool-icon">${m.icon}</span>
              <span class="fg-tool-name">${m.label}</span>
              <span class="fg-tool-status">${lbl}</span>
            </div>`;
        }).join("");

        // Key findings from report (extract bullet points)
        const findingLines = (report || "").split("\n")
            .filter(l => (l.startsWith("- ") || l.startsWith("* ")) && l.length > 10)
            .slice(0, 4);
        const findingsHtml = findingLines.length > 0 ? `
        <div class="fg-section">
          <div class="fg-section-header">
            <span class="fg-section-title">Key Findings</span>
          </div>
          <div class="fg-findings">
            ${findingLines.map((l, i) => {
                const text = l.slice(2).trim();
                const dot  = verdict === "LIKELY_FAKE" ? "fg-finding-red" : i === 0 && verdict === "SAFE" ? "fg-finding-green" : "fg-finding-yellow";
                return `<div class="fg-finding-item ${dot}"><div class="fg-finding-dot"></div><span>${escapeHtml(text)}</span></div>`;
            }).join("")}
          </div>
        </div>` : "";

        const overlay = document.createElement("div");
        overlay.id = "fg-overlay";
        overlay.innerHTML = `
      <!-- Hero -->
      <div class="fg-hero ${vc.hero}">
        <div class="fg-hero-top">
          <span class="fg-badge ${vc.badge}">
            <span class="fg-badge-icon">${vc.icon}</span>${vc.label}
          </span>
          <button class="fg-close-btn" id="fg-close-btn">✕</button>
        </div>
        <div class="fg-insight ${insightCls}">
          ${buildInsight(verdict, company, rob, scam)}
        </div>
      </div>

      <!-- Score meters -->
      ${metersHtml}

      <!-- Tool grid -->
      ${toolCardsHtml ? `
      <div class="fg-section">
        <div class="fg-section-header">
          <span class="fg-section-title">Investigation Tools</span>
          <span class="fg-section-count">${toolEntries.length} ran</span>
        </div>
        <div class="fg-tool-grid">${toolCardsHtml}</div>
      </div>` : ""}

      <!-- Key findings -->
      ${findingsHtml}

      <!-- Full report toggle -->
      ${report ? `
      <div class="fg-section">
        <button class="fg-toggle-btn" id="fg-report-toggle">
          <span id="fg-report-label">📋 Show Full Report</span>
          <span class="fg-toggle-arrow" id="fg-toggle-arrow">▼</span>
        </button>
        <div id="fg-report-content" style="display:none">
          <div class="fg-report-text" id="fg-report-text"></div>
        </div>
      </div>` : ""}

      <!-- Footer -->
      <div class="fg-footer">
        <span class="fg-footer-text">FraudGuard v2</span>
        <span class="fg-footer-dot"></span>
        <span class="fg-footer-text">RoBERTa + 13 Tools + GPT</span>
      </div>`;

        document.body.appendChild(overlay);

        document.getElementById("fg-close-btn").addEventListener("click", () => removeOverlay());

        if (report) {
            document.getElementById("fg-report-toggle").addEventListener("click", () => {
                const content = document.getElementById("fg-report-content");
                const arrow   = document.getElementById("fg-toggle-arrow");
                const lbl     = document.getElementById("fg-report-label");
                const open    = content.style.display === "none";
                content.style.display = open ? "block" : "none";
                if (open) document.getElementById("fg-report-text").innerHTML = mdToHtml(report);
                arrow.style.transform = open ? "rotate(180deg)" : "rotate(0deg)";
                lbl.textContent = open ? "📋 Hide Full Report" : "📋 Show Full Report";
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

    // ── SPA navigation handler (debounced — LinkedIn mutates DOM constantly) ──
    let lastUrl = location.href;
    let navDebounce = null;
    new MutationObserver(() => {
        if (navDebounce) return;                   // already scheduled, skip
        navDebounce = setTimeout(() => {
            navDebounce = null;
            if (location.href !== lastUrl) {
                lastUrl = location.href;
                removeOverlay();
                setTimeout(injectButton, 1500);
            }
        }, 500);                                    // check at most once per 500ms
    }).observe(document.body, { childList: true }); // childList only (no subtree)

    // ── Init ──────────────────────────────────────────────────
    injectButton();
})();
