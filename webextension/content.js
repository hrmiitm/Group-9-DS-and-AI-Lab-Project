// ============================================================
// Content Script — Injected on LinkedIn pages
// Handles: Button injection, DOM scraping, Results overlay
// ============================================================

(function () {
    "use strict";

    // Prevent double injection
    if (window.__linkedinJobPredictor) return;
    window.__linkedinJobPredictor = true;

    // ── Floating Analyze Button ──────────────────────────────
    function createAnalyzeButton() {
        if (document.getElementById("ljp-analyze-btn")) return;

        const btn = document.createElement("button");
        btn.id = "ljp-analyze-btn";
        btn.innerHTML = `
      <span class="ljp-btn-icon">🔍</span>
      <span class="ljp-btn-text">Analyze Job</span>
    `;
        btn.addEventListener("click", handleAnalyzeClick);
        document.body.appendChild(btn);
    }

    // ── Scrape Job Data ──────────────────────────────────────
    function scrapeJobData() {
        const data = {};

        // Job Title — try multiple selectors (LinkedIn A/B tests class names)
        data.title =
            getTextContent(".job-details-jobs-unified-top-card__job-title h1") ||
            getTextContent(".job-details-jobs-unified-top-card__job-title") ||
            getTextContent(".jobs-unified-top-card__job-title") ||
            getTextContent(".t-24.t-bold.inline") ||
            getTextContent(".top-card-layout__title") ||
            getTextContent("h1.topcard__title") ||
            getTextContent("h2.t-24.t-bold") ||
            getTextContent("h1") ||
            "";

        // Company Name
        data.company =
            getTextContent(".job-details-jobs-unified-top-card__company-name a") ||
            getTextContent(".job-details-jobs-unified-top-card__company-name") ||
            getTextContent(".jobs-unified-top-card__company-name a") ||
            getTextContent(".jobs-unified-top-card__company-name") ||
            getTextContent(".topcard__org-name-link") ||
            getTextContent(".top-card-layout__flavor--black-link") ||
            getTextContent("a.topcard__org-name-link") ||
            "";

        // Location
        data.location =
            getTextContent(".job-details-jobs-unified-top-card__bullet") ||
            getTextContent(".jobs-unified-top-card__bullet") ||
            getTextContent(".topcard__flavor--bullet") ||
            getTextContent(".top-card-layout__second-subline span") ||
            "";

        // Workplace Type (Remote, Hybrid, On-site)
        data.workplaceType =
            getTextContent(".job-details-jobs-unified-top-card__workplace-type") ||
            getTextContent(".jobs-unified-top-card__workplace-type") ||
            "";

        // Posted Date
        data.postedDate =
            getTextContent(".job-details-jobs-unified-top-card__posted-date") ||
            getTextContent(".jobs-unified-top-card__posted-date") ||
            getTextContent(".posted-time-ago__text") ||
            "";

        // Applicant Count
        data.applicantCount =
            getTextContent(".job-details-jobs-unified-top-card__applicant-count") ||
            getTextContent(".jobs-unified-top-card__applicant-count") ||
            getTextContent(".num-applicants__caption") ||
            "";

        // Job Description — critical field, try many selectors
        data.description =
            getTextContent(".jobs-description-content__text") ||
            getTextContent(".jobs-description__content") ||
            getTextContent(".jobs-box__html-content") ||
            getTextContent("#job-details") ||
            getTextContent(".job-details-jobs-unified-top-card__job-description") ||
            getTextContent(".description__text") ||
            getTextContent(".show-more-less-html__markup") ||
            getTextContent("article.jobs-description") ||
            // Fallback: try to find any "About the job" section
            getDescriptionFromAboutSection() ||
            "";

        // Salary
        data.salary =
            getTextContent(".job-details-jobs-unified-top-card__job-insight--highlight") ||
            getTextContent(".salary-main-rail__data-body") ||
            getTextContent(".compensation__salary") ||
            "";

        // Job criteria items (Seniority, Employment Type, etc.)
        const criteriaItems = document.querySelectorAll(
            ".job-details-jobs-unified-top-card__job-insight, .jobs-unified-top-card__job-insight, .description__job-criteria-item, .jobs-box__list-item"
        );
        criteriaItems.forEach((item) => {
            const text = item.textContent.trim();
            const label = (
                item.querySelector(
                    ".job-details-jobs-unified-top-card__job-insight-view-model-secondary, h3, .t-black--light"
                )?.textContent || ""
            ).trim().toLowerCase();
            const value = (
                item.querySelector("span:last-child, .description__job-criteria-text, .t-black.t-normal")
                    ?.textContent || ""
            ).trim();

            if (label.includes("seniority") || text.toLowerCase().includes("seniority"))
                data.seniorityLevel = value || text;
            if (label.includes("employment") || text.toLowerCase().includes("full-time") || text.toLowerCase().includes("part-time"))
                data.employmentType = value || text;
            if (label.includes("function") || text.toLowerCase().includes("function"))
                data.jobFunction = value || text;
            if (label.includes("industr") || text.toLowerCase().includes("industr"))
                data.industries = value || text;
        });

        // Company About / Description
        data.companyDescription =
            getTextContent(".jobs-company__company-description") ||
            getTextContent(".top-card-layout__card .topcard__flavor--metadata") ||
            "";

        // Clean up whitespace
        Object.keys(data).forEach((key) => {
            if (typeof data[key] === "string") {
                data[key] = data[key].replace(/\s+/g, " ").trim();
            }
        });

        console.log("[Content] Scraped data fields:", Object.keys(data).filter(k => data[k]).length, "non-empty");
        console.log("[Content] Title:", data.title?.substring(0, 60));
        console.log("[Content] Description length:", data.description?.length);

        return data;
    }

    /**
     * Fallback: try to find job description from "About the job" section
     * by walking the DOM for known heading patterns.
     */
    function getDescriptionFromAboutSection() {
        // Look for "About the job" heading and get its next sibling content
        const headings = document.querySelectorAll("h2, h3, .t-20.t-bold, .t-16.t-bold");
        for (const heading of headings) {
            const text = heading.textContent.trim().toLowerCase();
            if (text.includes("about the job") || text.includes("about this role")) {
                // Get the next sibling element's text
                let sibling = heading.nextElementSibling;
                while (sibling) {
                    const content = sibling.textContent.trim();
                    if (content.length > 50) {
                        return content;
                    }
                    sibling = sibling.nextElementSibling;
                }
                // Try parent's children
                const parent = heading.closest("section, div, article");
                if (parent) {
                    const fullText = parent.textContent.trim();
                    if (fullText.length > 100) return fullText;
                }
            }
        }
        return null;
    }

    function getTextContent(selector) {
        try {
            const el = document.querySelector(selector);
            if (!el) return null;
            const text = el.textContent.trim();
            return text.length > 0 ? text : null;
        } catch {
            return null;
        }
    }

    /**
     * Wait for job content to load in the DOM (LinkedIn loads async).
     * Retries up to maxAttempts with a delay between each attempt.
     */
    async function waitForJobContent(maxAttempts = 3, delayMs = 1000) {
        for (let attempt = 0; attempt < maxAttempts; attempt++) {
            const data = scrapeJobData();
            if (data.title || data.description) {
                return data;
            }
            console.log(`[Content] Waiting for job content (attempt ${attempt + 1}/${maxAttempts})...`);
            await new Promise(resolve => setTimeout(resolve, delayMs));
        }
        return scrapeJobData(); // Final attempt
    }

    // ── Extract Links from Job Description DOM ────────────────
    function extractLinksFromDOM() {
        const links = [];
        const linkSeen = new Set();

        // Selectors for job description areas
        const descriptionSelectors = [
            '.jobs-description__content',
            '.jobs-description',
            '.job-details-jobs-unified-top-card__job-insight',
            '.jobs-unified-top-card__job-insight',
            '.jobs-box__html-content',
            '.jobs-description-content__text',
            '[class*="description"]',
            '[class*="job-details"]',
        ];

        // Find the description container
        let descContainer = null;
        for (const sel of descriptionSelectors) {
            descContainer = document.querySelector(sel);
            if (descContainer) break;
        }

        // If we found a description container, extract links from it
        if (descContainer) {
            const anchors = descContainer.querySelectorAll('a[href]');
            for (const anchor of anchors) {
                const href = anchor.href;
                const text = anchor.textContent.trim();

                if (href && !linkSeen.has(href) && href.startsWith('http')) {
                    linkSeen.add(href);
                    links.push({
                        url: href,
                        text: text || href,
                        source: 'job_description_dom',
                    });
                }
            }
        }

        // Also check the "About the company" section
        const aboutSection = document.querySelector('.jobs-company__box');
        if (aboutSection) {
            const aboutAnchors = aboutSection.querySelectorAll('a[href]');
            for (const anchor of aboutAnchors) {
                const href = anchor.href;
                if (href && !linkSeen.has(href) && href.startsWith('http')) {
                    linkSeen.add(href);
                    links.push({
                        url: href,
                        text: anchor.textContent.trim() || href,
                        source: 'company_section_dom',
                    });
                }
            }
        }

        // Check for links in the "How you match" or insights section
        const insightLinks = document.querySelectorAll(
            '.job-details-how-you-match a[href], .jobs-unified-top-card a[href]'
        );
        for (const anchor of insightLinks) {
            const href = anchor.href;
            if (href && !linkSeen.has(href) && href.startsWith('http')) {
                linkSeen.add(href);
                links.push({
                    url: href,
                    text: anchor.textContent.trim() || href,
                    source: 'insights_dom',
                });
            }
        }

        console.log(`[Content] Extracted ${links.length} links from DOM`);
        return links;
    }

    // ── Progress Indicator ────────────────────────────────────
    function showProgressIndicator(progress) {
        let indicator = document.getElementById('ljp-progress');

        if (!indicator) {
            indicator = document.createElement('div');
            indicator.id = 'ljp-progress';
            document.body.appendChild(indicator);
        }

        const percentage = Math.round((progress.step / progress.totalSteps) * 100);
        const statusEmoji = progress.status === 'complete' ? '✅' :
            progress.status === 'failed' ? '❌' : '⏳';

        indicator.innerHTML = `
            <div class="ljp-progress-content">
                <div class="ljp-progress-header">🛡️ Analyzing Job</div>
                <div class="ljp-progress-step">
                    ${statusEmoji} Step ${progress.step}/${progress.totalSteps}: ${progress.label}
                </div>
                <div class="ljp-progress-bar-container">
                    <div class="ljp-progress-bar-fill" style="width: ${percentage}%"></div>
                </div>
                <div class="ljp-progress-percent">${percentage}%</div>
            </div>
        `;

        indicator.classList.add('ljp-progress-visible');

        // Auto-hide after last step completes
        if (progress.step === progress.totalSteps &&
            (progress.status === 'complete' || progress.status === 'failed')) {
            setTimeout(() => {
                indicator.classList.remove('ljp-progress-visible');
                setTimeout(() => indicator.remove(), 300);
            }, 1500);
        }
    }

    // Listen for progress updates from background
    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
        if (message.type === 'ANALYSIS_PROGRESS') {
            showProgressIndicator(message.data);
        }
    });

    // ── Handle Analyze Click ─────────────────────────────────
    async function handleAnalyzeClick() {
        const btn = document.getElementById("ljp-analyze-btn");

        // Show loading state
        btn.classList.add("ljp-loading");
        btn.innerHTML = `
      <span class="ljp-spinner"></span>
      <span class="ljp-btn-text">Analyzing...</span>
    `;
        btn.disabled = true;

        try {
            // Wait for job content to load (LinkedIn loads async)
            const jobData = await waitForJobContent(3, 1000);

            // Extract links from the job description DOM
            const domLinks = extractLinksFromDOM();

            // Validate we have something to analyze
            if (!jobData.title && !jobData.description) {
                throw new Error(
                    "No job data found on this page. Make sure you're viewing a specific job listing (click on a job from the list)."
                );
            }

            console.log(`[Content] Sending to pipeline: ${Object.keys(jobData).length} fields, ${domLinks.length} DOM links`);

            // Send job data + DOM links to background pipeline
            const response = await chrome.runtime.sendMessage({
                type: "ANALYZE_JOB",
                data: {
                    jobData: jobData,
                    domLinks: domLinks,
                },
            });

            if (response.success) {
                showResultsOverlay(response.data, jobData);
            } else {
                showErrorOverlay(response.error);
            }
        } catch (error) {
            showErrorOverlay(error.message);
        } finally {
            // Reset button
            btn.classList.remove("ljp-loading");
            btn.innerHTML = `
        <span class="ljp-btn-icon">🔍</span>
        <span class="ljp-btn-text">Analyze Job</span>
      `;
            btn.disabled = false;
        }
    }

    function escapeHtml(str) {
        const div = document.createElement("div");
        div.textContent = str;
        return div.innerHTML;
    }

    // ── Results Overlay ──────────────────────────────────────
    function showResultsOverlay(result, jobData) {
        removeOverlay();

        const verdictConfig = {
            SAFE: { emoji: "✅", label: "Safe to Apply", color: "#00c853", bg: "rgba(0, 200, 83, 0.1)" },
            SUSPICIOUS: { emoji: "⚠️", label: "Suspicious", color: "#ff9100", bg: "rgba(255, 145, 0, 0.1)" },
            LIKELY_FAKE: { emoji: "❌", label: "Likely Fake", color: "#ff1744", bg: "rgba(255, 23, 68, 0.1)" },
        };

        const v = verdictConfig[result.verdict] || verdictConfig["SUSPICIOUS"];

        const overlay = document.createElement("div");
        overlay.id = "ljp-overlay";
        overlay.innerHTML = `
      <div class="ljp-overlay-backdrop" id="ljp-backdrop"></div>
      <div class="ljp-results-panel">
        <div class="ljp-results-header">
          <div class="ljp-results-title">
            <span class="ljp-logo">🛡️</span>
            Job Legitimacy Report
          </div>
          <button class="ljp-close-btn" id="ljp-close-btn">✕</button>
        </div>
        <div class="ljp-job-info">
          <div class="ljp-job-name">${escapeHtml(jobData.title || "Unknown Job")}</div>
          <div class="ljp-company-name">${escapeHtml(jobData.company || "Unknown Company")}</div>
        </div>
        <div class="ljp-verdict-card" style="background: ${v.bg}; border-left: 4px solid ${v.color};">
          <div class="ljp-verdict-row">
            <span class="ljp-verdict-emoji">${v.emoji}</span>
            <div>
              <div class="ljp-verdict-label" style="color: ${v.color};">${v.label}</div>
              <div class="ljp-confidence">Confidence: ${result.confidence}%</div>
            </div>
          </div>
          <div class="ljp-confidence-bar">
            <div class="ljp-confidence-fill" style="width: ${result.confidence}%; background: ${v.color};"></div>
          </div>
        </div>
        <div class="ljp-section">
          <div class="ljp-section-title">📋 Summary</div>
          <p class="ljp-summary">${escapeHtml(result.summary || "")}</p>
        </div>
        <div class="ljp-section">
          <div class="ljp-section-title">🔎 Key Findings</div>
          <ul class="ljp-reasons">
            ${(result.reasons || []).map((r) => `<li>${escapeHtml(r)}</li>`).join("")}
          </ul>
        </div>
        ${result.tips ? `
        <div class="ljp-section ljp-tip">
          <div class="ljp-section-title">💡 Tip</div>
          <p>${escapeHtml(result.tips)}</p>
        </div>` : ""}
        <div class="ljp-footer">Powered by Gemini AI · Results are advisory only</div>
      </div>
    `;

        document.body.appendChild(overlay);
        requestAnimationFrame(() => overlay.classList.add("ljp-visible"));

        document.getElementById("ljp-close-btn").addEventListener("click", removeOverlay);
        document.getElementById("ljp-backdrop").addEventListener("click", removeOverlay);
    }

    function showErrorOverlay(message) {
        removeOverlay();

        const overlay = document.createElement("div");
        overlay.id = "ljp-overlay";
        overlay.innerHTML = `
      <div class="ljp-overlay-backdrop" id="ljp-backdrop"></div>
      <div class="ljp-results-panel ljp-error-panel">
        <div class="ljp-results-header">
          <div class="ljp-results-title"><span class="ljp-logo">⚠️</span> Analysis Error</div>
          <button class="ljp-close-btn" id="ljp-close-btn">✕</button>
        </div>
        <div class="ljp-error-message"><p>${escapeHtml(message)}</p></div>
        <div class="ljp-footer">Make sure your Gemini API key is set correctly</div>
      </div>
    `;

        document.body.appendChild(overlay);
        requestAnimationFrame(() => overlay.classList.add("ljp-visible"));

        document.getElementById("ljp-close-btn").addEventListener("click", removeOverlay);
        document.getElementById("ljp-backdrop").addEventListener("click", removeOverlay);
    }

    function removeOverlay() {
        const overlay = document.getElementById("ljp-overlay");
        if (overlay) {
            overlay.classList.remove("ljp-visible");
            setTimeout(() => overlay.remove(), 300);
        }
    }

    // ── Initialize ───────────────────────────────────────────
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", createAnalyzeButton);
    } else {
        createAnalyzeButton();
    }

    // Re-inject button on SPA navigation
    const observer = new MutationObserver(() => {
        if (!document.getElementById("ljp-analyze-btn")) {
            createAnalyzeButton();
        }
    });
    observer.observe(document.body, { childList: true, subtree: true });
})();
