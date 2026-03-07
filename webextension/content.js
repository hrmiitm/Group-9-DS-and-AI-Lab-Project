// ============================================================
// Content Script — Injected on LinkedIn pages
// Handles: Button injection on LinkedIn
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
        document.body.appendChild(btn);
    }

    // Initialize
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", createAnalyzeButton);
    } else {
        createAnalyzeButton();
    }
})();
