// ============================================================
// FraudGuard v2 — Popup Script
// Manages backend URL + LLM config stored in chrome.storage.local
// ============================================================

const DEFAULT_BACKEND = "https://hrmhrmhrm-company-backend-api.hf.space";

document.addEventListener("DOMContentLoaded", () => {
    const backendUrlInput  = document.getElementById("backend-url");
    const llmApiKeyInput   = document.getElementById("llm-api-key");
    const llmBaseUrlInput  = document.getElementById("llm-base-url");
    const llmModelInput    = document.getElementById("llm-model");
    const saveBtn          = document.getElementById("save-btn");
    const testBtn          = document.getElementById("test-btn");
    const toggleKeyBtn     = document.getElementById("toggle-key-btn");
    const statusDot        = document.getElementById("status-dot");
    const statusText       = document.getElementById("status-text");
    const advancedToggle   = document.getElementById("advanced-toggle");
    const advancedSection  = document.getElementById("advanced-section");
    const advancedArrow    = document.getElementById("advanced-arrow");

    // ── Load saved settings ───────────────────────────────────
    chrome.storage.local.get(
        ["backendUrl", "llmApiKey", "llmBaseUrl", "llmModel"],
        (result) => {
            backendUrlInput.value = result.backendUrl || DEFAULT_BACKEND;
            if (result.llmApiKey)  llmApiKeyInput.value  = result.llmApiKey;
            if (result.llmBaseUrl) llmBaseUrlInput.value = result.llmBaseUrl;
            if (result.llmModel)   llmModelInput.value   = result.llmModel;

            // Check connection on load
            checkConnection(result.backendUrl || DEFAULT_BACKEND);
        }
    );

    // ── Save settings ─────────────────────────────────────────
    saveBtn.addEventListener("click", () => {
        const backendUrl = backendUrlInput.value.trim().replace(/\/$/, "");
        if (!backendUrl) {
            shake(backendUrlInput);
            return;
        }

        const toStore = { backendUrl };
        const llmApiKey  = llmApiKeyInput.value.trim();
        const llmBaseUrl = llmBaseUrlInput.value.trim();
        const llmModel   = llmModelInput.value.trim();
        if (llmApiKey)  toStore.llmApiKey  = llmApiKey;
        if (llmBaseUrl) toStore.llmBaseUrl = llmBaseUrl;
        if (llmModel)   toStore.llmModel   = llmModel;

        chrome.storage.local.set(toStore, () => {
            const btnText = saveBtn.querySelector(".btn-text");
            btnText.textContent = "✓ Saved!";
            saveBtn.classList.add("saved");
            setTimeout(() => {
                btnText.textContent = "Save Settings";
                saveBtn.classList.remove("saved");
            }, 2000);
            checkConnection(backendUrl);
        });
    });

    // ── Test connection ───────────────────────────────────────
    testBtn.addEventListener("click", () => {
        const url = (backendUrlInput.value.trim() || DEFAULT_BACKEND).replace(/\/$/, "");
        checkConnection(url);
    });

    async function checkConnection(url) {
        setStatus("checking", "Checking connection...");
        try {
            const resp = await fetch(`${url}/health`, { signal: AbortSignal.timeout(6000) });
            if (resp.ok) {
                // Also check LLM status
                try {
                    const llmResp = await fetch(`${url}/api/v1/llm/status`, { signal: AbortSignal.timeout(6000) });
                    const llmData = await llmResp.json();
                    if (llmData.ok && llmData.api_key_from_env) {
                        setStatus("ready", `Connected — LLM key set on server (${llmData.effective_model || ""})`);
                    } else if (llmData.ok) {
                        setStatus("partial", "Connected — No LLM key on server, enter one above");
                    } else {
                        setStatus("ready", "Backend connected");
                    }
                } catch {
                    setStatus("ready", "Backend connected");
                }
            } else {
                setStatus("error", `Backend returned ${resp.status}`);
            }
        } catch (err) {
            setStatus("error", `Cannot reach ${url}`);
        }
    }

    function setStatus(state, msg) {
        statusText.textContent = msg;
        statusDot.className = "status-dot " + state;
    }

    // ── Toggle LLM key visibility ─────────────────────────────
    toggleKeyBtn.addEventListener("click", () => {
        const isPassword = llmApiKeyInput.type === "password";
        llmApiKeyInput.type = isPassword ? "text" : "password";
        toggleKeyBtn.textContent = isPassword ? "🔒" : "👁️";
    });

    // ── Advanced settings toggle ──────────────────────────────
    advancedToggle.addEventListener("click", () => {
        const isOpen = advancedSection.style.display !== "none";
        advancedSection.style.display = isOpen ? "none" : "block";
        advancedArrow.textContent = isOpen ? "▼" : "▲";
    });

    // ── Enter key saves ───────────────────────────────────────
    [backendUrlInput, llmApiKeyInput, llmBaseUrlInput, llmModelInput].forEach(input => {
        input.addEventListener("keydown", (e) => {
            if (e.key === "Enter") saveBtn.click();
        });
    });

    // ── Shake animation helper ────────────────────────────────
    function shake(el) {
        el.style.animation = "shake 0.35s ease";
        el.style.borderColor = "rgba(248, 113, 113, 0.6)";
        setTimeout(() => {
            el.style.animation = "";
            el.style.borderColor = "";
        }, 350);
    }
});
