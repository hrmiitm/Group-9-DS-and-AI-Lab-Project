// ============================================================
// Popup Script — API Key Management
// ============================================================

document.addEventListener("DOMContentLoaded", () => {
    const apiKeyInput = document.getElementById("api-key");
    const saveBtn = document.getElementById("save-btn");
    const toggleKeyBtn = document.getElementById("toggle-key");
    const statusDot = document.getElementById("status-dot");
    const statusText = document.getElementById("status-text");

    // ── Load saved key ───────────────────────────────────────
    chrome.storage.local.get(["geminiApiKey"], (result) => {
        if (result.geminiApiKey) {
            apiKeyInput.value = result.geminiApiKey;
            setStatus(true);
        } else {
            setStatus(false);
        }
    });

    // ── Save key ─────────────────────────────────────────────
    saveBtn.addEventListener("click", () => {
        const key = apiKeyInput.value.trim();

        if (!key) {
            shakeButton();
            return;
        }

        chrome.storage.local.set({ geminiApiKey: key }, () => {
            // Visual feedback
            saveBtn.classList.add("saved");
            saveBtn.querySelector(".btn-text").textContent = "✓ Saved!";
            setStatus(true);

            setTimeout(() => {
                saveBtn.classList.remove("saved");
                saveBtn.querySelector(".btn-text").textContent = "Save Key";
            }, 2000);
        });
    });

    // ── Toggle key visibility ────────────────────────────────
    toggleKeyBtn.addEventListener("click", () => {
        if (apiKeyInput.type === "password") {
            apiKeyInput.type = "text";
            toggleKeyBtn.textContent = "🔒";
        } else {
            apiKeyInput.type = "password";
            toggleKeyBtn.textContent = "👁️";
        }
    });

    // ── Helpers ──────────────────────────────────────────────
    function setStatus(active) {
        if (active) {
            statusDot.className = "status-dot active";
            statusText.textContent = "API key configured — Ready to analyze";
        } else {
            statusDot.className = "status-dot inactive";
            statusText.textContent = "No API key set — Enter key below";
        }
    }

    function shakeButton() {
        saveBtn.style.animation = "shake 0.4s ease";
        apiKeyInput.style.borderColor = "rgba(255, 107, 107, 0.5)";

        setTimeout(() => {
            saveBtn.style.animation = "";
            apiKeyInput.style.borderColor = "";
        }, 400);
    }
});
