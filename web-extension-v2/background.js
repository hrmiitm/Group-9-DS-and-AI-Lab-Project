// FraudGuard v2 — Background Service Worker (minimal)
// All analysis logic is now in content.js to avoid MV3 SW termination issues.

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === "FG_PING") {
        sendResponse({ ok: true, version: "2.0" });
    }
    return false;
});
