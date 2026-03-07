// ============================================================
// Background Service Worker — Gemini API Communication
// This is the file you'll swap out for your agentic flow later.
// ============================================================

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.type === "ANALYZE_JOB") {
        handleAnalyzeJob(request.data)
            .then((result) => sendResponse({ success: true, data: result }))
            .catch((error) =>
                sendResponse({ success: false, error: error.message })
            );
        return true;
    }
});

async function handleAnalyzeJob(jobData) {
    const apiKey = await getApiKey();
    if (!apiKey) {
        throw new Error(
            "No Gemini API key found. Click the extension icon to set your API key."
        );
    }
    const prompt = buildPrompt(jobData);
    const result = await callGeminiAPI(apiKey, prompt);
    return result;
}

function getApiKey() {
    return new Promise((resolve) => {
        chrome.storage.local.get(["geminiApiKey"], (result) => {
            resolve(result.geminiApiKey || null);
        });
    });
}
