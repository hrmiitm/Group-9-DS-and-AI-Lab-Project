# NotebookLM Prompt — FraudGuard Presentation Generation

## Instructions for Use

1. Open [NotebookLM](https://notebooklm.google.com/) (or any similar document-aware AI tool).
2. Upload the following source documents from this repository:
   - `docs/Final_Project_Report.md`
   - `docs/overview.md`
   - `docs/technical_doc.md`
   - `docs/user_guide.md`
   - `docs/contribution_summary.md`
   - `docs/future_work.md`
   - `README.md`
3. Copy and paste the prompt below into NotebookLM's query box.
4. The tool will generate output grounded in the uploaded source documents.

---

## The Prompt

```
You are a presentation writer and professional communicator helping a student team present their final DS & AI Lab project.

The project is called "FraudGuard" — an AI system that detects fake job listings using a fine-tuned RoBERTa transformer model combined with a 12-tool agentic verification pipeline and a Chrome browser extension.

Based ONLY on the source documents I have uploaded, please generate the following three outputs:

---

## OUTPUT 1: 12-SLIDE PRESENTATION OUTLINE WITH SPEAKER NOTES

Create a structured slide deck outline for a 12–15 minute presentation. For each slide, provide:
- A clear slide TITLE
- 3–5 bullet points of the key content for that slide
- SPEAKER NOTES (2–4 sentences that the presenter should say out loud for each slide)

The 12 slides should follow this structure:
1. Title Slide — Project name, team names, course, date
2. The Problem — Why fake job listings matter (use statistics and real examples from the docs)
3. Why Existing Solutions Fail — Brief gap analysis (rule-based → ML → DL → transformers)
4. Our Approach — System overview diagram (describe it in text: Input → RoBERTa → 12-tool pipeline → report)
5. Dataset and Preprocessing — EMSCAD dataset, class imbalance, preprocessing steps
6. Model Architecture — RoBERTa-base, Focal Loss, why full fine-tuning
7. Training and Hyperparameter Tuning — Optuna 25 trials, version progression (v1 to v3_1)
8. Evaluation Results — Metrics table, key finding (AUC 0.993, precision 0.957), comparison with baselines
9. The Web App Demo — What it does, 4-step pipeline, show what a result page contains
10. The Chrome Extension — How it works on LinkedIn, Gemini API, overlay design
11. Team Contributions — Who built what (table format, one line per person)
12. Conclusion and Future Work — What was achieved, key limitations, next steps

Keep the tone: CONFIDENT, CLEAR, and PROFESSIONAL. This is an academic project presentation — no hype, all substance.

---

## OUTPUT 2: DEMO VIDEO NARRATION SCRIPT

Write a 90–120 second narration script for a screen-recorded demo video. The script should:
- Open with one sentence explaining what the demo will show
- Narrate what is happening on screen at each step:
  * Typing a job description into the web-app
  * Clicking "Analyze"
  * Waiting (acknowledge the ~30 second processing time)
  * Walking through the results: verdict banner, tool grid, final report
  * Switching to the Chrome extension on LinkedIn
  * Clicking "Analyze Job"
  * Pointing out the overlay verdict
- Close with one sentence about what the viewer should take away
- Use natural, spoken English — not bullet points. It should sound like a person talking.

---

## OUTPUT 3: PRESENTATION TALKING POINTS GUIDE

Create a one-page cheat sheet (10–12 bullet points max) that the presenter can glance at during the Q&A after the presentation. Include:
- The most important number to remember (state the exact metric values)
- The one-sentence answer to "why RoBERTa?" 
- The one-sentence answer to "why Focal Loss?"
- The one-sentence answer to "why not just use GPT-4?"
- The one-sentence answer to "what's the biggest limitation?"
- A brief note on what "LIKELY_FAKE" means in the output (not a false positive machine)
- A brief note on the class imbalance problem and how it was addressed
- The relationship between the web-app and the Chrome extension
- Where to find the model weights (HuggingFace Hub)
- One honest statement about what the model doesn't do well (recall miss)

---

FORMAT REQUIREMENTS:
- Structure your output with clear headers: ## SLIDE DECK, ## DEMO NARRATION, ## TALKING POINTS
- Use markdown formatting throughout
- For the slide deck, use ### Slide 1, ### Slide 2, etc.
- Include **Speaker Notes:** sections in bold for each slide
- Keep total output under 3,000 words

Do not invent statistics or claims not present in the source documents. Ground every claim in the uploaded files.
```

---

## Expected Output Structure

When the prompt is submitted, NotebookLM should return three sections:

```
## SLIDE DECK
### Slide 1: Title
[content + speaker notes]
...
### Slide 12: Conclusion and Future Work
[content + speaker notes]

## DEMO NARRATION
[90–120 second script]

## TALKING POINTS
[10–12 bullet cheat sheet]
```

---

## Tips for Best Results

- If NotebookLM returns generic output, ask it to "be more specific — use the exact metric values from the documents."
- If a slide feels too long, ask it to "reduce Slide X to 3 bullet points maximum."
- For the speaker notes, ask it to "make the speaker notes sound natural and conversational, not like they're reading bullet points."
- The talking points should be printed on one page and given to the presenter before the Q&A begins.

---

## Supplementary Prompt: Slide Visuals Guidance

If you want to add visual suggestions to the slides, append this to the main prompt:

```
For each slide, also suggest one visual element that would strengthen the slide:
- A diagram type (flowchart, bar chart, table, etc.)
- What data or components it should show
- Where in the slide it should appear (full-width, right half, embedded)

Keep visual suggestions practical — only suggest charts that can be made from the data already in the documents.
```
