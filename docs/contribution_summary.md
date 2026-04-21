# Team Contribution Summary

**Project:** FraudGuard — Fake Job Listing Detection using Deep Learning and Agentic AI
**Course:** DS & AI Lab Project
**Team:** Group 9

---

## 1. Contribution Table

| Team Member | M1 | M2 | M3 | M4 | M5 | M6 | Estimated Total |
|---|---|---|---|---|---|---|---|
| **Arun Dutta** | Literature Review, Gap Analysis | Data preprocessing, split strategy, project documentation | Pipeline verification, EDA, documentation | HPO analysis, training report | Results documentation, results synthesis | Final report, contribution summary | ~25% |
| **Hritik Roshan Maurya** | Problem framing, system architecture design | LangChain agent (`job_parser_agent.py`), structured field extraction CLI | Model design, training strategy review | Model validation, system integration testing | LangChain ReAct agent, real-world PDF/HTML testing | API documentation, deployment documentation | ~25% |
| **Vivek Bajaj** | Dataset identification, DL pipeline design | Primary dataset curation, baseline classifier, HuggingFace Hub deployment | Training pipeline, Focal Loss implementation | Optuna HPO (25 trials), threshold calibration, final eval | Threshold tuning (0.87), final test scripts, model artifacts | Technical documentation, architecture review | ~25% |
| **Vishwas Mehta** | Fraud pattern research, case study collection | Chrome extension development (v1) | Metadata anomaly detector (`IsolationForest` + rules engine) | Chrome extension integration with live model | Extension-LinkedIn integration, inference debugging, OOD validation | User guide, extension documentation | ~25% |

---

## 2. Individual Contribution Descriptions

### Arun Dutta

Arun served as the team's **literature review and documentation lead** across all milestones. In Milestone 1, he conducted an exhaustive review of existing fake job detection approaches, synthesizing results from 10+ papers spanning rule-based systems, classical ML, deep learning (CNN/LSTM/BiLSTM), and transformer-based methods (BERT, RoBERTa). His gap analysis clearly identified the absence of multi-step agentic verification and generative explanations in existing systems, directly motivating the project's architectural choices.

In Milestone 2, Arun designed and implemented the data preprocessing pipeline: missing value handling, stratified train/val/test splitting with `random_state=42`, and tokenization wrappers. He also structured the project's documentation files and managed task allocation for the team.

In Milestones 3 and 4, Arun contributed to pipeline verification (ensuring all 11 pipeline steps passed on a 50-sample CPU run) and wrote the detailed hyperparameter experiment documentation in the Milestone 4 report, clearly narrating the progression from v1 through v3_1. In Milestone 5, he coordinated the final report synthesis and ensured test results from `test_results.json` were accurately reflected in the evaluation documentation.

For Milestone 6, Arun authored the Final Project Report, contribution summary, and assisted in the overall documentation audit.

---

### Hritik Roshan Maurya

Hritik was the team's **agentic AI engineering lead**, responsible for building the intelligence layer that converts unstructured job documents into structured, analyzable data. In Milestone 1, he defined the problem statement, scoped the system objectives, and designed the high-level agentic workflow architecture — the blueprint from which all subsequent development followed.

In Milestone 2, Hritik built the `job_parser_agent.py` module from scratch: a LangChain-based agent that uses GPT structured output (`.with_structured_output()`) to extract 16 standardized features (matching the Kaggle EMSCAD schema) from any job document format (PDF, DOCX, HTML, Markdown, plain text). This agent became the foundation for both the `testing_work/AgenticWork/` CLI tool and the `web-app/core/job_parser_agent.py` production module.

In Milestones 3 and 4, Hritik collaborated on model architecture decisions and system integration. In Milestone 5, he led the **LangChain ReAct agent** development for orchestrating multi-tool fraud investigation and tested the model on real-world unstructured job data from PDFs and HTML scraped from LinkedIn, ensuring the pipeline generalized beyond the clean Kaggle dataset.

For Milestone 6, Hritik authored the API documentation, deployment details section of the technical doc, and the system design considerations.

---

### Vivek Bajaj

Vivek was the team's **model development and optimization lead**, responsible for the full ML training lifecycle. In Milestone 1, he identified the EMSCAD dataset as the primary benchmark and designed the high-level transformer training workflow.

In Milestone 2, Vivek built the first working fraud classifier notebook featuring RoBERTa-base, integrated Optuna for hyperparameter optimization, and deployed the initial model to HuggingFace Hub. He also designed the synthetic data augmentation strategy using GPT-4 for Milestone 2 (later evaluated in `v5_synth`).

Milestones 3 and 4 represent Vivek's core technical contributions: he ran the 25-trial Optuna HPO campaign that produced the final `v3_1` model, implemented the Focal Loss function with dynamic gamma tuning, executed the threshold calibration sweep that set the final operating threshold at 0.87, and managed all training artifacts (model weights, metrics JSONs, training curves). His systematic version-by-version ablation study (v1 through v5) is the methodological backbone of the Milestone 4 report.

In Milestone 5, Vivek tuned the final threshold to its reported value and ran the final test-set evaluation, producing the numbers reported throughout the documentation.

For Milestone 6, Vivek co-authored the technical documentation and reviewed the architecture documentation for accuracy.

---

### Vishwas Mehta

Vishwas was the team's **browser integration and applied ML lead**, responsible for bringing the fraud detection system to real-world users. In Milestone 1, he studied current fraud patterns, analyzed news reports and case studies from across India and globally, and contributed the "Current Status of Fake Job Listings" section with documented real-world cases.

In Milestone 2, Vishwas built the Chrome extension (v1) — the first end-to-end integration of the project's fraud detection into a real browser environment. The extension used vanilla JavaScript and a direct API call to provide instant fraud predictions on LinkedIn job pages.

In Milestone 3, Vishwas designed and implemented the **Metadata Anomaly Detector** module (`testing_work/src/tools/metadata_detector/`): a three-component system combining IsolationForest-based anomaly detection, a data-driven rules engine (with rules discovered via LightGBM feature importance, SHAP, and EBM analysis), and a combined risk scorer. This module provides an ML-based fraud signal that is complementary to the text-based RoBERTa classifier.

In Milestones 4 and 5, Vishwas upgraded the Chrome extension to v2.0 (LangChain-inspired multi-step pipeline with Gemini AI, link scraping, and real-time overlay UI), integrated it with LinkedIn's live job pages, and debugged the inference pipeline for edge cases (SPA navigation, API key management, result overlay rendering). He also led out-of-distribution validation on LinkedIn job postings to verify model behavior on real-world data beyond the Kaggle test set.

For Milestone 6, Vishwas authored the User Guide (particularly the Chrome extension installation and usage sections) and the extension documentation.

---

## 3. Cross-Milestone Collaboration

All four milestones from M3 onwards were completed through a highly collaborative approach. While primary responsibilities were distributed as above, all team members:

- Participated in weekly code review sessions
- Jointly debugged pipeline failures and edge cases
- Shared ownership of the final test results
- Contributed to report writing and presentation preparation

The team operated with continuous peer support and cross-functional overlap, ensuring no single point of failure in any milestone delivery.
