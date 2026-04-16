# Licenses and Attributions

**Project:** FraudGuard — Fake Job Listing Detection
**Last Updated:** 2026-04-15

---

## 1. Project License

This project is released under the **MIT License**.

```
MIT License

Copyright (c) 2026 Group 9 — DS & AI Lab Project
(Arun Dutta, Hritik Roshan Maurya, Vivek Bajaj, Vishwas Mehta)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 2. Dataset License

### EMSCAD — Employment Scam Aegean Dataset

| Attribute | Value |
|---|---|
| **Name** | Fake Job Postings (EMSCAD) |
| **Source** | University of the Aegean, Department of Information and Communication Systems Engineering |
| **Kaggle Mirror** | [shivamb/real-or-fake-fake-jobposting-prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) |
| **License** | CC BY-SA 4.0 (Creative Commons Attribution-ShareAlike 4.0 International) |
| **Records** | 17,880 job postings (866 fraudulent) |

**Citation:**
```
Vidros, S., Kolias, C., Kambourakis, G., & Maglaras, L. (2017).
Automatic Detection of Online Recruitment Frauds: Characteristics, Methods, and a Public Dataset.
Future Internet, 9(1), 6.
https://doi.org/10.3390/fi9010006
```

Under CC BY-SA 4.0, you are free to use, share, and adapt the dataset provided you:
- Give appropriate credit to the original authors
- Indicate if changes were made
- Distribute your contributions under the same license

---

## 3. Pre-Trained Model License

### RoBERTa-base (Facebook AI Research)

| Attribute | Value |
|---|---|
| **Model** | `roberta-base` |
| **Organization** | Facebook AI Research (Meta AI) |
| **HuggingFace Model Card** | [facebook/roberta-base](https://huggingface.co/facebook/roberta-base) |
| **License** | MIT License |
| **Paper** | Liu et al. (2019), "RoBERTa: A Robustly Optimized BERT Pretraining Approach" |

**Citation:**
```
Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M.,
Zettlemoyer, L., & Stoyanov, V. (2019).
RoBERTa: A Robustly Optimized BERT Pretraining Approach.
arXiv:1907.11692.
https://arxiv.org/abs/1907.11692
```

### Fine-Tuned Model (This Project)

| Attribute | Value |
|---|---|
| **Model** | `aditya963/fraud-job-classifier` |
| **HuggingFace Hub** | [aditya963/fraud-job-classifier](https://huggingface.co/aditya963/fraud-job-classifier) |
| **License** | MIT License |
| **Based On** | `facebook/roberta-base` (MIT) |
| **Training Data** | EMSCAD dataset (CC BY-SA 4.0) |

Note: Because the training dataset is CC BY-SA 4.0, any derivative works based on the trained model that incorporate dataset outputs may be subject to the ShareAlike requirement. Consult a legal professional for production deployments.

---

## 4. Third-Party Library Licenses

| Library | License | Usage |
|---|---|---|
| PyTorch (`torch`) | BSD-style | Deep learning framework |
| HuggingFace Transformers | Apache 2.0 | RoBERTa model + tokenizer |
| HuggingFace Datasets | Apache 2.0 | Dataset loading utilities |
| scikit-learn | BSD-3-Clause | Train/test splitting, metrics |
| Optuna | MIT | Bayesian hyperparameter optimization |
| Flask | BSD-3-Clause | Web application framework |
| LangChain (`langchain`, `langchain-openai`, `langchain-community`) | MIT | LLM orchestration and tool chaining |
| Pydantic | MIT | Data validation (JobPosting schema) |
| DuckDuckGo Search (`ddgs`) | MIT | Web and news search for verification tools |
| Trafilatura | Apache 2.0 | Website content extraction |
| python-whois | MIT | Domain WHOIS lookups |
| email-validator | CC0 / Public Domain | Email syntax and MX verification |
| phonenumbers | Apache 2.0 | Phone number parsing and validation |
| Requests | Apache 2.0 | HTTP requests |
| BeautifulSoup4 | MIT | HTML parsing (LinkedIn scraper) |
| Pandas | BSD-3-Clause | Data manipulation |
| NumPy | BSD-3-Clause | Numerical computing |
| Markupsafe | BSD-3-Clause | HTML escaping in Flask |
| python-docx / docx2txt | MIT | Word document parsing |
| PyPDF / pypdfium2 | MIT / Apache 2.0 | PDF parsing |
| Unstructured | Apache 2.0 | Document loading (DOCX, HTML, MD) |
| Jinja2 | BSD-3-Clause | HTML templating |
| Werkzeug | BSD-3-Clause | Flask WSGI toolkit |
| TensorBoard (optional) | Apache 2.0 | Training visualization |

---

## 5. AI Service Terms of Use

This project integrates with the following external AI services. Users must comply with their respective terms:

| Service | Usage in Project | Terms Reference |
|---|---|---|
| **OpenRouter** (via AIPipe) | LLM calls for job parsing, tool inference, fraud reports | [openrouter.ai/terms](https://openrouter.ai/terms) |
| **Google Gemini API** | Chrome extension job analysis | [Google AI Terms of Service](https://ai.google.dev/terms) |
| **HuggingFace Hub** | Model hosting and download | [HuggingFace Terms of Service](https://huggingface.co/terms-of-service) |

---

## 6. Data Privacy Notice

This project processes text from job postings which may contain personally identifiable information (PII) such as:
- Company names and contact details
- Email addresses and phone numbers
- Job seeker information (if analyzing resumes)

**Storage:** Analysis results are stored locally in `web-app/results/<uuid>.json`. No data is sent to external servers except:
- The job text sent to the configured LLM API (OpenRouter/AIPipe)
- Verification tool queries (company names, domains) sent to public services (DuckDuckGo, WHOIS)

For production deployments, implement appropriate data retention policies and GDPR/DPDP compliance measures.
