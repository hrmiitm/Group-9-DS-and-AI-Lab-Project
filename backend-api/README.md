---
title: FraudGuard Backend API
emoji: 🛡️
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# FraudGuard Backend API

Async FastAPI service for fake job posting detection. It provides 13 investigative tools and an RoBERTa ML model classifier endpoint.

## Local Development

1. **Create and activate a virtual environment:**
   ```bash
   cd backend-api
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the server:**
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```
   The API will be available at `http://localhost:8000`. Swagger docs at `http://localhost:8000/docs`.

## Deployment to HuggingFace Spaces

This backend is designed to be fully compatible with HuggingFace Spaces (Docker template).

1. **Create a New Space:**
   - Go to [HuggingFace Spaces](https://huggingface.co/spaces) and click "Create new Space".
   - Select **Docker** as the Space SDK.
   - Choose a blank template.

2. **Upload Files:**
   - Upload all files from the `backend-api/` directory (including `Dockerfile`, `app.py`, `requirements.txt`, etc.) to the root of your HuggingFace Space. You can do this via the HF web interface or by cloning the space's git repository.

3. **Configure Secrets (Environment Variables):**
   - Go to your Space's **Settings** -> **Variables and secrets**.
   - Add the following secrets (optional, but highly recommended for the LLM to work out of the box):
     - `OPENAI_API_KEY`: Your LLM API key (OpenAI, OpenRouter, AIPipe token, etc.).
     - `OPENAI_BASE_URL`: (Optional) Custom base URL if using a proxy like AIPipe or OpenRouter.
     - `LLM_MODEL`: (Optional) The model to use by default (e.g., `openai/gpt-4o-mini`).

4. **Build and Run:**
   - HuggingFace will automatically build the Docker image and deploy the FastAPI app on port `7860`.
   - Your API will be live at `https://[your-username]-[space-name].hf.space`.

## Endpoints Summary

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check + LLM config status |
| GET | `/api/v1/tools` | List all 13 tools with metadata |
| GET | `/api/v1/tools/{name}` | Single tool metadata |
| POST | `/api/v1/run/{tool_name}` | Execute any tool dynamically |
| POST | `/api/v1/run-batch` | Execute multiple tools concurrently |
| POST | `/api/v1/llm/extract` | Parse JD text → structured fields |
| POST | `/api/v1/llm/deep-research` | Recover missing fields via web search |
| POST | `/api/v1/llm/tool-inference` | 2-4 bullet analysis of tool output |
| POST | `/api/v1/llm/final-summary` | Compile all inferences → fraud verdict |
