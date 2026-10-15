# DocuMind AI — lightweight Gemini deployment update

This pack replaces the backend API module with a Gemini API-powered RAG implementation and provides a lightweight dependency list for Render. The current React UI routes (`/documents`, `/knowledge/ask`, and document deletion) are kept compatible.

## Apply files

1. Back up your current root `api.py` (for example, rename it to `api_local_backup.py`).
2. Copy this pack's `api.py` into the project root, replacing the existing `api.py`.
3. Copy `requirements-deploy.txt`, `render.yaml`, and `.env.example` into the project root.
4. Do not replace the existing `requirements.txt`; it is still useful for the local model/evaluation scripts.
5. `.env` is for local use only and must never be committed. Hosting uses Render's Environment settings.

## Local test

Run in the project-root terminal:

```cmd
python -m pip install -r requirements-deploy.txt
```

Copy `.env.example` to `.env`, replace the placeholder with your private Gemini API key, then run:

```cmd
python -m py_compile api.py
python -m pytest -v
uvicorn api:app --reload
```

Open `http://127.0.0.1:8000/` and expect a JSON health response. Test uploads/questions through the existing UI. The old local database will be re-embedded using Gemini on first knowledge-base question if it contains old-model vectors; this uses API quota.

## Render backend

1. Push the updated project to GitHub without `.env`.
2. On Render, choose **New + → Blueprint**, connect `aashalohana/DocuMindAI`, and use the root `render.yaml`.
3. When prompted, enter the Gemini key into the secret environment variable `GEMINI_API_KEY` on Render. Never put it into the repository.
4. After Render deploys, copy the service's public `https://....onrender.com` URL.
5. Deploy the Vercel frontend from the same repo with Root Directory `frontend`, Framework `Vite`, Build Command `npm run build`, Output Directory `dist`, and Environment Variable `VITE_API_URL` set to the Render backend URL.
6. After Vercel gives you a public `https://....vercel.app` URL, go back to Render → Environment, set `FRONTEND_ORIGINS` to that exact URL (no trailing slash), save and redeploy.

## Limits and safety

- Render's free backend sleeps when idle and its local SQLite filesystem is ephemeral, so demo uploads/library entries can be lost after a sleep/restart/redeploy. This is suitable for a portfolio demo, not persistent production use.
- Use synthetic/public sample documents only. Google's Gemini free-tier usage may be processed to improve products according to Google's current terms/pricing notes.
- `MIN_SIMILARITY_SCORE=0.35` is a provisional threshold after changing embedding models, not a validated quality guarantee. Test multiple in-scope and out-of-scope questions.
