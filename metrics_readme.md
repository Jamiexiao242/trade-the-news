## Metrics quick run

You need two processes:
1. Metrics API (reads `METRICS_PATH`, default `metrics.db/jsonl`)
2. Frontend (polls the API and shows charts)

### Run API (FastAPI/Uvicorn)
```bash
METRICS_PATH=metrics.db uvicorn metrics_api:app --host 0.0.0.0 --port 2010
```
If you prefer jsonl:
```bash
METRICS_PATH=metrics.jsonl uvicorn metrics_api:app --host 0.0.0.0 --port 2010
```
The API endpoint: `http://localhost:2010/metrics?limit=500`

### Run frontend (without Node: use Vite dev server pre-bundled)
If you have npm/yarn:
```bash
# one-time: npm install
METRICS_API=http://localhost:2010 npm run dev
```
If you cannot use Node, you can open a simple static server with Python pointing to a prebuilt bundle (not included here) or ask me to add a tiny HTML that fetches `/metrics`. For now, the React component `web/metrics_analyzer.tsx` expects `METRICS_API` env or defaults to `/metrics`.

### Single command example
Use two terminals:
- Terminal 1: run the API (command above)
- Terminal 2: run `npm run dev` (with METRICS_API set)

### Notes
- Ensure `metrics.db` has the `events` table; API will auto-create if missing.
- If you prefer a single port, place a reverse proxy to forward `/metrics` to the API port.
