# Advance RAG Pipeline

Multimodal, layout-aware RAG over PDFs with **Qdrant** and **Gemini**.

## Pipeline
1. **Parse** (`ingestion.py`): PyMuPDF4LLM gives Markdown in correct reading order for multi-column pages, with tables as Markdown (kept whole, split by rows with repeated header if large).
2. **Visuals**: pages with images or chart-like vector drawings (and scanned pages) are rendered and described by Gemini vision into `figure` chunks.
3. **Embed + index** (`store.py`): Gemini `gemini-embedding-001` dense vectors + Qdrant BM25 sparse vectors in one collection.
4. **Retrieve**: dense + sparse prefetch fused with RRF inside Qdrant, optional source filter.
5. **Rerank**: Gemini scores candidates (falls back to fusion order on failure).
6. **Answer** (`pipeline.py`): grounded generation with `[n]` citations to file/page/type.

## Setup
```
pip install -r requirements.txt
copy .env.example .env      # set GEMINI_API_KEY (QDRANT_URL optional; empty = local on-disk Qdrant)
```
Put PDFs in `data/raw/`, then:
```
python scripts/cli.py index
python scripts/cli.py ask "your question"
```
Settings live in `configs/default.yaml`.

## Next steps
FastAPI service, eval set (`eval/`) with retrieval hit-rate and faithfulness checks, OCR fallback, incremental indexing.

## Web UI
```
uvicorn rag.api:app --app-dir src --port 8000
```
Open http://localhost:8000: upload PDFs, click **Load & index** to watch chunks load live, then ask questions (answers show section and page citations).
API: `GET /files`, `POST /upload`, `POST /index/stream` (progress events), `POST /ask`.
