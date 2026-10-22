# Advance RAG Pipeline

Ask questions about your PDFs and get short, cited answers. Built with **Qdrant** (vector database) and **Gemini** (embeddings, chart reading, answers). Designed for documents with multiple columns, tables, charts and images.

**What makes it different from a basic RAG:**

- Reads PDFs in the right order across columns and keeps tables as tables.
- Chunks never cut a sentence in half.
- Searches small chunks, then answers from the bigger section around them (parent-child).
- Combines meaning search and keyword search, then re-ranks with a local model.
- Writes 3 draft answers and keeps the best-supported one; says "I don't know" when the documents don't contain the answer.
- Everything runs on free tools (Gemini free tier, local Qdrant or a free Qdrant cloud cluster, a small local reranker).

---

## How it works

```
 UPLOAD ──► INDEXING (once per document) ─────────────────────────────────────────────┐
            1 Parse → 2 Charts → 3 Chunk → 4 Embed & store (Qdrant)                    │
                                                                                       ▼
 QUESTION ► ASKING (every question) ◄──────────────────────────────────────── Qdrant: chunks + parents
            5 Hybrid search → 6 Rerank → 7 Parent sections → 8 Gemini drafts
            → 9 Answer rerank → 10 Tidy → answer with [1] [2] citations
```

### Indexing (click **Load & index**; once per document)

**Stage 1 - Parse** (`ingestion.py`, `textclean.py`)
- PyMuPDF4LLM reads each page and returns Markdown in the correct reading order, even for multi-column pages. Tables come out as Markdown tables, headings as `#` lines.
- The text is cleaned: leftovers like `<br>`, `<sup>14</sup>` footnote tags and "picture text" markers are removed.

**Stage 2 - Charts and images** (`ingestion.py`, `gemini.py`)
- Only pages that look like charts (many vector drawings), have a large image, or are scanned get sent to Gemini vision. Plain text pages never do.
- Gemini describes the chart (title, axes, key values, trends) and transcribes text inside images. Each description becomes a `figure` chunk linked to the section it sits in.
- Free-tier protection: at most `max_vision_pages_per_run` (10) pages per run, most chart-like first; results are cached in `data/processed/vision_cache.json` so a page is never described twice; 4 pages are described in parallel; if the daily quota is hit, indexing continues without the remaining charts.

**Stage 3 - Chunking** (`chunking/`)
- **Section structure:** taken from the PDF outline (table of contents) when it has one, otherwise from the Markdown headings. Page number, heading path and file name are kept on every chunk.
- **Parent sections:** each section becomes a *parent* (about 300-2500 tokens). Tiny sections are merged into the next one; huge ones are split at paragraph and sentence boundaries. Documents with no headings still get parents (blocks grouped up to about 2500 tokens), so every chunk has a parent.
- **Child chunks:** each parent is cut into *children* of about 250 tokens (200 for number-heavy text) with about 50 tokens (one sentence) of overlap. These small chunks are what the search matches.
- **Golden rule:** a chunk never ends in the middle of a sentence. Chunks are packed from whole sentences (abbreviations like `Dr.` `e.g.` `Fig. 3` and decimals like `3.5%` are handled), and an over-long sentence stays whole in its own chunk. A sentence that continues across a page break is joined.
- **Tables stay tables:** a table is never cut mid-row. A big table is split by rows with its header repeated. The line that labels a table ("Table 3", "The 24-inch configuration:") is kept with it, and tables in the same section share one parent, so the model can compare them.
- **Context header:** every chunk is embedded as `<document> > <section> (p. N)` + text, so the search knows where it comes from.

**Stage 4 - Embed and store** (`store.py`, `gemini.py`)
- Gemini `gemini-embedding-001` turns each chunk into a 768-number vector (meaning search). Qdrant builds a BM25 vector from the same text (keyword search).
- Two collections in Qdrant: `docs` (children, searched) and `docs_parents` (section text, looked up by id).
- **Re-indexing replaces the old data:** the new version is read first; then everything stored for that document (chunks and parents) is deleted and the new chunks are inserted. Other documents are untouched. If storing fails, partial data is removed, so a half-indexed document never shows as "Indexed".

### Asking (every question)

**Stage 5 - Hybrid search** (`store.py`)
- One Gemini call embeds your question. Qdrant runs a meaning search and a keyword search (30 candidates each) and merges them with Reciprocal Rank Fusion (RRF). Keywords catch exact numbers and names; embeddings catch paraphrases. The Scope box limits the search to one document.

**Stage 6 - Rerank + refuse check** (`reranker.py`)
- A small local cross-encoder (`Xenova/ms-marco-MiniLM-L-12-v2`, about 120 MB, runs on ONNX) reads the question and each chunk together and re-orders them. The best 8 are kept.
- If even the best chunk scores below `min_rerank_score` (0.30), the app answers "I don't know based on the provided records." immediately, without asking Gemini.

**Stage 7 - Parent sections** (`pipeline.py`)
- Each found chunk is swapped for its parent section (the larger text around it). Duplicates are removed and the total is limited to `max_context_tokens` (6000).

**Stage 8 - Gemini drafts** (`pipeline.py`, `gemini.py`)
- The prompt (`ANSWER_PROMPT`) contains the numbered context entries and the question, plus rules: answer first, say each fact once, use symbols (`91.1%`, `$5`, `100V`), cite with `[n]`, say what is missing, or reply "I don't know...".
- Gemini writes 3 drafts in a single request (one unit of quota).

**Stage 9 - Answer rerank** (`answer_rerank.py`)
- Each draft is split into lines and scored against the context it cites, using the same local model:

| Check | Weight | Meaning |
|---|---|---|
| support | 0.35 | average and weakest line's match with the evidence |
| number grounding | 0.15 | share of numbers that appear in the context |
| term grounding | 0.20 | share of the answer's words that appear in the context |
| relevance | 0.15 | how well the answer fits the question |
| citation check | 0.15 | cited entries really support the cited lines |

- The best draft wins and unsupported lines are removed. If every draft is "I don't know", the best one is weak (`min_answer_support` 0.25), or it has mostly invented numbers and wording, the app refuses.
- This is a relevance model, not a fact checker: it catches answers that drift from the evidence, not every wrong answer.

**Stage 10 - Tidy** (`textclean.py`)
- Repeated sentences are removed (even when glued together), "91.1 percent" becomes `91.1%`, and sentences stuck together after a citation are separated. Tables and headings are not touched.

**Stage 11 - Display** (`ui/index.html`)
- Bold text, bullets, tables and `[1]` citation badges are rendered. Hover a badge to see its file, section and pages. **View Chunks → Active in prompt** shows the chunks used for the last answer with their scores.

---

## Quick start

```
pip install -r requirements.txt
copy .env.example .env          # then fill in the keys below
cd "<path>\Advance RAG Pipeline"
uvicorn rag.api:app --app-dir src --port 8000
```

Open http://localhost:8000. Run the command **from the project folder**, otherwise `--app-dir src` cannot find the code.

`.env`:

| Key | Needed | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | yes | from Google AI Studio |
| `QDRANT_URL`, `QDRANT_API_KEY` | no | Qdrant Cloud cluster. If empty, Qdrant runs locally in `data/processed/qdrant`. |

### Using the web UI
1. **Upload PDFs** → the file appears with an amber **Uploaded** tag and is selected.
2. **Load & index** → watch the progress bar (reading pages, describing charts, loading chunks). The tag turns green **Indexed**. One document is processed at a time.
3. Ask questions in the **Ask Workbench**. Pick the document in **Scope**.
4. **View Chunks** opens the chunk viewer: search, filter by type (text / table / figure), export as JSON.
5. **Re-index** replaces a document's old index with a new one.

The page is drawn at 75% so it fits one screen without scrolling (`--ui-zoom` in the page's CSS).

### Command line
```
python scripts/cli.py index                                   # index every PDF in data/raw
python scripts/cli.py ask "your question" [--source file.pdf]
python scripts/cli.py eval eval/questions.jsonl [--no-answer] [--out report.json]
```

### API
| Endpoint | Purpose |
|---|---|
| `GET /` | the web UI |
| `GET /files` | uploaded PDFs with state (uploaded / processing / indexed), size, chunk count |
| `POST /upload` | upload PDFs (replacing a file clears its old index) |
| `POST /index/stream?source=file.pdf[&force=true]` | process one document, streams progress events |
| `POST /ask` `{question, source?}` | answer + sources + reranked chunks |
| `GET /chunks?source=&type=&q=&offset=&limit=` | stored chunks of a document |
| `GET /pdf/{name}` | open an uploaded PDF |
| `GET /status`, `GET /health` | state checks |

---

## Configuration (`configs/default.yaml`)

| Setting | Default | Effect |
|---|---|---|
| `parsing.max_vision_pages_per_run` | 10 | chart pages described per run (0 = no limit) |
| `parsing.describe_visuals` | true | set `false` to skip Gemini vision completely |
| `chunking.target_tokens` / `parent_child.child_target_tokens` | 250 | prose chunk size |
| `parent_child.parent_max_tokens` / `parent_min_tokens` | 2500 / 300 | parent section size |
| `retrieval.top_k_candidates` | 30 | candidates per search branch |
| `retrieval.top_k_final` | 8 | chunks kept after reranking |
| `retrieval.max_context_tokens` | 6000 | context sent to Gemini |
| `retrieval.rerank_model` | MiniLM-L-12 | local reranker (`BAAI/bge-reranker-base` is a bigger option) |
| `retrieval.min_rerank_score` | 0.30 | below this the app refuses |
| `generation.model` | gemini-2.5-flash | answer model |
| `generation.vision_model` | gemini-2.5-flash | chart description model |
| `generation.answer_rerank` / `answer_candidates` | true / 3 | best-of-N answers (`false` = one plain answer) |
| `generation.min_answer_support` | 0.25 | best draft below this → refuse |

Restart the server after editing the file.

## Gemini calls and cost (free tier)

| When | Gemini calls |
|---|---|
| Upload | 0 |
| Index | up to 10 chart descriptions (cached after the first time) + about 1 embedding request per 32 chunks |
| Each question | 1 embedding + 1 answer request (3 drafts in one request) |
| Question refused at the search stage | 1 (the embedding only) |

Free-tier limits seen on this project: 20 `generate_content` requests per day per model, 100 embedding requests per minute. On the paid tier `gemini-2.5-flash` costs about $0.30 / $2.50 per 1M input / output tokens (see Google's pricing page); a question is roughly a cent or less (estimate). To spread quota, use a different model for answers and for vision.

## Project layout

```
configs/default.yaml        all settings
src/rag/
  config.py                 loads the settings
  ingestion.py              PDF -> Markdown -> chunks, chart descriptions, vision cache
  chunking/                 sentences.py (sentence splitter), dynamic.py (chunker),
                            parent_child.py (sections, parents, children)
  textclean.py              cleanup of PDF text and of answers (tidy_answer)
  gemini.py                 Gemini client: embeddings, text, vision, N drafts, quota handling
  store.py                  Qdrant: collections, hybrid search, parents, per-document delete
  reranker.py               local cross-encoder
  answer_rerank.py          scoring and choosing the best draft
  pipeline.py               ties everything together (read its top comment first)
  api.py, ui/index.html     FastAPI service and the web page
  evaluate.py               retrieval and answer evaluation
  logging_setup.py          writes logs/rag.log
scripts/cli.py              command-line entry point
tests/                      53 automated tests
eval/questions.example.jsonl  template for your own evaluation questions
docs/PIPELINE_PLAN.md       the original design plan
```

## Logs, tests and evaluation

- **Logs:** `logs/rag.log` (rotating text file) records parsing, chunk counts, quota hits, Gemini retries, every question with its best rerank score, and the scores of each draft answer.
- **Tests:** `python -m pytest -q tests` (they use fake models and an in-memory Qdrant; no keys or downloads needed).
- **Evaluation:** copy `eval/questions.example.jsonl`, replace the `TODO` entries with real questions (expected pages and keywords), then run `cli.py eval`. It reports hit-rate@k, MRR, correct refusals and false refusals. Use it to tune `min_rerank_score` and `min_answer_support`; the current thresholds are starting values.

## Known limits

- The layout model that reads PDFs needs memory. On a machine with under about 1 GB free it can fail with "bad allocation"; close other apps and retry. Reading a 200-page PDF takes about 2 minutes.
- The reranker is small and English-focused; non-English documents may rank worse.
- Answer scoring is a relevance proxy and can pass a plausible but wrong answer if every draft is wrong.
- Chart descriptions are limited by the free-tier daily quota; unprocessed chart pages have only their surrounding text.
- A chunk stops at a page boundary only when the sentence ends there; the order of chunks within one page is not stored.
- Scanned (image-only) PDFs rely on Gemini vision, which is quota-limited.

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'rag'` | run `uvicorn` from the project folder |
| `GEMINI_API_KEY is not set` | keys go in `.env`, not `.env.example` |
| 429 `RESOURCE_EXHAUSTED` (per day) | the free quota is used up; wait, or change `generation.model` |
| 429 on embeddings (per minute) | free tier allows 100 per minute; indexing a big document is slow |
| "Nothing is indexed yet" | click **Load & index** and wait for the green **Indexed** tag |
| A floating panel/brain icon on the right | a browser extension, not this app; disable it for `localhost` |
