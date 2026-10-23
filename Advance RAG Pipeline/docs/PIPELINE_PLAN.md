# Advance RAG Pipeline: Design Plan

Stack: **Qdrant** (vector DB) + **Gemini** (embeddings, vision, generation).
Inputs: PDF, PPT, DOCX, TXT, reports, annual reports, research papers, manuals.
Documents can have multiple columns, tables, images and charts.

---

## Golden rule: never chunk in the middle of a sentence

This applies to **every** strategy below.

- Text is first split into sentences. Chunks are built by packing whole sentences.
- A chunk boundary may only fall **after** a sentence end (`.`, `?`, `!`) or at a paragraph/heading break.
- Overlap is also made of whole sentences (the last 1-2 sentences of the previous chunk), never a character count.
- Sentence splitting must not break on abbreviations (`Dr.`, `e.g.`, `Fig. 3`, `U.S.`), decimals (`3.5%`), numbered lists (`1.`), or URLs.
- If a single sentence is longer than the size limit, keep it whole in its own chunk (oversize is acceptable).
- Tables, lists and code blocks are atomic: never cut inside a table row, list item or code block.
- A test enforces this: every chunk must start at a sentence start and end at a sentence end.

---

## Always-do rules (non-negotiable)

1. **Hybrid search**: dense vectors for meaning, BM25 for exact numbers, SKUs and names.
2. **Always rerank** with a cross-encoder (BGE-Reranker, local and free). Aim to put the true answer in the top 3.
3. **Context headers** (title, parent section, page) prepended to every chunk's text before embedding.
4. **Tables as Markdown** (or HTML for merged cells), keeping row/column relationships intact.
5. **Strict refusal**: below the reranker-score threshold, answer "I don't know based on the provided records."

## 1. Route each document by structure

| Signal | Document class | Strategy |
|---|---|---|
| PDF outline/TOC or detected headings, 30+ pages | Structured long (books, project reports, annual reports, GEP) | **Parent-child** |
| Headings present, under 30 pages (policies, contracts, papers) | Structured short | Structure-aware (sections) |
| One slide per page, or `.pptx` | Slides | One chunk per slide |
| No headings, flat prose (txt, transcripts, notes) | Unstructured | **Dynamic** chunking |
| Mostly tables (CSV, XLSX) | Tabular | Row groups, header repeated |
| Almost no extractable text | Scanned | OCR / Gemini vision, then re-route |

A config override always wins. Unclear documents fall back to dynamic chunking.

## 2. Parsing

- **PDF**: PyMuPDF4LLM gives reading order across columns and Markdown tables. The PDF outline (`get_toc`) gives headings; font-size analysis is the fallback.
- **PPTX**: `python-pptx` for title, bullets, notes, tables; Gemini vision for slide images and charts.
- **DOCX / HTML / MD**: use native heading levels. **TXT**: dynamic chunking.
- **Visuals**: Gemini describes charts and images (title, axes, values, trends, text in image) into `figure` chunks that keep page and section path.
- **Cleanup**: remove repeated headers/footers and page numbers, fix hyphenation and broken lines, drop the TOC page, blank pages and boilerplate.

## 3. Parent-child chunking (structured long documents)

1. Build the section tree from the TOC/headings: Chapter > Section > Subsection.
2. **Parent** = a leaf section (or subsection if the leaf is tiny), capped at about 2-3k tokens. Oversize parents are split at subheadings, then paragraphs, then **sentence boundaries**.
3. **Child** = 200-400 token pieces made of whole sentences, with 1-2 sentence overlap.
4. Tables and figures are atomic children, with caption and a one-line summary.
5. Children are embedded and BM25-indexed (the matching layer). Parents live in a docstore keyed by `parent_id`.
6. Child metadata: `doc_id`, `parent_id`, `section_path`, `page_start`, `page_end`, `type`.
7. At query time: retrieve children, dedupe by `parent_id`, send parents to the LLM.

## 4. Dynamic chunking (everything else)

- Target size depends on content: about 400-600 tokens for prose, smaller for dense tables/code, one slide per chunk.
- Split order: paragraph, then **sentence** (never word-level). Pack whole sentences up to the target.
- Optional semantic splitting (cut where adjacent-sentence similarity drops) for long transcripts only, still on sentence boundaries.

## 5. Contextual header on every chunk

Prepend before embedding: `<Doc title> > <Chapter> > <Section> (p. N)`.
Cheap and usually the biggest retrieval gain. Works for all strategies.

## 6. Per-type specifics

- **Annual / financial reports**: statements and notes as table chunks; metadata `fiscal_year`, `section` (MD&A, Risk, Financials).
- **Economic outlook reports (GEP)**: chapters are parents; each chart or box is a figure child with title and values.
- **Research papers**: abstract is its own chunk; sections are parents; captions stay with figures/tables; references dropped or stored separately.
- **Policies / contracts**: clause-level children with clause number in the header; section is the parent.
- **Slides**: title + bullets + notes + vision description per slide.
- **Manuals (iMac PER style)**: keep procedure steps together; warnings stay with their step.

## 7. Indexing

- One Qdrant collection: Gemini dense vectors + BM25 sparse vectors.
- Payload indexes: `doc_id`, `doc_type`, `section_path`, `page`, `type`, `fiscal_year`.
- **Qdrant only, no other database.** Parents live in a second Qdrant collection (`parents`, payload only, with a dummy 1-dim vector), holding parent text and the section tree. Children point to it by `parent_id`.
- Incremental: store a file hash in Qdrant payload, re-index only changed files, delete old points first.

### Free-only constraints
- Qdrant runs locally (free, on-disk or Docker). No paid DB or cloud service.
- Gemini free tier: use a rate limiter and retries, batch embeddings, and cache results.
- Gemini vision runs **only on chart-like pages and pages with large images** (and scanned pages). Plain text and table pages never call it.
- Vision descriptions are stored in Qdrant, so a page is never described twice.

## 8. Retrieval and answering

1. Rewrite follow-up questions into standalone ones (optionally HyDE / query variants).
2. Extract metadata filters (document, year, section) from the query.
3. Hybrid search: dense + sparse, fused with RRF.
4. **Always rerank** the top 30-50 with a free local cross-encoder (`BAAI/bge-reranker-base`, or `-v2-m3` for multilingual). No Gemini calls and no per-question cost.
   - Pipeline: hybrid search returns top 30-50, the cross-encoder scores each (query, chunk) pair, and the **top 5-10** go to the LLM.
   - Model: `BAAI/bge-reranker-v2-m3` (best free option, multilingual); fall back to `bge-reranker-base` if CPU is too slow.
   - Rerank the child chunk text with its context header, so the reranker sees the section and page.
   - **Answer reranking (best of N):**
     1. One Gemini request with `candidate_count=3` (temperature about 0.7) drafts 3 candidate answers from the top chunks. One request keeps the free tier safe.
     2. Each candidate is split into sentences (our sentence splitter) and citations `[n]` are parsed out.
     3. Each sentence is scored by the local cross-encoder (sigmoid-normalised to 0-1) against the chunk it cites plus the top 3 chunks, keeping the best score.
     4. Per-candidate metrics, all 0-1:
        - `support` = 0.7 * mean(sentence scores) + 0.3 * min(sentence scores)
        - `number_grounding` = share of numbers, dates and names in the answer that appear verbatim in the context
        - `relevance` = cross-encoder(question, answer)
        - `citation_validity` = share of sentences whose cited chunk scores above the support threshold
     5. `final = 0.45*support + 0.20*number_grounding + 0.20*relevance + 0.15*citation_validity - length_penalty`. Weights are tuned on the evaluation set.
     6. The highest `final` wins. Sentences below the support threshold are dropped from the winner. If `support` is below the refusal threshold, or too few sentences remain, return the refusal message.
5. **Strict refusal:** if the best reranker score is below a tuned threshold (RRF scores are rank-based, so the threshold is applied to reranker scores), skip generation and reply "I don't know based on the provided records."
6. Expand to parents, dedupe, apply token budget.
7. Grounded answer with `[n]` citations (document, section path, page). The prompt also tells the LLM to refuse when the context does not contain the answer.

## 9. Evaluation

- 30-50 Q&A pairs per document class: factual, table lookup, chart, multi-section, unanswerable.
- Metrics: hit-rate@k, MRR, faithfulness, citation accuracy.
- A/B: child size, parent cap, contextual headers on/off, rerank on/off, parent-child vs dynamic on GEP.

## 10. Build order

1. Document router + normalized parse output.
2. Sentence splitter + sentence-safe chunkers (with the boundary test).
3. Section tree + parent-child chunker (test on `GEP-Jun-2026.pdf`).
4. Dynamic chunker + PPTX/DOCX/TXT loaders.
5. Qdrant schema, docstore, incremental indexing.
6. Parent expansion at query time, then evaluation harness.
7. FastAPI service.

---

## Sample text for testing the sentence rule

Dr. Smith reviewed the Q3 results (see Fig. 2). Revenue grew 3.5% to $1.2 million, e.g. driven by U.S. sales. Was the trend expected? The board said yes! The next section covers risks.

Expected: any chunk boundary falls only after "(see Fig. 2).", "sales.", "expected?", "yes!" or at the paragraph end. It must never fall after "Dr.", "Fig.", "e.g." or inside "3.5%".
