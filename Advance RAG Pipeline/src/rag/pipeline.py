"""The RAG pipeline: this file connects all the other modules.

INDEXING (index_stream) - runs once per document, when you click "Load & index":
  1. ingest_pdf     read the PDF, split it into chunks (+ parent sections), describe charts with Gemini
  2. embed + store  Gemini turns each chunk into a vector; chunks go to Qdrant (parents to a 2nd collection)

ASKING (ask) - runs for every question:
  1. hybrid search   Qdrant finds ~30 candidate chunks (meaning search + keyword search)
  2. rerank          a small local model re-orders them; keep the best 8
  3. refuse check    if even the best chunk is weak -> "I don't know" (no Gemini call for the answer)
  4. context merge   keep precise child chunks plus parent sections within the context budget
  5. prompt          put the question + numbered context into ANSWER_PROMPT
  6. generate        Gemini writes 3 draft answers in ONE request
  7. answer rerank   score the drafts against the context, keep the best-supported one
  8. tidy            remove repeated sentences, write "91 percent" as "91%"
"""
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .answer_rerank import is_refusal, pick_best, prune_citations
from .config import Config, load_config
from .gemini import Gemini
from .ingestion import VisionBudget, ingest_pdf
from .reranker import CrossEncoderReranker
from .textclean import clean_text, tidy_answer
from .store import (
    count_source, delete_source, ensure_collections, fetch_parents, get_client, hybrid_search,
    upsert_chunks_iter, upsert_parents,
)

# The instructions sent to Gemini together with the retrieved context and the question.
# Edit the rules here to change the style of the answers.
ANSWER_PROMPT = """You are a precise document assistant. Answer the question using ONLY the numbered context below.

How to write the answer:
- Start with the direct answer in one or two sentences. No introduction and no filler.
- Say every fact exactly ONCE. Never repeat a sentence, a bullet or a figure, and never restate the first sentence as a bullet.
- Add bullets only for extra supporting details that the first sentence does not already contain. If the first sentence says everything, write no bullets at all.
- For a calculation (difference, total, ratio), give the result and its inputs in one sentence, for example: 91.5% - 91.1% = 0.4 percentage points. Do not list the inputs again on separate lines.
- Write numbers and units the way the documents do. Use symbols and standard abbreviations, not words: 91.1% (not "91.1 percent"), $5 (not "5 dollars"), 25 °C, 100V, 65W, 24-inch, 4.5K. Keep figures exactly as in the context.
- Put only the key result in **bold**, not every number.
- Use a small Markdown table only when comparing several items.
- Cite every claim with its context number in square brackets, written [n] and placed right after the claim, before the full stop: "GPO produced 15,901,600 passports [8]." Never write a bare number like "passports 8." and never write a sources list.
- Cite the single best source for each claim. Add a second citation only when it gives information the first does not. Use only numbers that appear in the context.
- Connect related facts across the context. For example, if a material or figure is listed for a part of the product, name that part.
- If the context only answers part of the question, answer that part and say plainly what is missing.
- If the context does not contain the answer at all, reply exactly: I don't know based on the provided records.
- Context entries marked "figure" are AI-written descriptions of charts or images.

Context:
{context}

Question: {question}

Answer:"""

log = logging.getLogger("rag.pipeline")

REFUSAL = "I don't know based on the provided records."


class NotIndexed(RuntimeError):
    """Asked a question before any document was indexed."""


@dataclass
class Answer:
    """What ask() returns."""
    text: str  # the answer shown to the user
    sources: list[dict]  # the context entries behind [1], [2]... (file, section, pages)
    refused: bool = False  # True when the answer is "I don't know..."
    ranked: list[dict] = field(default_factory=list)  # reranked chunks: payload + score


class RAGPipeline:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or load_config()  # settings from configs/default.yaml
        self.gemini = Gemini(self.cfg)  # embeddings, chart descriptions, answers
        self.client = get_client(self.cfg)  # Qdrant (cloud if QDRANT_URL is set, else local folder)
        self.reranker = CrossEncoderReranker(self.cfg.retrieval.rerank_model)  # small local model, loads on first use

    def index(self) -> int:
        # Index every PDF in data/raw (CLI). Each document replaces its own old chunks.
        last: dict = {}
        for last in self.index_stream():
            pass
        return last["chunks"]

    def is_indexed(self) -> bool:
        return self.client.collection_exists(self.cfg.qdrant.collection)

    def doc_status(self, name: str) -> int:
        # Number of stored chunks for a document (0 = not indexed)
        return count_source(self.client, self.cfg, name)

    def index_stream(self, source: str | None = None):
        # Yields progress events (used by the UI). With `source`, only that one PDF is processed.
        raw = self.cfg.paths.raw_dir
        files = [raw / Path(source).name] if source else sorted(raw.rglob("*.pdf"))
        files = [f for f in files if f.exists()]
        if not files:
            raise SystemExit(f"PDF not found in {raw}")
        t0 = time.time()
        log.info("Indexing started: %s", ", ".join(f.name for f in files))
        yield {"event": "start", "files": [f.name for f in files]}
        ensure_collections(self.client, self.cfg)  # create the Qdrant collections if they do not exist yet
        budget = VisionBudget(self.cfg.parsing.max_vision_pages_per_run)  # shared by all files
        total_chunks = total_parents = 0
        for i, path in enumerate(files, 1):
            yield {"event": "parsing", "file": path.name, "index": i, "total": len(files)}
            # Run the slow parse + vision step in a worker thread so progress can stream live
            events: queue.Queue = queue.Queue()
            box: dict = {}

            def work(path=path, events=events, box=box) -> None:
                try:
                    box["one"] = ingest_pdf(path, self.cfg, self.gemini, events.put, budget)
                except Exception as e:  # re-raised in the generator below
                    box["error"] = e
                events.put(None)

            threading.Thread(target=work, daemon=True).start()
            while (ev := events.get()) is not None:
                yield ev
            if "error" in box:
                log.error("Failed on %s: %s", path.name, box["error"])
                raise box["error"]
            one = box["one"]  # the parsed document: chunks + parent sections
            yield {"event": "parsed", "file": path.name, "chunks": len(one.chunks),
                   "parents": len(one.parents)}

            # Step 2: replace this document's old data, then store the new chunks
            delete_source(self.client, self.cfg, path.name)
            try:
                upsert_parents(self.client, self.cfg, one.parents)  # parent sections (no embedding needed)
                done = 0
                # embeds each batch of chunks with Gemini and saves it in Qdrant
                for part in upsert_chunks_iter(self.client, self.cfg, self.gemini, one.chunks):
                    done += len(part)
                    preview = [
                        {"type": c.metadata["type"], "section": c.metadata.get("section", ""),
                         "page": c.metadata["page"], "text": c.text[:140]}
                        for c in part[:3]
                    ]
                    if done % 320 < len(part) or done == len(one.chunks):
                        log.info("%s: embedded and stored %d / %d chunks", path.name, done, len(one.chunks))
                    yield {"event": "embedding", "file": path.name, "done": done,
                           "total": len(one.chunks), "preview": preview}
            except BaseException:
                # never leave a half-indexed document that would look "indexed"
                delete_source(self.client, self.cfg, path.name)
                log.error("Failed while storing %s; removed partial data", path.name)
                raise
            total_chunks += len(one.chunks)
            total_parents += len(one.parents)
        log.info("Indexing finished: %d chunks, %d parent sections in %.0fs",
                 total_chunks, total_parents, time.time() - t0)
        yield {"event": "done", "chunks": total_chunks, "parents": total_parents}

    def _expand_to_parents(self, points: list) -> list[dict]:
        # Small chunks are good for SEARCHING; the LLM answers better from the bigger section around them.
        # So: swap each found chunk for its parent section, drop duplicates (several chunks can share
        # one parent, keeping rank order) and stop when the token budget (max_context_tokens) is full.
        parents = fetch_parents(
            self.client, self.cfg,
            list({p.payload["parent_id"] for p in points if p.payload.get("parent_id")}),
        )
        seen, out, used = set(), [], 0
        for p in points:
            pid = p.payload.get("parent_id")
            if pid in seen:
                continue
            seen.add(pid)
            pay = parents.get(pid) or p.payload  # no parent (dynamic doc): use the chunk itself
            tokens = len(pay["text"]) // 4
            if out and used + tokens > self.cfg.retrieval.max_context_tokens:
                continue
            used += tokens
            out.append(pay)
        return out


    def _merge_contexts(
        self,
        child_contexts: list[dict],
        parent_contexts: list[dict],
        max_tokens: int,
    ) -> list[dict]:
        """Combine ranked child chunks with their parent sections under a token budget.

        Child chunks come first because they often contain exact table values that can
        disappear when only the larger parent section is used. Parent sections follow
        to supply surrounding context. Each context keeps its original metadata so the
        prompt citations and UI source references remain aligned.
        """
        out: list[dict] = []
        seen: set[tuple] = set()
        used_tokens = 0

        def identity(item: dict) -> tuple:
            # Include provenance in the key: identical text on different pages should
            # still retain its own source citation.
            return (
                item.get("source", ""),
                item.get("page"),
                item.get("page_end", item.get("page")),
                item.get("section", ""),
                item.get("type", ""),
                item.get("text", "").strip(),
            )

        def estimated_tokens(item: dict) -> int:
            # Lightweight estimate; avoids another tokenizer dependency.
            return max(1, (len(item.get("text", "")) + 3) // 4)

        # Add the high-ranked, precise child chunks first.
        # If the first item alone exceeds the budget, keep it rather than return no context.
        for item in child_contexts:
            key = identity(item)
            if key in seen:
                continue
            cost = estimated_tokens(item)
            if out and used_tokens + cost > max_tokens:
                continue
            out.append(item)
            seen.add(key)
            used_tokens += cost

        # Add parent sections while there is room, skipping duplicate payloads.
        for item in parent_contexts:
            key = identity(item)
            if key in seen:
                continue
            cost = estimated_tokens(item)
            if used_tokens + cost > max_tokens:
                continue
            out.append(item)
            seen.add(key)
            used_tokens += cost

        return out

    def ask(self, question: str, source: str | None = None, generate: bool = True) -> Answer:
        # `source` limits the search to one PDF; generate=False skips Gemini (used by evaluation)
        if not self.is_indexed():
            raise NotIndexed("Nothing is indexed yet. Upload PDFs and click Load & index first.")

        # Step 1 - hybrid search in Qdrant: meaning (embedding) + keywords (BM25), merged into ~30 chunks
        points = hybrid_search(self.client, self.cfg, self.gemini, question, source=source)

        # Step 2 - rerank: a local cross-encoder reads (question, chunk) together and re-orders them
        texts = [
            f"{p.payload['source']} > {p.payload.get('section', '')} (p.{p.payload['page']})\n"
            f"{p.payload['text']}"
            for p in points
        ]
        ranked = self.reranker.rerank(question, points, texts, self.cfg.retrieval.top_k_final)  # best 8
        
        info = [{**r.item.payload, "id": str(r.item.id), "score": r.score} for r in ranked]
        best = ranked[0].score if ranked else 0.0
        log.info("ask %r source=%s: %d candidates, best rerank score %.2f", question, source,
                 len(points), best)

        # Step 3 - refuse early: nothing relevant found, so do not ask Gemini to guess
        if not ranked or best < self.cfg.retrieval.min_rerank_score:
            log.info("refused (best score %.2f < %.2f)", best, self.cfg.retrieval.min_rerank_score)
            return Answer(REFUSAL, [], refused=True, ranked=info)

        # Step 4 - swap the small chunks for their parent sections (more context for the answer)
        # Keep the exact child chunks selected by the reranker.
        # Parent expansion can lose specific table values needed for precise answers.
        child_contexts = [r.item.payload for r in ranked]
        
        # Expand the retrieved child chunks into their parent sections.
        parent_contexts = self._expand_to_parents(
            [r.item for r in ranked]
        )
        
        # Keep child chunks first so exact figures and table entries are preserved.
        contexts = self._merge_contexts(
            child_contexts,
            parent_contexts,
            max_tokens=self.cfg.retrieval.max_context_tokens,
        )

        def pages(c: dict) -> str:
            lo, hi = c["page"], c.get("page_end", c["page"])
            return f"p.{lo}" if lo == hi else f"p.{lo}-{hi}"

        # [1], [2]... in the answer point to these entries (the UI shows them on hover)
        sources = [
            {"n": i + 1, "source": c["source"], "page": c["page"], "pages": pages(c),
             "section": c.get("section", ""), "type": c["type"], "snippet": c["text"][:200],
             "text": clean_text(c["text"])}
            for i, c in enumerate(contexts)
        ]
        if not generate:  # retrieval-only (used by evaluation to save Gemini quota)
            return Answer("", sources, ranked=info)

        # Step 5 - build the prompt: numbered context entries + the question
        context = "\n\n".join(
            f"[{i + 1}] ({c['source']}, {c.get('section') or 'no section'}, {pages(c)}, {c['type']})\n"
            f"{clean_text(c['text'])}"
            for i, c in enumerate(contexts)
        )
        prompt = ANSWER_PROMPT.format(context=context, question=question)
        gen = self.cfg.generation

        # Simple mode (answer_rerank: false): one answer from Gemini, then tidy it
        if not gen.answer_rerank or gen.answer_candidates <= 1:
            raw_text = self.gemini.generate(prompt)
            text = prune_citations(  # one clearly supporting source per claim
                tidy_answer(raw_text), [clean_text(c["text"]) for c in contexts], self.reranker.score)
            return Answer(
                text,
                sources,
                refused=is_refusal(text),
                ranked=info,
            )

        # Step 6 - best of N: Gemini writes several drafts in ONE request; tidy_answer (step 8) removes
        # repeated sentences and writes "91 percent" as "91%" in each draft before scoring
        drafts = [tidy_answer(d) for d in self.gemini.generate_candidates(
            prompt, gen.answer_candidates, gen.candidate_temperature)]

        # Step 7 - answer rerank: score each draft against the context (support, numbers, wording,
        # citations) and keep the best one; drop unsupported lines; refuse if even the best is weak
        text, refused, scored = pick_best(
            drafts, question, [clean_text(c["text"]) for c in contexts],
            self.reranker.score, gen.min_answer_support,
        )
        if not refused:  # Step 7b - keep ONE citation per claim when a single source clearly backs it
            text = prune_citations(text, [clean_text(c["text"]) for c in contexts], self.reranker.score)
        log.info("answer rerank: %d draft(s), %s", len(drafts), "; ".join(
            f"final={c.final:.2f} support={c.support:.2f} numbers={c.numbers:.2f} "
            f"terms={c.terms:.2f} relevance={c.relevance:.2f} cites={c.cites:.2f}" + (" REFUSED" if c.refused else "")
            for c in scored))
        return Answer(text, [] if refused else sources, refused=refused, ranked=[] if refused else info)
