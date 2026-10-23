"""Layout-aware PDF parsing.

Per page we produce:
  * text/table chunks  - PyMuPDF4LLM returns Markdown in correct reading order across
                         multiple columns, with tables as Markdown tables.
  * figure chunks      - pages with embedded images or chart-like vector drawings are
                         rendered and described by Gemini vision (chart values, trends,
                         image content, text inside images).
Text is chunked by rag.chunking (whole sentences only; tables and code kept atomic):
  * long structured documents (outline/headings, 30+ pages) -> parent-child chunks
  * everything else -> dynamic chunks
Every chunk is embedded with a context header: "<doc title> > <section> (p. N)".
"""
import hashlib
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf
import pymupdf4llm

from .chunking import ChunkParams, ParentParams, build_parent_child, chunk_markdown, looks_structured
from .config import Config
from .gemini import Gemini, QuotaExhausted
from .textclean import clean_text

log = logging.getLogger("rag.ingestion")


class VisionBudget:
    """How many Gemini vision calls one index run may make (free tier: ~20 requests/day/model)."""

    def __init__(self, limit: int):
        self.remaining = limit if limit > 0 else None  # None = unlimited

    def take(self, n: int) -> int:
        if self.remaining is None:
            return n
        got = min(n, self.remaining)
        self.remaining -= got
        return got


VISION_WORKERS = 4  # parallel Gemini vision calls (retries handle free-tier rate limits)

VISUAL_PROMPT = (
    "You are indexing a document page for retrieval. Describe every chart, graph, diagram, "
    "photo or image on this page. For charts: title, axes, series, and the key data values and "
    "trends. Transcribe any text inside images. Ignore plain body text and ordinary tables. "
    "If the page has no meaningful visual, reply exactly: NONE"
)


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def embed_text(self) -> str:
        # Context header + text: used for embeddings and BM25 (payload keeps the raw text)
        m = self.metadata
        path = " > ".join(x for x in (Path(m.get("source", "")).stem, m.get("section", "")) if x)
        return f"{path} (p. {m.get('page', '?')})\n{self.text}"


def _make_id(source: str, page: int, kind: str, idx: int, text: str) -> str:
    # Stable ID so re-indexing the same content gives the same point
    h = hashlib.sha1(f"{source}|{page}|{kind}|{idx}|{text}".encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"  # UUID-shaped for Qdrant


def _page_has_visual(page: pymupdf.Page, cfg: Config) -> bool:
    # True if the page has a big image or many vector drawings (likely a chart)
    area = page.rect.width * page.rect.height
    for info in page.get_image_info():
        r = pymupdf.Rect(info["bbox"]) & page.rect
        if r.width * r.height / area >= cfg.parsing.min_image_area_ratio:
            return True
    return len(page.get_drawings()) >= cfg.parsing.min_drawings_for_chart


def _is_scanned(page: pymupdf.Page, md: str) -> bool:
    # Almost no text but has images = scanned page
    return len(md.strip()) < 40 and bool(page.get_image_info())


@dataclass
class Ingested:
    chunks: list[Chunk] = field(default_factory=list)   # searchable (children / dynamic / figures)
    parents: list[Chunk] = field(default_factory=list)  # section text sent to the LLM


def _dynamic_chunks(src: str, pages_md: list[str], params: ChunkParams) -> list[Chunk]:
    out: list[Chunk] = []
    for i, md in enumerate(pages_md):
        page_no = i + 1
        for idx, c in enumerate(chunk_markdown(md, params, page_no)):
            meta = {"source": src, "page": page_no, "page_end": page_no,
                    "type": c.kind, "section": c.section}
            out.append(Chunk(_make_id(src, page_no, c.kind, idx, c.text), c.text, meta))
    return out


def _parent_child_chunks(
    src: str, pages_md: list[str], toc: list, pp: ParentParams
) -> tuple[list[Chunk], list[Chunk]]:
    parents, children = build_parent_child(list(enumerate(pages_md, 1)), toc, pp)
    parent_chunks = [
        Chunk(
            _make_id(src, p.page_start, "parent", i, p.text),
            p.text,
            {"source": src, "page": p.page_start, "page_end": p.page_end,
             "type": "parent", "section": p.section},
        )
        for i, p in enumerate(parents)
    ]
    child_chunks = [
        Chunk(
            _make_id(src, c.page_start, c.kind, n, c.text),
            c.text,
            {"source": src, "page": c.page_start, "page_end": c.page_end, "type": c.kind,
             "section": c.section, "parent_id": parent_chunks[pi].id},
        )
        for n, (pi, c) in enumerate(children)
    ]
    return parent_chunks, child_chunks


def _emit(progress, **event) -> None:
    if progress:
        progress({"event": "progress", **event})


def _describe_pages(
    doc, path: Path, cfg: Config, gemini, pages: list[int], progress, budget: VisionBudget
) -> dict[int, str]:
    """Describe chart/image pages with Gemini vision.

    Results are cached on disk (a re-index never pays for the same page twice) and
    several pages are described in parallel. A failed page is skipped, not fatal.
    """
    if not pages:
        return {}
    cache_file = cfg.paths.cache_dir / "vision_cache.json"
    cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}
    digest = hashlib.sha1(path.read_bytes()).hexdigest()[:16]

    def key(n: int) -> str:
        return f"{digest}:{n}:{cfg.parsing.render_dpi}:{cfg.generation.vision_model}"

    out = {n: cache[key(n)] for n in pages if key(n) in cache}
    todo = [n for n in pages if n not in out]  # `pages` is ordered most chart-like first
    allowed = budget.take(len(todo))
    skipped, todo = len(todo) - allowed, todo[:allowed]
    done, failed, total = len(out), 0, len(out) + len(todo)
    log.info("%s: %d visual pages (%d cached, %d to describe now, %d skipped by the per-run limit)",
             path.name, len(pages), done, len(todo), skipped)
    if skipped:
        _emit(progress, file=path.name, stage="vision", done=done, total=total, skipped=skipped)
    _emit(progress, file=path.name, stage="vision", done=done, total=total)

    def save() -> None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cache), encoding="utf-8")

    # Render in this thread (PyMuPDF documents are not thread-safe), call Gemini in parallel
    images = {n: doc[n - 1].get_pixmap(dpi=cfg.parsing.render_dpi).tobytes("png") for n in todo}
    stop = threading.Event()  # set when the daily quota is hit: no worker makes another call

    def describe(img: bytes) -> str:
        if stop.is_set():
            raise QuotaExhausted("daily quota already reached")
        try:
            return gemini.describe_image(img, VISUAL_PROMPT)
        except QuotaExhausted:
            stop.set()
            raise

    with ThreadPoolExecutor(max_workers=VISION_WORKERS) as pool:
        futures = {pool.submit(describe, img): n for n, img in images.items()}
        for fut in as_completed(futures):
            n = futures[fut]
            try:
                out[n] = cache[key(n)] = fut.result().strip()
            except QuotaExhausted as e:  # daily limit hit: stop calling Gemini, keep what we have
                log.error("%s", e)
                for f in futures:
                    f.cancel()
                _emit(progress, file=path.name, stage="vision", done=done, total=total,
                      quota_hit=True, message=str(e))
                break
            except Exception as e:
                failed += 1
                log.warning("%s p.%d: vision failed, skipped (%s)", path.name, n, e)
            done += 1
            _emit(progress, file=path.name, stage="vision", done=done, total=total, failed=failed)
            if done % 10 == 0:
                save()
    if todo:
        save()
    if failed:
        log.warning("%s: %d visual pages failed", path.name, failed)
    return out


def ingest_pdf(
    path: Path, cfg: Config, gemini: Gemini | None, progress=None, budget: VisionBudget | None = None
) -> Ingested:
    doc = pymupdf.open(path)
    src = path.name
    result = Ingested()
    log.info("Parsing %s (%d pages)", src, len(doc))

    pages_md: list[str] = []
    for i in range(len(doc)):  # page by page so progress can be reported
        pg = pymupdf4llm.to_markdown(doc, pages=[i], page_chunks=True, show_progress=False)
        pages_md.append(clean_text(pg[0]["text"]))  # drop <br>, <sup> and picture-text markers
        if (i + 1) % 5 == 0 or i + 1 == len(doc):
            _emit(progress, file=src, stage="parsing", done=i + 1, total=len(doc))

    # 1) Text, table and code chunks. Router: structured long document -> parent-child
    pp = ParentParams(**cfg.parent_child.model_dump())
    toc = [(lvl, title, pg) for lvl, title, pg in doc.get_toc() if pg >= 1]
    if looks_structured(pages_md, toc, pp):
        log.info("%s: parent-child chunking (%d outline entries)", src, len(toc))
        result.parents, result.chunks = _parent_child_chunks(src, pages_md, toc, pp)
    else:
        log.info("%s: dynamic chunking", src)
        result.chunks = _dynamic_chunks(src, pages_md, ChunkParams(**cfg.chunking.model_dump()))
    log.info("%s: %d text/table chunks, %d parent sections",
             src, len(result.chunks), len(result.parents))

    # 2) Figure chunks: Gemini describes charts/images on chart-like or large-image pages
    visual_pages = []
    if gemini and cfg.parsing.describe_visuals:
        visual_pages = [
            i + 1 for i, md in enumerate(pages_md)
            if _is_scanned(doc[i], md) or _page_has_visual(doc[i], cfg)
        ]
        # most chart-like pages first, so a limited vision budget goes to the best candidates
        visual_pages.sort(key=lambda n: -len(doc[n - 1].get_drawings()))
    descriptions = _describe_pages(
        doc, path, cfg, gemini, visual_pages, progress,
        budget or VisionBudget(cfg.parsing.max_vision_pages_per_run),
    )
    for page_no, desc in sorted(descriptions.items()):
        if desc and desc.upper() != "NONE":
            meta = {"source": src, "page": page_no, "page_end": page_no, "type": "figure"}
            parent = next(
                (p for p in result.parents
                 if p.metadata["page"] <= page_no <= p.metadata["page_end"]), None
            )
            if parent:  # link the figure to the section it sits in
                meta |= {"parent_id": parent.id, "section": parent.metadata["section"]}
            result.chunks.append(Chunk(_make_id(src, page_no, "figure", 0, desc), desc, meta))
    log.info("%s: done, %d chunks total", src, len(result.chunks))
    return result


