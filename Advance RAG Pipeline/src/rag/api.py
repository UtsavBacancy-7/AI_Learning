"""FastAPI service + simple web UI.

Run:  uvicorn rag.api:app --app-dir src --port 8000   then open http://localhost:8000
"""
import json
import logging
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .config import load_config
from .gemini import QuotaExhausted
from .logging_setup import setup_logging
from .pipeline import NotIndexed, RAGPipeline
from .store import delete_source, scroll_all

from fastapi import Request
from fastapi.responses import JSONResponse
from qdrant_client.http.exceptions import ResponseHandlingException

setup_logging()  # writes logs/rag.log
log = logging.getLogger("rag.api")
UI_FILE = Path(__file__).parent / "ui" / "index.html"

app = FastAPI(title="Advance RAG Pipeline")


@app.exception_handler(ResponseHandlingException)
async def qdrant_unreachable(request: Request, exc: ResponseHandlingException):
    log.error("Qdrant did not answer: %s", exc)
    return JSONResponse(status_code=503, content={
        "detail": "Qdrant did not answer in time. Check your internet connection and QDRANT_URL, then try again."})
_pipeline: RAGPipeline | None = None
_index_lock = threading.Lock()  # only one document is processed at a time
_chunk_cache: dict[str | None, tuple[float, list[dict]]] = {}  # source -> (time loaded, chunks)
CHUNK_CACHE_SECONDS = 60  # the chunk viewer asks again for every tab and search; do not re-read Qdrant each time
_processing: str | None = None  # name of the document being processed


def get_pipeline() -> RAGPipeline:
    # Create once on first use (loading the reranker is slow)
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


def sse(event: dict) -> str:
    # One Server-Sent-Events message
    return f"data: {json.dumps(event)}\n\n"


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    source: str | None = None  # restrict to one PDF file name


class SourceOut(BaseModel):
    n: int
    source: str
    pages: str
    section: str
    type: str
    snippet: str
    text: str = ""  # the full cited entry, shown when a citation is clicked
    page: int = 1  # first page of the entry, used to open the PDF at that page


class RefOut(BaseModel):
    id: str
    source: str
    page: int
    page_end: int
    section: str
    type: str
    score: float  # cross-encoder relevance, 0-1
    tokens: int
    text: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    sources: list[SourceOut]
    references: list[RefOut] = []


@app.get("/")
def ui() -> FileResponse:
    return FileResponse(UI_FILE)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/status")
def status() -> dict:
    # Lets the UI tell the user whether anything has been indexed yet
    return {"indexed": get_pipeline().is_indexed()}


def page_chunks(payloads: list[dict], kind: str | None, offset: int, limit: int, q: str | None = None) -> dict:
    """Counts per type (unfiltered), plus one page of chunks in document order."""
    counts: dict[str, int] = {}
    for p in payloads:
        counts[p["type"]] = counts.get(p["type"], 0) + 1
    rows = [p for p in payloads if not kind or p["type"] == kind]
    if q:
        rows = [p for p in rows if q.lower() in p["text"].lower()]
    rows.sort(key=lambda p: (p["source"], p["page"], p["type"]))
    items = [
        {"id": p.get("_id", ""), "source": p["source"], "type": p["type"], "page": p["page"],
         "page_end": p.get("page_end", p["page"]), "section": p.get("section", ""),
         "text": p["text"], "tokens": max(1, len(p["text"]) // 4), "has_parent": bool(p.get("parent_id"))}
        for p in rows[offset : offset + limit]
    ]
    return {"total": len(rows), "counts": counts, "items": items}


@app.get("/chunks")
def chunks(source: str | None = None, type: str | None = None, q: str | None = None,
           offset: int = 0, limit: int = 50) -> dict:
    # Loaded (indexed) chunks of the selected document, for the UI chunk viewer
    pipe = get_pipeline()
    if not pipe.is_indexed():
        return {"total": 0, "counts": {}, "items": []}
    cached = _chunk_cache.get(source)
    if cached and time.time() - cached[0] < CHUNK_CACHE_SECONDS:
        payloads = cached[1]
    else:
        payloads = [{**p.payload, "_id": str(p.id)} for p in scroll_all(pipe.client, pipe.cfg, source)]
        _chunk_cache[source] = (time.time(), payloads)
    return page_chunks(payloads, type, max(offset, 0), min(max(limit, 1), 200), q)


@app.get("/pdf/{name}")
def pdf(name: str) -> FileResponse:
    # Open an uploaded PDF in the browser
    path = load_config().paths.raw_dir / Path(name).name
    if not path.exists() or path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(path, media_type="application/pdf")


@app.get("/files")
def files() -> list[dict]:
    # Every uploaded PDF with its state: uploaded (not indexed yet), processing, or indexed
    raw = load_config().paths.raw_dir
    names = sorted(p.name for p in raw.rglob("*.pdf")) if raw.exists() else []
    pipe = get_pipeline()
    out = []
    for n in names:
        chunks = pipe.doc_status(n)
        state = "processing" if n == _processing else ("indexed" if chunks else "uploaded")
        st = (raw / n).stat()
        out.append({"name": n, "status": state, "chunks": chunks, "size": st.st_size,
                    "modified": st.st_mtime, "collection": pipe.cfg.qdrant.collection})
    return out


@app.post("/upload")
async def upload(files: list[UploadFile]) -> dict:
    raw = load_config().paths.raw_dir
    raw.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        name = Path(f.filename or "").name  # strip any folder part
        if not name.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Only PDF files are supported: {name}")
        if name == _processing:
            raise HTTPException(status_code=409, detail=f"{name} is being processed right now")
        (raw / name).write_bytes(await f.read())
        pipe = get_pipeline()
        if pipe.doc_status(name):  # new content under an old name: the old index is stale
            delete_source(pipe.client, pipe.cfg, name)
        saved.append(name)
    _chunk_cache.clear()
    return {"saved": saved}


@app.post("/index/stream")
def index_stream(source: str, force: bool = False) -> StreamingResponse:
    # Processes ONE document and streams progress events (parse, describe charts, embed, store)
    global _processing
    name = Path(source).name
    if not (load_config().paths.raw_dir / name).exists():
        raise HTTPException(status_code=404, detail=f"{name} has not been uploaded")
    if not _index_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail=f"Another document is being processed ({_processing})")
    try:
        if not force and get_pipeline().doc_status(name):
            raise HTTPException(status_code=409, detail=f"{name} is already indexed")
    except HTTPException:
        _index_lock.release()
        raise
    _processing = name

    def events():
        global _processing
        try:
            for ev in get_pipeline().index_stream(name):
                yield sse(ev)
        except (SystemExit, Exception) as e:  # report any failure to the UI
            log.exception("Indexing failed")
            yield sse({"event": "error", "message": str(e) or type(e).__name__})
        finally:
            _processing = None
            _chunk_cache.clear()  # the stored chunks changed
            _index_lock.release()

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/index")
def index() -> dict:
    # Re-index every PDF in data/raw (no progress stream)
    try:
        return {"chunks": get_pipeline().index()}
    except SystemExit as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        ans = get_pipeline().ask(req.question, source=req.source)
    except NotIndexed as e:
        raise HTTPException(status_code=409, detail=str(e))
    except QuotaExhausted as e:
        log.error("%s", e)
        raise HTTPException(status_code=429, detail=str(e))
    except RuntimeError as e:  # e.g. a missing package: show the real reason, not a bare 500
        log.exception("ask failed")
        raise HTTPException(status_code=500, detail=str(e))
    refs = [] if ans.refused else [
        RefOut(id=c["id"], source=c["source"], page=c["page"], page_end=c.get("page_end", c["page"]),
               section=c.get("section", ""), type=c["type"], score=round(c["score"], 3),
               tokens=max(1, len(c["text"]) // 4), text=c["text"])
        for c in ans.ranked
    ]
    return AskResponse(answer=ans.text, refused=ans.refused, sources=ans.sources, references=refs)
