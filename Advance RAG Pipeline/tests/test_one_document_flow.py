import json
import types

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient, models

import rag.api as api
from rag.store import DENSE, count_source, delete_source, ensure_collections

DIM = 4


def _cfg(raw=None):
    return types.SimpleNamespace(
        qdrant=types.SimpleNamespace(collection="docs"),
        embedding=types.SimpleNamespace(dim=DIM),
        paths=types.SimpleNamespace(raw_dir=raw),
    )


def test_store_helpers_add_count_and_delete_one_document():
    client, cfg = QdrantClient(":memory:"), _cfg()
    ensure_collections(client, cfg)
    ensure_collections(client, cfg)  # second call must not fail or wipe anything
    pts = [
        models.PointStruct(id=i, vector={DENSE: [1.0] * DIM}, payload={"source": s, "type": "text", "page": 1})
        for i, s in enumerate(["a.pdf", "a.pdf", "b.pdf"])
    ]
    client.upsert("docs", points=pts)
    client.upsert("docs_parents", points=[
        models.PointStruct(id=10, vector=[1.0], payload={"source": "a.pdf"}),
        models.PointStruct(id=11, vector=[1.0], payload={"source": "b.pdf"}),
    ])
    assert (count_source(client, cfg, "a.pdf"), count_source(client, cfg, "b.pdf")) == (2, 1)

    ensure_collections(client, cfg)  # still keeps both documents
    assert count_source(client, cfg, "a.pdf") == 2

    delete_source(client, cfg, "a.pdf")  # removes chunks AND parents of a.pdf only
    assert (count_source(client, cfg, "a.pdf"), count_source(client, cfg, "b.pdf")) == (0, 1)
    assert client.count("docs_parents").count == 1


class FakePipeline:
    def __init__(self):
        self.chunks = {}  # name -> count
        self.client, self.cfg = object(), _cfg()
        self.indexed_calls = []

    def doc_status(self, name):
        return self.chunks.get(name, 0)

    def index_stream(self, source=None):
        self.indexed_calls.append(source)
        yield {"event": "start", "files": [source]}
        self.chunks[source] = 3
        yield {"event": "done", "chunks": 3, "parents": 1}


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    pipe = FakePipeline()
    monkeypatch.setattr(api, "get_pipeline", lambda: pipe)
    monkeypatch.setattr(api, "load_config", lambda: _cfg(tmp_path))
    monkeypatch.setattr(api, "delete_source", lambda client, cfg, name: pipe.chunks.pop(name, None))
    return TestClient(api.app), pipe, tmp_path


def _states(c):
    return {f["name"]: f["status"] for f in c.get("/files").json()}


def test_upload_index_one_document_and_tags(app_client):
    c, pipe, _ = app_client
    c.post("/upload", files=[("files", ("a.pdf", b"%PDF-a")), ("files", ("b.pdf", b"%PDF-b"))])
    assert _states(c) == {"a.pdf": "uploaded", "b.pdf": "uploaded"}

    r = c.post("/index/stream", params={"source": "a.pdf"})  # only a.pdf is processed
    events = [json.loads(x[5:]) for x in r.text.split("\n\n") if x.startswith("data:")]
    assert [e["event"] for e in events] == ["start", "done"]
    assert pipe.indexed_calls == ["a.pdf"]
    assert _states(c) == {"a.pdf": "indexed", "b.pdf": "uploaded"}  # b.pdf untouched

    # an indexed document is not processed a second time unless forced
    assert c.post("/index/stream", params={"source": "a.pdf"}).status_code == 409
    assert c.post("/index/stream", params={"source": "a.pdf", "force": True}).status_code == 200
    assert pipe.indexed_calls == ["a.pdf", "a.pdf"]


def test_missing_file_404_and_one_at_a_time(app_client):
    c, _, _ = app_client
    assert c.post("/index/stream", params={"source": "nope.pdf"}).status_code == 404

    c.post("/upload", files=[("files", ("a.pdf", b"%PDF-a"))])
    assert api._index_lock.acquire(blocking=False)  # pretend another document is running
    try:
        r = c.post("/index/stream", params={"source": "a.pdf"})
        assert r.status_code == 409 and "being processed" in r.json()["detail"]
    finally:
        api._index_lock.release()


def test_reupload_resets_index_status_and_rejects_non_pdf(app_client):
    c, pipe, _ = app_client
    c.post("/upload", files=[("files", ("a.pdf", b"%PDF-1"))])
    c.post("/index/stream", params={"source": "a.pdf"})
    assert _states(c)["a.pdf"] == "indexed"

    c.post("/upload", files=[("files", ("a.pdf", b"%PDF-2"))])  # new content: old chunks are stale
    assert _states(c)["a.pdf"] == "uploaded"
    assert c.post("/upload", files=[("files", ("x.txt", b"hi"))]).status_code == 400


def test_files_list_has_size_and_date_and_pdf_is_served(app_client):
    c, _, tmp = app_client
    c.post("/upload", files=[("files", ("a.pdf", b"%PDF-a"))])
    f = c.get("/files").json()[0]
    assert f["size"] == 6 and f["modified"] > 0 and f["collection"] == "docs"
    assert c.get("/pdf/a.pdf").headers["content-type"] == "application/pdf"
    assert c.get("/pdf/missing.pdf").status_code == 404
    assert c.get("/pdf/..%2Fsecret.txt").status_code == 404


def test_chunk_pages_have_ids_tokens_and_text_search():
    rows = [
        {"_id": "id-1", "source": "a.pdf", "page": 1, "type": "text", "text": "Recycled aluminum stand"},
        {"_id": "id-2", "source": "a.pdf", "page": 2, "type": "table", "text": "| a | b |"},
    ]
    d = api.page_chunks(rows, None, 0, 10, q="ALUMINUM")
    assert d["total"] == 1 and d["items"][0]["id"] == "id-1" and d["items"][0]["tokens"] >= 1
    assert d["counts"] == {"text": 1, "table": 1}  # tab counts ignore the search


def test_ask_returns_reranked_references(app_client):
    c, pipe, _ = app_client
    ranked = [{"id": "abc-123", "source": "a.pdf", "page": 3, "page_end": 3, "section": "S", "type": "text",
               "text": "Aluminum is recycled.", "score": 0.94}]
    pipe.ask = lambda q, source=None: types.SimpleNamespace(
        text="It is recycled.", refused=False, ranked=ranked,
        sources=[{"n": 1, "source": "a.pdf", "pages": "p.3", "section": "S", "type": "text", "snippet": "x"}])
    d = c.post("/ask", json={"question": "what?"}).json()
    assert d["references"][0]["id"] == "abc-123" and d["references"][0]["score"] == 0.94
    pipe.ask = lambda q, source=None: types.SimpleNamespace(text="I don't know", refused=True, ranked=ranked, sources=[])
    assert c.post("/ask", json={"question": "what?"}).json()["references"] == []


def test_ask_returns_the_full_cited_text_and_page_for_each_source(app_client):
    c, pipe, _ = app_client
    pipe.ask = lambda q, source=None: types.SimpleNamespace(
        text="GPO produced 15,901,600 passports [1].", refused=False, ranked=[],
        sources=[{"n": 1, "source": "a.pdf", "pages": "p.8", "page": 8, "section": "GPO by the Numbers",
                  "type": "text", "snippet": "short", "text": "The full entry. GPO produced 15,901,600 passports in FY 2018."}])
    s = c.post("/ask", json={"question": "passports?"}).json()["sources"][0]
    assert s["text"].startswith("The full entry") and s["page"] == 8 and s["n"] == 1
