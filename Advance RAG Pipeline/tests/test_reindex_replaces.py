import types

from qdrant_client import QdrantClient, models

import rag.pipeline as pl
import rag.store as store
from rag.ingestion import Chunk, Ingested


class FakeGemini:
    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


def _make_pipeline(tmp_path, monkeypatch, sizes):
    """A pipeline on an in-memory Qdrant. `sizes[name]` = (children, parents) that ingest returns next."""
    for n in ("a.pdf", "b.pdf"):
        (tmp_path / n).write_bytes(b"%PDF")
    cfg = types.SimpleNamespace(
        paths=types.SimpleNamespace(raw_dir=tmp_path),
        qdrant=types.SimpleNamespace(collection="docs"),
        embedding=types.SimpleNamespace(dim=4),
        parsing=types.SimpleNamespace(max_vision_pages_per_run=0),
    )
    p = object.__new__(pl.RAGPipeline)
    p.cfg, p.gemini, p.client = cfg, FakeGemini(), QdrantClient(":memory:")

    monkeypatch.setattr(store, "_bm25_doc", lambda text: models.SparseVector(indices=[1], values=[1.0]))

    def fake_ingest(path, cfg, gemini, progress=None, budget=None):
        n_children, n_parents, tag = sizes[path.name]
        meta = lambda i, t: {"source": path.name, "type": t, "page": 1, "page_end": 1}  # noqa: E731
        return Ingested(
            [Chunk(f"00000000-0000-0000-0000-{tag}{path.name[0]}{i:010d}"[:36], f"text {tag} {i}", meta(i, "text"))
             for i in range(n_children)],
            [Chunk(f"11111111-1111-1111-1111-{tag}{path.name[0]}{i:010d}"[:36], f"parent {tag} {i}", meta(i, "parent"))
             for i in range(n_parents)],
        )

    monkeypatch.setattr(pl, "ingest_pdf", fake_ingest)
    return p


def _count(p, name, collection="docs"):
    flt = models.Filter(must=[models.FieldCondition(key="source", match=models.MatchValue(value=name))])
    return p.client.count(collection, count_filter=flt, exact=True).count


def test_reindexing_removes_the_old_chunks_and_inserts_the_new_ones(tmp_path, monkeypatch):
    sizes = {"a.pdf": (3, 1, "1"), "b.pdf": (2, 1, "1")}
    p = _make_pipeline(tmp_path, monkeypatch, sizes)

    list(p.index_stream("a.pdf"))
    list(p.index_stream("b.pdf"))
    assert (_count(p, "a.pdf"), _count(p, "b.pdf")) == (3, 2)
    assert (_count(p, "a.pdf", "docs_parents"), _count(p, "b.pdf", "docs_parents")) == (1, 1)

    sizes["a.pdf"] = (5, 2, "2")  # the document changed: re-index it
    list(p.index_stream("a.pdf"))
    assert _count(p, "a.pdf") == 5  # old 3 chunks are gone, not added on top (that would be 8)
    assert _count(p, "a.pdf", "docs_parents") == 2
    texts = {pt.payload["text"] for pt in store.scroll_all(p.client, p.cfg, "a.pdf")}
    assert texts == {f"text 2 {i}" for i in range(5)}  # nothing from the first version is left
    assert (_count(p, "b.pdf"), _count(p, "b.pdf", "docs_parents")) == (2, 1)  # other documents untouched


def test_a_failed_parse_keeps_the_old_index_and_a_failed_store_leaves_nothing_half_done(tmp_path, monkeypatch):
    sizes = {"a.pdf": (3, 1, "1"), "b.pdf": (2, 1, "1")}
    p = _make_pipeline(tmp_path, monkeypatch, sizes)
    list(p.index_stream("a.pdf"))

    good_ingest = pl.ingest_pdf
    monkeypatch.setattr(pl, "ingest_pdf", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("parse crashed")))
    try:
        list(p.index_stream("a.pdf"))
    except RuntimeError:
        pass
    assert _count(p, "a.pdf") == 3  # the parse failed before anything was removed: old index still there

    monkeypatch.setattr(pl, "ingest_pdf", good_ingest)
    p.gemini.embed_documents = lambda texts: (_ for _ in ()).throw(RuntimeError("embedding failed"))
    try:
        list(p.index_stream("a.pdf"))
    except RuntimeError:
        pass
    assert _count(p, "a.pdf") == 0  # storing failed: no half-indexed document is left behind
