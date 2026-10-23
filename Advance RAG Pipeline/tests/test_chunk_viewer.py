import types

from qdrant_client import QdrantClient, models

from rag.api import page_chunks
from rag.store import scroll_all


def _p(source, page, kind="text", text="t", **extra):
    return {"source": source, "page": page, "type": kind, "text": text, **extra}


def test_page_chunks_counts_filters_sorts_and_paginates():
    rows = [_p("a.pdf", 3), _p("a.pdf", 1, "table"), _p("a.pdf", 2), _p("a.pdf", 1, parent_id="x")]
    d = page_chunks(rows, None, 0, 3)
    assert d["total"] == 4 and d["counts"] == {"text": 3, "table": 1}
    assert [i["page"] for i in d["items"]] == [1, 1, 2]  # document order, first page of 3

    d = page_chunks(rows, None, 3, 3)
    assert [i["page"] for i in d["items"]] == [3]

    d = page_chunks(rows, "table", 0, 10)
    assert d["total"] == 1 and d["counts"] == {"text": 3, "table": 1}  # counts ignore the type filter
    assert d["items"][0]["page_end"] == 1 and d["items"][0]["has_parent"] is False


def test_scroll_all_filters_by_source_and_pages_through_everything():
    client = QdrantClient(":memory:")
    client.create_collection("docs", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
    pts = [
        models.PointStruct(id=i, vector=[1.0, 0.0], payload=_p("a.pdf" if i < 300 else "b.pdf", i))
        for i in range(400)
    ]
    client.upsert("docs", points=pts)
    cfg = types.SimpleNamespace(qdrant=types.SimpleNamespace(collection="docs"))
    assert len(scroll_all(client, cfg)) == 400  # more than one 256-point scroll page
    only_b = scroll_all(client, cfg, "b.pdf")
    assert len(only_b) == 100 and all(p.payload["source"] == "b.pdf" for p in only_b)
