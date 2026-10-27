"""Qdrant hybrid store: Gemini dense vectors + BM25 sparse vectors, fused with RRF."""
import os

from qdrant_client import QdrantClient, models

from .config import Config
from .gemini import Gemini
from .ingestion import Chunk

DENSE, SPARSE = "dense", "bm25"


def get_client(cfg: Config) -> QdrantClient:
    # Qdrant server/cloud if QDRANT_URL is set, otherwise local on-disk
    url = os.getenv("QDRANT_URL")
    if url:
        return QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY") or None,
                            timeout=cfg.qdrant.timeout_seconds)
    cfg.paths.qdrant_local_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(cfg.paths.qdrant_local_path))


def _bm25_doc(text: str) -> models.Document:
    # Qdrant builds the BM25 sparse vector from this text
    return models.Document(text=text, model="Qdrant/bm25")


def parents_name(cfg: Config) -> str:
    return f"{cfg.qdrant.collection}_parents"


def _source_filter(source: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))])


def ensure_collections(client: QdrantClient, cfg: Config) -> None:
    # Create the collections if missing. Existing documents are left untouched.
    name, pname = cfg.qdrant.collection, parents_name(cfg)
    if not client.collection_exists(name):
        client.create_collection(
            name,
            vectors_config={
                DENSE: models.VectorParams(size=cfg.embedding.dim, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
    # Parent sections: payload-only collection (dummy 1-dim vector), looked up by id
    if not client.collection_exists(pname):
        client.create_collection(
            pname, vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE)
        )
    # Indexes for fast filtering (creating one that already exists is harmless)
    for coll, fields in ((name, ("source", "type")), (pname, ("source",))):
        for field in fields:
            client.create_payload_index(coll, field, models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(name, "page", models.PayloadSchemaType.INTEGER)


def delete_source(client: QdrantClient, cfg: Config, source: str) -> None:
    # Remove one document's chunks and parent sections (before re-indexing it, or after a failure)
    for coll in (cfg.qdrant.collection, parents_name(cfg)):
        if client.collection_exists(coll):
            client.delete(coll, points_selector=models.FilterSelector(filter=_source_filter(source)))


def count_source(client: QdrantClient, cfg: Config, source: str) -> int:
    # How many chunks of this document are stored (0 = not indexed)
    if not client.collection_exists(cfg.qdrant.collection):
        return 0
    return client.count(cfg.qdrant.collection, count_filter=_source_filter(source), exact=True).count


def upsert_chunks(client: QdrantClient, cfg: Config, gemini: Gemini, chunks: list[Chunk]) -> None:
    for _ in upsert_chunks_iter(client, cfg, gemini, chunks):
        pass


def upsert_chunks_iter(client: QdrantClient, cfg: Config, gemini: Gemini, chunks: list[Chunk]):
    # Embed and upload chunks in batches; yields each batch once it is stored (for progress UIs)
    batch = 64
    for i in range(0, len(chunks), batch):
        part = chunks[i : i + batch]
        dense = gemini.embed_documents([c.embed_text for c in part])
        client.upsert(
            cfg.qdrant.collection,
            points=[
                models.PointStruct(
                    id=c.id,
                    vector={DENSE: d, SPARSE: _bm25_doc(c.embed_text)},
                    payload={**c.metadata, "text": c.text},
                )
                for c, d in zip(part, dense)
            ],
        )
        yield part


def upsert_parents(client: QdrantClient, cfg: Config, parents: list[Chunk]) -> None:
    # Store parent section text (no embedding needed)
    for i in range(0, len(parents), 64):
        client.upsert(
            parents_name(cfg),
            points=[
                models.PointStruct(id=p.id, vector=[1.0], payload={**p.metadata, "text": p.text})
                for p in parents[i : i + 64]
            ],
        )


def fetch_parents(client: QdrantClient, cfg: Config, ids: list[str]) -> dict[str, dict]:
    # Parent id -> payload
    if not ids:
        return {}
    pts = client.retrieve(parents_name(cfg), ids=ids, with_payload=True)
    return {str(p.id): p.payload for p in pts}


def scroll_all(client: QdrantClient, cfg: Config, source: str | None = None) -> list:
    # Every stored chunk (optionally of one document), without vectors
    flt = (
        models.Filter(must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))])
        if source
        else None
    )
    out, offset = [], None
    while True:
        pts, offset = client.scroll(
            cfg.qdrant.collection, scroll_filter=flt, limit=256, offset=offset,
            with_payload=True, with_vectors=False,
        )
        out += pts
        if offset is None:
            return out


def hybrid_search(
    client: QdrantClient,
    cfg: Config,
    gemini: Gemini,
    query: str,
    limit: int | None = None,
    source: str | None = None,
) -> list[models.ScoredPoint]:
    # Dense + sparse search, merged with RRF; optional filter by source file
    flt = (
        models.Filter(must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))])
        if source
        else None
    )
    k = cfg.retrieval.top_k_candidates
    res = client.query_points(
        cfg.qdrant.collection,
        prefetch=[
            models.Prefetch(query=gemini.embed_query(query), using=DENSE, limit=k, filter=flt),
            models.Prefetch(query=_bm25_doc(query), using=SPARSE, limit=k, filter=flt),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit or k,
        with_payload=True,
    )
    return res.points
