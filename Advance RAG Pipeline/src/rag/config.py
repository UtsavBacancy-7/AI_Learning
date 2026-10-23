import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


class Paths(BaseModel):
    raw_dir: Path
    qdrant_local_path: Path
    cache_dir: Path


class QdrantCfg(BaseModel):
    collection: str


class Parsing(BaseModel):
    render_dpi: int
    describe_visuals: bool
    max_vision_pages_per_run: int
    min_drawings_for_chart: int
    min_image_area_ratio: float


class Chunking(BaseModel):
    target_tokens: int
    dense_target_tokens: int
    min_tokens: int
    overlap_sentences: int
    max_table_tokens: int


class ParentChild(BaseModel):
    parent_max_tokens: int
    parent_min_tokens: int
    child_target_tokens: int
    child_dense_target_tokens: int
    min_pages: int
    min_headings: int


class Embedding(BaseModel):
    model: str
    dim: int


class Retrieval(BaseModel):
    top_k_candidates: int
    top_k_final: int
    max_context_tokens: int
    rerank_model: str
    min_rerank_score: float


class Generation(BaseModel):
    model: str
    vision_model: str
    temperature: float
    answer_rerank: bool = True  # draft several answers, keep the best-supported one
    answer_candidates: int = 3
    candidate_temperature: float = 0.7
    min_answer_support: float = 0.25


class RateLimits(BaseModel):
    # Free-tier limits per minute, kept a little below the real ones. 0 = no limit (paid tier).
    generate_per_minute: int = 4  # generate_content requests per model (free tier: 5)
    embed_texts_per_minute: int = 90  # texts sent to the embedding model (free tier: 100)
    embed_tokens_per_minute: int = 25000  # embedding tokens per minute (free tier: 30,000)


class Config(BaseModel):
    paths: Paths
    qdrant: QdrantCfg
    parsing: Parsing
    chunking: Chunking
    parent_child: ParentChild
    embedding: Embedding
    retrieval: Retrieval
    generation: Generation
    rate_limits: RateLimits = RateLimits()


def load_config(path: Path | None = None) -> Config:
    path = path or ROOT / "configs" / "default.yaml"
    cfg = Config(**yaml.safe_load(path.read_text()))
    # Make relative paths absolute from the project root
    for name in ("raw_dir", "qdrant_local_path", "cache_dir"):
        p = getattr(cfg.paths, name)
        setattr(cfg.paths, name, p if p.is_absolute() else ROOT / p)
    return cfg


def gemini_key() -> str:
    # Read the API key from the environment (.env)
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set (copy .env.example to .env)")
    return key
