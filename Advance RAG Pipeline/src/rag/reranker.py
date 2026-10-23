"""Cross-encoder reranking (free, local). Scores each (question, chunk) pair together.

Small models (MiniLM, ~100 MB) run through fastembed/ONNX, which is already installed for BM25,
so no PyTorch is needed. Any other model name falls back to sentence-transformers.
"""
import math
from dataclasses import dataclass


@dataclass
class Scored:
    item: object
    score: float  # 0-1 (sigmoid of the model logit)


ANSWER_SCALE = 4.0  # answer scoring: logit/4 keeps supported (+9), weak (-2) and off-topic (-11) lines apart


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


class CrossEncoderReranker:
    def __init__(self, model_name: str, model=None):
        self.model_name = model_name
        self._model = model  # injectable for tests (needs .predict(pairs) -> logits)

    def _load(self):
        # Load lazily so importing the module stays cheap
        if self._model is not None:
            return self._model
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            if self.model_name in {m["model"] for m in TextCrossEncoder.list_supported_models()}:
                self._model = _FastembedModel(TextCrossEncoder(model_name=self.model_name))
                return self._model
        except ImportError:
            pass
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise RuntimeError(
                f"Cannot load the reranker '{self.model_name}'. Use a fastembed model such as "
                "Xenova/ms-marco-MiniLM-L-12-v2, or run: pip install sentence-transformers"
            ) from e
        self._model = CrossEncoder(self.model_name)
        return self._model

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        # Relevance probability (0-1) for arbitrary (query, passage) pairs; used by answer reranking
        if not pairs:
            return []
        return [sigmoid(float(x) / ANSWER_SCALE) for x in self._load().predict(pairs)]

    def rerank(self, query: str, items: list, texts: list[str], top_n: int) -> list[Scored]:
        # Score every candidate against the query, return the best top_n
        if not items:
            return []
        logits = self._load().predict([(query, t) for t in texts])
        scored = [Scored(it, sigmoid(float(s))) for it, s in zip(items, logits)]
        return sorted(scored, key=lambda s: -s.score)[:top_n]


class _FastembedModel:
    """Gives fastembed's reranker the same .predict(pairs) shape as sentence-transformers."""

    def __init__(self, encoder):
        self.encoder = encoder

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        # fastembed scores one query against many documents, so score each distinct query as a group
        groups: dict[str, list[int]] = {}
        for i, (query, _) in enumerate(pairs):
            groups.setdefault(query, []).append(i)
        out = [0.0] * len(pairs)
        for query, idx in groups.items():
            for i, s in zip(idx, self.encoder.rerank(query, [pairs[i][1] for i in idx])):
                out[i] = float(s)
        return out
