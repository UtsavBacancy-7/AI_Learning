from rag.reranker import CrossEncoderReranker


class FakeModel:
    """Scores a pair by how many query words appear in the passage."""

    def predict(self, pairs):
        return [sum(w in p.lower() for w in q.lower().split()) * 2 - 3 for q, p in pairs]


def test_best_match_ranks_first_and_top_n_applies():
    r = CrossEncoderReranker("fake", model=FakeModel())
    texts = ["cats sleep", "revenue grew in 2024", "revenue fell", "unrelated text"]
    out = r.rerank("revenue grew 2024", list(range(4)), texts, top_n=2)
    assert [s.item for s in out] == [1, 2]
    assert all(0 <= s.score <= 1 for s in out)
    assert out[0].score > out[1].score


def test_empty_candidates():
    assert CrossEncoderReranker("fake", model=FakeModel()).rerank("q", [], [], 5) == []


def test_missing_sentence_transformers_gives_a_clear_error(monkeypatch):
    import sys

    import pytest

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # makes the import fail
    with pytest.raises(RuntimeError, match="pip install sentence-transformers"):
        CrossEncoderReranker("x").rerank("q", [1], ["t"], 1)
