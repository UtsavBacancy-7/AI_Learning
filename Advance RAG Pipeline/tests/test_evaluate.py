from types import SimpleNamespace

from rag.evaluate import evaluate, first_relevant_rank, is_relevant, summarize

CH = lambda page, text, end=None, source="a.pdf": {  # noqa: E731
    "source": source, "page": page, "page_end": end or page, "text": text,
}


def test_is_relevant_checks_source_pages_and_keywords():
    q = {"source": "a.pdf", "expected_pages": [5], "expected_keywords": ["growth"]}
    assert is_relevant(CH(4, "GDP Growth was 3%", end=6), q)  # page range overlaps
    assert not is_relevant(CH(9, "growth"), q)  # wrong page
    assert not is_relevant(CH(5, "inflation"), q)  # keyword missing
    assert not is_relevant(CH(5, "growth", source="b.pdf"), q)  # wrong file
    assert not is_relevant(CH(5, "x"), {})  # no expectations defined -> never relevant


def test_rank_and_summary():
    q = {"expected_keywords": ["gdp"]}
    assert first_relevant_rank([CH(1, "x"), CH(2, "GDP up")], q) == 2
    assert first_relevant_rank([CH(1, "x")], q) is None

    results = [
        {"type": "factual", "answerable": True, "refused": False, "rank": 1, "answer_ok": True},
        {"type": "factual", "answerable": True, "refused": False, "rank": 4, "answer_ok": False},
        {"type": "factual", "answerable": True, "refused": True, "rank": None, "answer_ok": None},
        {"type": "unanswerable", "answerable": False, "refused": True, "rank": None, "answer_ok": None},
    ]
    s = summarize(results)["overall"]
    assert s["hit@1"] == 0.333 and s["hit@5"] == 0.667
    assert s["mrr"] == round((1 + 0.25 + 0) / 3, 3)
    assert s["false_refusal_rate"] == 0.333 and s["correct_refusal_rate"] == 1.0
    assert s["answer_keyword_rate"] == 0.5


def test_evaluate_runs_pipeline_per_question():
    class Fake:
        def ask(self, question, source=None, generate=True):
            if "cake" in question:
                return SimpleNamespace(text="I don't know", refused=True, ranked=[])
            return SimpleNamespace(text="GDP grew 3%", refused=False, ranked=[CH(2, "GDP grew")])

    qs = [
        {"id": 1, "question": "gdp?", "answerable": True, "expected_keywords": ["gdp"]},
        {"id": 2, "question": "cake?", "type": "unanswerable", "answerable": False},
    ]
    results, summary = evaluate(Fake(), qs)
    assert results[0]["rank"] == 1 and results[0]["answer_ok"] is True
    assert results[1]["refused"] is True
    assert summary["overall"]["correct_refusal_rate"] == 1.0
