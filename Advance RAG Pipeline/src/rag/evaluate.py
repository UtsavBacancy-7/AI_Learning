"""Evaluation: retrieval hit-rate / MRR, answer keyword check, and refusal accuracy.

Question file (JSONL), one object per line:
  id, question, type (factual|table|chart|multi_section|unanswerable),
  answerable (default true), source (optional file filter),
  expected_pages (optional list of page numbers),
  expected_keywords (optional list; a retrieved chunk / answer matches if ANY/ALL appear)

All checks are rule-based, so the evaluation costs no LLM calls (use --no-answer to also
skip answer generation and only measure retrieval + refusal).
"""
import json
from collections import defaultdict
from pathlib import Path

KS = (1, 3, 5, 8)


def load_questions(path: Path) -> list[dict]:
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for r in rows:
        r.setdefault("answerable", r.get("type") != "unanswerable")
    return rows


def is_relevant(chunk: dict, q: dict) -> bool:
    """A retrieved chunk is relevant if it is from the right file, overlaps an expected page,
    and mentions an expected keyword (each check applies only when the question defines it)."""
    if q.get("source") and chunk.get("source") != q["source"]:
        return False
    pages = q.get("expected_pages")
    if pages:
        lo, hi = chunk["page"], chunk.get("page_end", chunk["page"])
        if not any(lo <= p <= hi for p in pages):
            return False
    kws = q.get("expected_keywords")
    if kws and not any(k.lower() in chunk["text"].lower() for k in kws):
        return False
    return bool(pages or kws)


def first_relevant_rank(ranked: list[dict], q: dict) -> int | None:
    for i, c in enumerate(ranked, 1):
        if is_relevant(c, q):
            return i
    return None


def answer_has_keywords(answer: str, q: dict) -> bool | None:
    kws = q.get("expected_keywords")
    return all(k.lower() in answer.lower() for k in kws) if kws else None


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 3) if xs else None


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results into overall and per-type metrics."""

    def block(rows: list[dict]) -> dict:
        ans = [r for r in rows if r["answerable"]]
        unans = [r for r in rows if not r["answerable"]]
        out = {"questions": len(rows)}
        for k in KS:
            out[f"hit@{k}"] = _mean([float(r["rank"] is not None and r["rank"] <= k) for r in ans])
        out["mrr"] = _mean([1 / r["rank"] if r["rank"] else 0.0 for r in ans])
        out["false_refusal_rate"] = _mean([float(r["refused"]) for r in ans])
        out["correct_refusal_rate"] = _mean([float(r["refused"]) for r in unans])
        kw = [r["answer_ok"] for r in ans if r["answer_ok"] is not None]
        out["answer_keyword_rate"] = _mean([float(x) for x in kw])
        return out

    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_type[r["type"]].append(r)
    return {"overall": block(results), "by_type": {t: block(rs) for t, rs in by_type.items()}}


def evaluate(pipeline, questions: list[dict], generate: bool = True) -> tuple[list[dict], dict]:
    """Run every question through the pipeline and score it."""
    results = []
    for q in questions:
        ans = pipeline.ask(q["question"], source=q.get("source"), generate=generate)
        results.append(
            {
                "id": q.get("id"),
                "type": q.get("type", "factual"),
                "answerable": q["answerable"],
                "refused": ans.refused,
                "rank": first_relevant_rank(ans.ranked, q),
                "answer_ok": answer_has_keywords(ans.text, q) if generate and not ans.refused else None,
                "answer": ans.text,
            }
        )
    return results, summarize(results)
