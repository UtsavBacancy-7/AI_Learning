import types

import pymupdf
import pytest

from rag.gemini import QuotaExhausted, _guard
from rag.ingestion import VisionBudget, _describe_pages


def _pdf(tmp_path, pages=5):
    path = tmp_path / "doc.pdf"
    doc = pymupdf.open()
    for i in range(pages):
        doc.new_page().insert_text((72, 72), f"page {i + 1}")
    doc.save(path)
    return path


def _cfg(tmp_path):
    return types.SimpleNamespace(
        paths=types.SimpleNamespace(cache_dir=tmp_path / "cache"),
        parsing=types.SimpleNamespace(render_dpi=50),
        generation=types.SimpleNamespace(vision_model="m"),
    )


class FakeGemini:
    def __init__(self, fail_after=None):
        self.calls, self.fail_after = 0, fail_after

    def describe_image(self, png, prompt):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise QuotaExhausted("daily quota")
        return "a chart"


def test_budget_limits_calls_and_results_are_cached(tmp_path):
    path, cfg = _pdf(tmp_path), _cfg(tmp_path)
    doc, g, events = pymupdf.open(path), FakeGemini(), []
    out = _describe_pages(doc, path, cfg, g, [1, 2, 3, 4, 5], events.append, VisionBudget(2))
    assert g.calls == 2 and len(out) == 2
    assert any(e.get("skipped") == 3 for e in events)

    # second run: the 2 cached pages cost nothing, 2 more are described
    out = _describe_pages(doc, path, cfg, g, [1, 2, 3, 4, 5], None, VisionBudget(2))
    assert g.calls == 4 and len(out) == 4


def test_daily_quota_stops_calls_and_keeps_partial_results(tmp_path):
    path, cfg = _pdf(tmp_path), _cfg(tmp_path)
    doc, events = pymupdf.open(path), []
    g = FakeGemini(fail_after=1)
    g_workers = _describe_pages.__globals__
    g_workers["VISION_WORKERS"] = 1  # deterministic order
    out = _describe_pages(doc, path, cfg, g, [1, 2, 3, 4, 5], events.append, VisionBudget(0))
    assert len(out) == 1 and g.calls == 2  # stopped right after the quota error
    assert any(e.get("quota_hit") for e in events)


def test_guard_turns_daily_429_into_quota_exhausted_but_not_per_minute():
    @_guard
    def daily():
        raise RuntimeError("429 RESOURCE_EXHAUSTED ... GenerateRequestsPerDayPerProjectPerModel-FreeTier")

    @_guard
    def per_minute():
        raise RuntimeError("429 RESOURCE_EXHAUSTED ... GenerateRequestsPerMinutePerProjectPerModel")

    with pytest.raises(QuotaExhausted):
        daily()
    with pytest.raises(RuntimeError) as e:
        per_minute()
    assert not isinstance(e.value, QuotaExhausted)
