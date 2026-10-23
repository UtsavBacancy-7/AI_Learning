import types

import pytest

from rag.answer_rerank import REFUSAL, evaluate, is_refusal, pick_best, windows
from rag.gemini import CandidatesUnsupported, Gemini, _guard

CONTEXTS = [
    "The stand is made from 100 percent recycled aluminum.\n\nThe enclosure uses 35 percent recycled plastic.",
    "Revenue grew 3 percent in 2024.",
]
QUESTION = "What material is the stand made from?"


def overlap_scorer(pairs):
    """Fake relevance model: the share of the query's words found in the passage."""
    out = []
    for q, p in pairs:
        qs = {w for w in q.lower().replace("[", " ").replace("]", " ").split() if len(w) > 3}
        out.append(len(qs & set(p.lower().split())) / len(qs) if qs else 0.0)
    return out


GOOD = "The stand is made from **100 percent recycled aluminum** [1]."
BAD = "The stand is made from carbon fibre and titanium bolts sourced in 1999 [1]."


def test_supported_answer_beats_unsupported_one():
    good = evaluate(GOOD, QUESTION, CONTEXTS, overlap_scorer)
    bad = evaluate(BAD, QUESTION, CONTEXTS, overlap_scorer)
    assert good.final > bad.final
    assert good.numbers == 1.0 and bad.numbers < 1.0  # 1999 is nowhere in the context
    assert good.cites == 1.0 and good.support > bad.support


def test_pick_best_returns_the_winner_and_all_scores():
    text, refused, scored = pick_best([BAD, GOOD], QUESTION, CONTEXTS, overlap_scorer)
    assert text == GOOD and not refused and len(scored) == 2


def test_refusals_and_weak_support():
    assert is_refusal("I don't know based on the provided records.")
    assert pick_best([REFUSAL, REFUSAL], QUESTION, CONTEXTS, overlap_scorer)[1] is True
    # a refusal draft is ignored when another draft is a real answer
    assert pick_best([REFUSAL, GOOD], QUESTION, CONTEXTS, overlap_scorer)[0] == GOOD
    # nothing in the answer matches the evidence at all -> refuse
    off_topic = "Zebras migrate across the savanna every single year [1]."
    assert pick_best([off_topic], QUESTION, CONTEXTS, overlap_scorer, min_support=0.3)[1] is True


def test_unsupported_lines_are_dropped_but_structure_is_kept():
    answer = (
        "The stand is made from **100 percent recycled aluminum** [1].\n\n"
        "- Aluminum is recycled aluminum for the stand [1]\n"
        "- Penguins enjoy cold weather near glaciers somewhere far away [1]\n\n"
        "| Part | Material |\n|---|---|\n| Stand | Aluminum |"
    )
    text, refused, _ = pick_best([answer], QUESTION, CONTEXTS, overlap_scorer)
    assert not refused
    assert "Penguins" not in text and "recycled aluminum" in text
    assert "| Part | Material |" in text and "| Stand | Aluminum |" in text  # table rows are kept


def test_windows_split_long_contexts_without_losing_text():
    para = ". ".join(f"Sentence number {i} about the stand" for i in range(80)) + "."
    parts = windows(para, size=300)
    assert 1 < len(parts) <= 12 and all(len(p) <= 300 for p in parts)


def test_guard_maps_candidate_errors_and_generate_candidates_falls_back():
    @_guard
    def boom():
        raise RuntimeError("400 INVALID_ARGUMENT: Multiple candidates is not enabled for this model")

    with pytest.raises(CandidatesUnsupported):
        boom()

    g = object.__new__(Gemini)
    calls = []
    g._candidates = lambda p, n, t: (_ for _ in ()).throw(CandidatesUnsupported("no"))
    g.generate = lambda p, **k: calls.append(p) or "single answer"
    assert g.generate_candidates("prompt", 3, 0.7) == ["single answer"] and calls == ["prompt"]
    g._candidates = lambda p, n, t: ["a", "b", "c"]
    assert g.generate_candidates("prompt", 3, 0.7) == ["a", "b", "c"]
    assert g.generate_candidates("prompt", 1, 0.7) == ["single answer"]


def test_fluent_but_invented_answer_is_refused():
    invented = "The stand is made of 98 percent recycled steel sourced from Germany in 1987 [1]."
    text, refused, scored = pick_best([invented], QUESTION, CONTEXTS, overlap_scorer)
    assert refused and text == REFUSAL
    assert scored[0].terms < 0.55 and scored[0].numbers < 0.6


# ---- one citation per claim -------------------------------------------------
CITE_CONTEXTS = [
    "Staff cataloged a total of 34,367 titles in FY 2018.",  # [1] backs the claim
    "The agency reports on printing and binding volumes for several years.",  # [2] unrelated
    "Board members met four times during the fiscal year.",  # [3] unrelated
]


def test_three_citations_for_one_fact_become_the_single_best_one():
    from rag.answer_rerank import prune_citations

    text = "GPO staff cataloged a total of 34,367 titles in FY 2018 [1], [2], [3]."
    assert prune_citations(text, CITE_CONTEXTS, overlap_scorer) == (
        "GPO staff cataloged a total of 34,367 titles in FY 2018 [1]."
    )
    assert prune_citations(text.replace("[1], [2], [3]", "[3][1][2]"), CITE_CONTEXTS, overlap_scorer).endswith("2018 [1].")
    assert prune_citations(text.replace("[1], [2], [3]", "[2, 3, 1]"), CITE_CONTEXTS, overlap_scorer).endswith("2018 [1].")


def test_citations_are_kept_when_no_single_source_is_clearly_enough():
    from rag.answer_rerank import prune_citations

    vague = "Zebras migrate across wide savannas [2], [3]."  # nothing backs this: keep what the model wrote
    assert prune_citations(vague, CITE_CONTEXTS, overlap_scorer) == vague


def test_each_claim_keeps_its_own_citation_and_tables_are_untouched():
    from rag.answer_rerank import prune_citations

    text = (
        "Staff cataloged a total of 34,367 titles in FY 2018 [1], [2]. Board members met four times during the fiscal year [3].\n"
        "| a [1], [2] | b |"
    )
    out = prune_citations(text, CITE_CONTEXTS, overlap_scorer)
    assert "34,367 titles in FY 2018 [1]." in out and "fiscal year [3]." in out
    assert out.endswith("| a [1], [2] | b |")  # table rows are not rewritten
    assert prune_citations("No citations here at all.", CITE_CONTEXTS, overlap_scorer) == "No citations here at all."
