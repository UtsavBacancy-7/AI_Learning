from rag.chunking import split_sentences
from rag.chunking.parent_child import ParentParams, build_parent_child, looks_structured


def _para(topic: str, n: int) -> str:
    return " ".join(f"The {topic} figure for item {i} was 4.{i} percent, e.g. in the U.S." for i in range(n))


def _doc():
    pages = [
        (1, f"# Overview\n\n{_para('overview', 6)}"),
        (2, f"## Scope\n\nShort scope note.\n\n## Growth\n\n{_para('growth', 80)}"),
        (3, f"# Risks\n\n{_para('risk', 10)}\n\n| a | b |\n|---|---|\n| 1 | 2 |"),
    ]
    toc = [(1, "Overview", 1), (2, "Scope", 2), (2, "Growth", 2), (1, "Risks", 3)]
    return pages, toc


SMALL = ParentParams(parent_max_tokens=400, parent_min_tokens=100, child_target_tokens=80,
                     child_dense_target_tokens=60, min_pages=2, min_headings=2)


def test_children_link_to_parents_with_section_paths():
    pages, toc = _doc()
    parents, children = build_parent_child(pages, toc, SMALL)
    assert parents and children
    assert all(0 <= i < len(parents) for i, _ in children)
    sections = {p.section for p in parents}
    assert "Overview" in sections and "Overview > Growth" in sections and "Risks" in sections
    for i, c in children:
        assert c.section.startswith(parents[i].section.split(" > ")[0])


def test_small_section_merges_and_big_section_splits():
    pages, toc = _doc()
    parents, _ = build_parent_child(pages, toc, SMALL)
    assert not any(p.section == "Overview > Scope" for p in parents)  # tiny section merged
    growth = [p for p in parents if p.section == "Overview > Growth"]
    assert len(growth) > 1  # the 80-sentence section exceeds the parent cap
    assert all(len(p.text) // 4 > 50 for p in growth)  # no orphan heading-only parents
    original = set(split_sentences(_para("growth", 80)))
    for p in growth:  # splits are sentence-safe
        assert all(s in original or s.startswith("#") for s in split_sentences(p.text))


def test_children_are_sentence_safe():
    pages, toc = _doc()
    _, children = build_parent_child(pages, toc, SMALL)
    allowed = set()
    for _, md in pages:
        allowed |= set(split_sentences(md))
    for _, c in children:
        if c.kind == "text":
            assert all(s in allowed for s in split_sentences(c.text)), c.text


def test_sentence_continued_across_pages_is_stitched():
    pages = [(1, "# Intro\n\nRevenue grew strongly during the"), (2, "year and margins improved. Done.")]
    parents, children = build_parent_child(pages, None, SMALL)
    assert any("during the year and margins improved." in c.text for _, c in children)
    assert (parents[0].page_start, parents[0].page_end) == (1, 2)


def test_headings_without_toc_and_router():
    pages, _ = _doc()
    parents, _ = build_parent_child(pages, None, SMALL)
    assert any(p.section == "Overview > Scope" or p.section == "Overview" for p in parents)
    mds = [md for _, md in pages]
    assert looks_structured(mds, None, SMALL)
    assert not looks_structured(mds[:1], None, SMALL)
    assert looks_structured(mds, [(1, "x", 1)], SMALL)


def test_tables_stay_whole_children():
    pages, toc = _doc()
    _, children = build_parent_child(pages, toc, SMALL)
    tables = [c for _, c in children if c.kind == "table"]
    assert tables and tables[0].text.startswith("| a | b |")
