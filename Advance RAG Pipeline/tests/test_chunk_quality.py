from rag.chunking import ChunkParams, build_parent_child, chunk_markdown, looks_structured, split_sentences
from rag.chunking.dynamic import est_tokens
from rag.chunking.parent_child import ParentParams


def _prose(n):
    return " ".join(f"Item {i} was reviewed by the team and the results for part {i} were recorded in the log." for i in range(n))


def test_prose_chunks_are_150_to_300_tokens_with_small_sentence_overlap():
    chunks = chunk_markdown(_prose(120))
    sizes = [est_tokens(c.text) for c in chunks]
    assert len(chunks) > 4
    assert all(s <= 300 for s in sizes[:-1]), sizes  # none above 300 tokens
    assert all(s >= 150 for s in sizes[:-1]), sizes  # none below 150 (the last one may be a short tail)

    for prev, nxt in zip(chunks, chunks[1:]):  # overlap = whole sentences, at most ~50 tokens
        shared = [s for s in split_sentences(nxt.text) if s in split_sentences(prev.text)]
        assert shared and sum(est_tokens(s) for s in shared) <= 55


def test_table_keeps_its_label_with_the_values():
    md = "The two configurations are compared below:\n\n| Config | Display |\n|---|---|\n| 24-inch | 4.5K |"
    table = [c for c in chunk_markdown(md) if c.kind == "table"][0]
    assert table.text.startswith("The two configurations are compared below:\n| Config | Display |")

    plain = [c for c in chunk_markdown("Just a sentence.\n\n| a | b |\n|---|---|\n| 1 | 2 |") if c.kind == "table"][0]
    assert plain.text.startswith("| a | b |")  # an ordinary sentence is not mistaken for a label


def test_every_chunk_gets_a_parent_even_without_headings():
    pages = [(1, _prose(30)), (2, "| Config | Chip |\n|---|---|\n| Base | M4 |\n\n" + _prose(10))]
    parents, children = build_parent_child(pages, None, ParentParams())
    assert parents and children
    assert all(0 <= pi < len(parents) for pi, _ in children)  # nothing is left without a parent
    for pi, c in children:
        if c.kind == "text":  # the parent section really contains its child's text
            assert all(s in parents[pi].text for s in split_sentences(c.text))
    assert looks_structured([md for _, md in pages], None, ParentParams())  # short, unstructured docs too


def test_two_configuration_tables_share_one_parent_for_comparison():
    md = (
        "## Configurations\n\nThe 24-inch configuration is listed here:\n\n| Item | Value |\n|---|---|\n| Display | 4.5K |\n\n"
        "The 27-inch configuration is listed here:\n\n| Item | Value |\n|---|---|\n| Display | 5K |"
    )
    parents, children = build_parent_child([(1, md)], None, ParentParams())
    tables = [(pi, c) for pi, c in children if c.kind == "table"]
    assert len(tables) == 2 and tables[0][0] == tables[1][0]  # same parent section
    assert "24-inch" in tables[0][1].text and "27-inch" in tables[1][1].text
    assert "4.5K" in parents[tables[0][0]].text and "5K" in parents[tables[0][0]].text
