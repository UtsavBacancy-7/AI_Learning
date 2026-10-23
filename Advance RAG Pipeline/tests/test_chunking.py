from rag.chunking import ChunkParams, chunk_markdown, split_sentences

SAMPLE = (
    "Dr. Smith reviewed the Q3 results (see Fig. 2). Revenue grew 3.5% to $1.2 million, "
    "e.g. driven by U.S. sales. Was the trend expected? The board said yes! "
    "The next section covers risks."
)


def test_sample_sentence_boundaries():
    assert split_sentences(SAMPLE) == [
        "Dr. Smith reviewed the Q3 results (see Fig. 2).",
        "Revenue grew 3.5% to $1.2 million, e.g. driven by U.S. sales.",
        "Was the trend expected?",
        "The board said yes!",
        "The next section covers risks.",
    ]


def test_no_split_on_initials_urls_and_decimals():
    text = "J. Smith visited www.example.com today. Growth was 2.5 percent."
    assert len(split_sentences(text)) == 2


def test_wrapped_lines_and_lists():
    text = "The company grew in\nfiscal year\n2024. Profit rose.\n\n- First item.\n- Second item."
    s = split_sentences(text)
    assert s[0] == "The company grew in fiscal year 2024."
    assert s[-2:] == ["- First item.", "- Second item."]


def _long_text(n: int = 120) -> str:
    return " ".join(
        f"Sentence number {i} says that Dr. Lee measured {i}.5 units in the U.S. today."
        if i % 3
        else f"However, result {i} was not expected? It was, e.g. very odd!"
        for i in range(n)
    )


def test_chunks_never_cut_mid_sentence():
    text = _long_text()
    original = set(split_sentences(text))
    chunks = chunk_markdown(text, ChunkParams(target_tokens=120, dense_target_tokens=80))
    assert len(chunks) > 3
    for c in chunks:
        parts = split_sentences(c.text)
        assert parts and all(s in original for s in parts), c.text


def test_every_sentence_is_covered_and_overlap_is_whole_sentences():
    text = _long_text()
    original = split_sentences(text)
    chunks = chunk_markdown(text, ChunkParams(target_tokens=120, dense_target_tokens=80))
    covered = {s for c in chunks for s in split_sentences(c.text)}
    assert covered == set(original)


def test_oversized_sentence_stays_whole():
    big = "This is a very long sentence " + "with many words " * 400 + "that ends here."
    chunks = chunk_markdown(f"Short intro. {big} Short outro.", ChunkParams(target_tokens=100))
    assert any(big in c.text for c in chunks)


def test_table_is_atomic_and_header_repeats():
    rows = "\n".join(f"| row{i} | {i} |" for i in range(200))
    table = f"| name | value |\n|---|---|\n{rows}"
    chunks = chunk_markdown(table, ChunkParams(max_table_tokens=200))
    assert len(chunks) > 1
    for c in chunks:
        assert c.kind == "table"
        assert c.text.startswith("| name | value |")
        assert all(line.startswith("|") and line.endswith("|") for line in c.text.splitlines())


def test_headings_become_section_path():
    md = "# Report\n\n## Revenue\n\nRevenue grew. It was strong.\n\n## Risks\n\nRisks exist."
    chunks = chunk_markdown(md)
    assert [c.section for c in chunks] == ["Report > Revenue", "Report > Risks"]


def test_dense_text_gets_smaller_chunks():
    dense = " ".join(f"Q{i} revenue was 1{i}23.45 and profit 6{i}78.90 in 2024." for i in range(80))
    prose = " ".join(f"The team discussed topic {chr(65 + i % 26)} at length today." for i in range(80))
    assert len(chunk_markdown(dense)) > len(chunk_markdown(prose)) * 0.9
