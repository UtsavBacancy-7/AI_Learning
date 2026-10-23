from rag.textclean import clean_text


def test_removes_markdown_leftovers_but_keeps_content():
    raw = "<!-- Start of picture text --> Design and<br>Source<br><!-- End of picture text -->"
    assert clean_text(raw) == "Design and Source"
    assert clean_text("in 13 components.<sup>14</sup> Next") == "in 13 components. Next"
    assert clean_text("**Aluminum.** We use 100 percent recycled aluminum.") == (
        "**Aluminum.** We use 100 percent recycled aluminum."
    )  # normal markdown is untouched
    assert clean_text("| a<br>b | c |") == "| a b | c |"
    assert clean_text("5 < 6 and 7 > 3") == "5 < 6 and 7 > 3"  # not mistaken for a tag
