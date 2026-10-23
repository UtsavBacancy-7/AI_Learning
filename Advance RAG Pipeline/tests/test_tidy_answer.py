from rag.textclean import tidy_answer

# The answer from the screenshot: repeated bullets, sentences glued after a citation, "percent" words
MESSY = (
    "The power adapter's efficiency at 115V is **0.4** percentage points higher than its efficiency at 100V [1].\n\n"
    "- Power adapter efficiency at **100V** is **91.1 percent** [1].\n"
    "- Power adapter efficiency at **115V** is **91.5 percent** [1].The power adapter's efficiency at **115V** is "
    "**0.4** percentage points higher than its efficiency at **100V** [1].\n"
    "- Power adapter efficiency at **100V** is **91.1%** [1].\n"
    "- Power adapter efficiency at **115V** is **91.5%** [1].The power adapter's efficiency at 115V is 0.4 "
    "percentage points higher than its efficiency at 100V [1].\n"
)


def test_percent_words_become_symbols_but_percentage_points_stay():
    out = tidy_answer("Efficiency is 91.1 percent, up 0.4 percentage points, from 50 per cent.")
    assert out == "Efficiency is 91.1%, up 0.4 percentage points, from 50%."


def test_repeated_lines_and_sentences_are_removed():
    out = tidy_answer(MESSY)
    assert out.count("0.4") == 1  # the difference is said once
    assert out.count("100V** is **91.1%") == 1 and out.count("115V** is **91.5%") == 1
    assert "percent" not in out.replace("percentage", "")
    assert out.splitlines()[0].startswith("The power adapter's efficiency at 115V is **0.4**")


def test_glued_sentences_are_split_and_structure_is_kept():
    out = tidy_answer("- First fact is 5 percent [1].Second fact is different [2].\n\n| a | b |\n|---|---|\n| 1 | 2 |")
    assert "- First fact is 5% [1]. Second fact is different [2]." in out
    assert out.endswith("| a | b |\n|---|---|\n| 1 | 2 |")  # tables are untouched


def test_a_clean_answer_is_unchanged():
    clean = "The stand uses **100% recycled aluminum** [1].\n\n- Plastic: at least 50% recycled in 13 components [2]."
    assert tidy_answer(clean) == clean


def test_same_sentence_repeated_three_times_in_one_line():
    # the answer from the screenshot: one sentence, three copies glued together after the citations
    one = "iMac consumes **58%** less energy than the requirement for ENERGY STAR [1][2]."
    repeated = one + one.replace("**58%** less energy", "**58% less energy**") + one
    assert tidy_answer(repeated) == one

    two = "**100%** of the materials in iMac are covered by Apple's Regulated Substances Specification [1]."
    assert tidy_answer(two + two + two) == two
    assert tidy_answer(two + "\n" + two) == two  # also when the copies are on separate lines


def test_drafts_are_never_glued_together():
    from types import SimpleNamespace as NS

    from rag.gemini import drafts_from

    def cand(*texts, thought=False):
        return NS(content=NS(parts=[NS(text=t, thought=thought) for t in texts]))

    # a normal response: one draft per candidate
    assert drafts_from([cand("A"), cand("B"), cand("C")], 3) == ["A", "B", "C"]
    # the 3 drafts packed as 3 parts of ONE candidate must stay 3 drafts, not "ABC"
    assert drafts_from([cand("A [1].", "B [1].", "C [1].")], 3) == ["A [1].", "B [1].", "C [1]."]
    # a single answer streamed in pieces is still one answer when only one draft was asked for
    assert drafts_from([cand("Hello ", "world")], 1) == ["Hello world"]
    # thinking parts are not part of the answer
    assert drafts_from([NS(content=NS(parts=[NS(text="reasoning", thought=True), NS(text="Answer", thought=False)]))], 1) == ["Answer"]
