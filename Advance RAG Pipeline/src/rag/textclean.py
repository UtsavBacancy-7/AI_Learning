"""Text cleanup: PDF-to-Markdown leftovers before indexing, and a final tidy-up for answers."""
import re

_PICTURE_MARKER = re.compile(r"<!--\s*(?:Start|End) of picture text\s*-->", re.I)
_SUP = re.compile(r"<sup>.*?</sup>", re.I | re.S)  # footnote numbers like components.<sup>14</sup>
_BREAK = re.compile(r"<br\s*/?>", re.I)
_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_SPACES = re.compile(r"[ \t]{2,}")


def clean_text(text: str) -> str:
    text = _PICTURE_MARKER.sub("", text)
    text = _SUP.sub("", text)
    text = _BREAK.sub(" ", text)
    text = _TAG.sub("", text)
    return _SPACES.sub(" ", text).strip()


# ---------------------------------------------------------------- answers

_PERCENT = re.compile(r"(\d)\s*(?:per cent|percent)\b(?!age)", re.I)  # "91.1 percent" -> "91.1%"
_AFTER_CITE = re.compile(r"(?<=\]\.)\s*(?=\S)")  # a sentence ends at "[n]." even if the next word is lowercase
_GLUED = re.compile(r"(\[\d+(?:\s*,\s*\d+)*\])(?=[A-Z])")  # "[1]The next" -> "[1] The next"
_LIST_MARK = re.compile(r"^(\s*(?:[-*•]|\d+[.)])\s+)")
_CITE = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")
_WORD = re.compile(r"[a-z0-9.%$]+")


def _key(sentence: str) -> str:
    """What a sentence says, ignoring case, markdown, citations and punctuation."""
    s = _CITE.sub("", sentence).replace("**", "").lower()
    return " ".join(_WORD.findall(s)).strip(" .")


def tidy_answer(text: str) -> str:
    """Final polish for an answer: numbers as %, glued sentences split, repeated lines removed."""
    from .chunking.sentences import split_sentences  # local import: keeps this module light

    text = _PERCENT.sub(r"\1%", text)
    text = _GLUED.sub(r"\1 ", text)

    seen: set[str] = set()
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith(("|", "#")):  # blank lines, tables, headings
            out.append(line)
            continue
        m = _LIST_MARK.match(line)
        prefix, body = (m.group(1), line[m.end():]) if m else ("", line)
        kept = []
        pieces = [s for chunk in _AFTER_CITE.split(body) for s in split_sentences(chunk)]
        for sentence in pieces:
            k = _key(sentence)
            if k and k in seen:
                continue  # this sentence was already said
            seen.add(k)
            kept.append(sentence)
        if kept:
            out.append(prefix + " ".join(kept))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
