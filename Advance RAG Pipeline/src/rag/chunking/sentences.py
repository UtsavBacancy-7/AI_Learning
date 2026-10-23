"""Rule-based sentence splitter (no dependencies).

Errs on the side of NOT splitting: a missed boundary only makes a chunk a bit longer,
while a wrong boundary would cut a sentence in the middle.
"""
import re

# Abbreviations whose trailing period never ends a sentence
ABBREV = {
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "vs", "fig", "figs", "no", "nos",
    "vol", "eq", "eqs", "sec", "ch", "pp", "p", "cf", "e.g", "i.e", "approx", "dept",
    "ref", "refs", "tab", "ed", "eds", "est",
}

# Sentence end (. ? ! plus closing quote/bracket), whitespace, then something that starts a sentence
_BOUNDARY = re.compile(r"""([.?!]+["')\]]*)(\s+)(?=["'(\[]?[A-Z0-9])""")
# List markers (bullets or 1-2 digit numbers) that start a new unit
_LIST = re.compile(r"^\s*(?:[-*•+]|\d{1,2}[.)])\s+")
_LAST_TOKEN = re.compile(r"([^\s(\[\"']+)$")


def _units(text: str) -> list[str]:
    """Group lines into units: one paragraph, list item or heading each."""
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        cur: str | None = None
        for line in para.splitlines():
            if not line.strip():
                continue
            starts_new = (
                cur is None
                or _LIST.match(line)
                or line.lstrip().startswith("#")
                or cur.lstrip().startswith("#")
            )
            if starts_new:
                if cur is not None:
                    units.append(cur)
                cur = " ".join(line.split())
            else:
                cur += " " + " ".join(line.split())  # soft-wrapped line
        if cur is not None:
            units.append(cur)
    return units


def _is_real_boundary(body: str, match: re.Match) -> bool:
    punct = match.group(1)
    if not punct.startswith("."):
        return True  # ? and ! always end a sentence
    if len(punct) > 1:
        return True  # ellipsis
    m = _LAST_TOKEN.search(body[: match.start()])
    token = m.group(1).lower().rstrip(".") if m else ""
    if token in ABBREV:
        return False
    if re.fullmatch(r"[a-z]", token):  # initials like "J. Smith", "Option A."
        return False
    return True


def _split_unit(unit: str) -> list[str]:
    prefix = ""
    marker = _LIST.match(unit)
    if marker:
        prefix, unit = unit[: marker.end()], unit[marker.end():]
    sentences, start = [], 0
    for m in _BOUNDARY.finditer(unit):
        if _is_real_boundary(unit, m):
            sentences.append(unit[start : m.end(1)])
            start = m.end()
    sentences.append(unit[start:])
    sentences = [s.strip() for s in sentences if s.strip()]
    if prefix and sentences:
        sentences[0] = prefix + sentences[0]
    return sentences


def sentences_with_breaks(text: str) -> list[tuple[str, bool]]:
    """Return (sentence, starts_new_unit) pairs. A unit is a paragraph, list item or heading."""
    out: list[tuple[str, bool]] = []
    for unit in _units(text):
        for i, s in enumerate(_split_unit(unit)):
            out.append((s, i == 0))
    return out


def split_sentences(text: str) -> list[str]:
    return [s for s, _ in sentences_with_breaks(text)]
