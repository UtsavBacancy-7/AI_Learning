"""Dynamic chunking: size adapts to content, boundaries are always sentence ends.

Rules:
  * Prose is packed from whole sentences only, never cut mid-sentence.
  * Overlap is whole sentences, never characters.
  * Tables, code blocks and headings are handled as atomic blocks.
  * Dense text (many numbers) gets smaller chunks than plain prose.
"""
import re
from dataclasses import dataclass
from typing import NamedTuple

from .sentences import sentences_with_breaks

TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$")
HEADING = re.compile(r"^(#{1,6})\s+(.*)")
# A line that labels the table that follows: "Table 3 ...", "Configuration: ...", "... as follows:"
CAPTION = re.compile(r"(^(table|figure|exhibit|chart)\b)|(:\s*$)", re.I)
# Sentences that depend on the previous one: avoid starting a new chunk with them
DEPENDENT = re.compile(
    r"^(however|therefore|thus|hence|moreover|furthermore|additionally|also|"
    r"this|these|those|that|it|they|he|she|as a result|for example|in addition)\b",
    re.I,
)


@dataclass
class ChunkParams:
    target_tokens: int = 250   # ordinary prose: 150-300 tokens
    dense_target_tokens: int = 200
    min_tokens: int = 50
    overlap_sentences: int = 1
    max_table_tokens: int = 1000


@dataclass
class Block:
    kind: str  # heading | table | code | prose
    text: str
    page: int = 0
    page_end: int | None = None  # set when a block spans pages


@dataclass
class DynamicChunk:
    text: str
    kind: str  # text | table | code
    section: str = ""  # heading path, e.g. "Chapter 1 > Overview"
    page_start: int = 0
    page_end: int = 0


class Sent(NamedTuple):
    text: str
    new_unit: bool  # starts a paragraph / list item
    p0: int
    p1: int


def est_tokens(text: str) -> int:
    return max(1, len(text) // 4)  # ~4 characters per token


def markdown_blocks(md: str, page: int = 0) -> list[Block]:
    """Split markdown into heading, table, code and prose blocks."""
    lines, i, out = md.splitlines(), 0, []

    def is_special(line: str) -> bool:
        s = line.strip()
        return s.startswith("```") or bool(TABLE_LINE.match(line)) or bool(HEADING.match(s))

    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
        elif line.strip().startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("```"):
                j += 1
            out.append(Block("code", "\n".join(lines[i : j + 1]), page))
            i = j + 1
        elif TABLE_LINE.match(line):
            j = i
            while j < len(lines) and TABLE_LINE.match(lines[j]):
                j += 1
            out.append(Block("table", "\n".join(lines[i:j]), page))
            i = j
        elif HEADING.match(line.strip()):
            out.append(Block("heading", line.strip(), page))
            i += 1
        else:
            j = i
            while j < len(lines) and lines[j].strip() and not is_special(lines[j]):
                j += 1
            out.append(Block("prose", "\n".join(lines[i:j]), page))
            i = j
    return out


def _digit_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    return sum(c.isdigit() for c in chars) / len(chars) if chars else 0.0


def join_sentences(sents: list[Sent]) -> str:
    text = ""
    for k, s in enumerate(sents):
        text += s.text if k == 0 else ("\n" if s.new_unit else " ") + s.text
    return text


def pack_sentences(sents: list[Sent], target: int, p: ChunkParams) -> list[list[Sent]]:
    """Pack whole sentences into chunks of about `target` tokens (with sentence overlap)."""
    max_tok, hard = int(target * 1.2), int(target * 1.5)
    chunks: list[list[Sent]] = []
    cur: list[Sent] = []
    cur_tok = overlap_n = 0

    def flush() -> None:
        nonlocal cur, cur_tok, overlap_n
        chunks.append(cur)
        carry: list[Sent] = []
        for item in reversed(cur[-p.overlap_sentences :] if p.overlap_sentences else []):
            if sum(est_tokens(x.text) for x in carry) + est_tokens(item.text) > 0.2 * target:
                break
            carry.insert(0, item)
        cur, overlap_n = carry, len(carry)
        cur_tok = sum(est_tokens(x.text) for x in cur)

    for s in sents:
        t = est_tokens(s.text)
        if len(cur) > overlap_n:  # has new content; decide whether to cut before s
            total = cur_tok + t
            soft_ok = not DEPENDENT.match(s.text) and not cur[-1].text.rstrip().endswith(":")
            if (
                total > hard
                or (total > max_tok and soft_ok)
                or (cur_tok >= target and soft_ok)
                or (s.new_unit and cur_tok >= 0.6 * target and soft_ok)
            ):
                flush()
        cur.append(s)
        cur_tok += t

    if len(cur) > overlap_n:  # leftover with new content
        fresh = cur[overlap_n:]
        fresh_tok = sum(est_tokens(x.text) for x in fresh)
        if chunks and fresh_tok < p.min_tokens:
            prev_tok = sum(est_tokens(x.text) for x in chunks[-1])
            if prev_tok + fresh_tok <= hard:  # merge a tiny tail into the previous chunk
                chunks[-1] = chunks[-1] + fresh
                return chunks
        chunks.append(cur)
    return chunks


def _split_table(table: str, max_tokens: int) -> list[str]:
    """Split a big table by rows, repeating the header. Rows are never cut."""
    rows = table.splitlines()
    if est_tokens(table) <= max_tokens or len(rows) <= 3:
        return [table]
    n_head = 2 if len(rows) > 1 and re.match(r"^\s*\|[\s:|-]+\|\s*$", rows[1]) else 1
    header, parts, cur = rows[:n_head], [], rows[:n_head]
    for r in rows[n_head:]:
        if est_tokens("\n".join(cur + [r])) > max_tokens and len(cur) > n_head:
            parts.append("\n".join(cur))
            cur = list(header)
        cur.append(r)
    parts.append("\n".join(cur))
    return parts


def _split_code(code: str, max_tokens: int) -> list[str]:
    """Split a big code block at blank lines. Lines are never cut."""
    if est_tokens(code) <= max_tokens:
        return [code]
    parts, cur = [], ""
    for group in re.split(r"\n\s*\n", code):
        if cur and est_tokens(cur + "\n\n" + group) > max_tokens:
            parts.append(cur)
            cur = group
        else:
            cur = f"{cur}\n\n{group}" if cur else group
    return parts + [cur] if cur else parts


def chunk_blocks(
    blocks: list[Block],
    params: ChunkParams | None = None,
    base_path: list[tuple[int, str]] | None = None,
) -> list[DynamicChunk]:
    """Chunk a block stream. `base_path` is the heading path that applies before the first block."""
    p = params or ChunkParams()
    out: list[DynamicChunk] = []
    path = list(base_path or [])
    prose: list[Sent] = []
    caption = ""  # a short lead-in / "Table 3" line right before a table, kept with that table

    def section() -> str:
        return " > ".join(h for _, h in path)

    def flush_prose() -> None:
        nonlocal prose
        if prose:
            target = p.dense_target_tokens if _digit_ratio(join_sentences(prose)) > 0.12 else p.target_tokens
            for grp in pack_sentences(prose, target, p):
                out.append(
                    DynamicChunk(
                        join_sentences(grp), "text", section(),
                        min(s.p0 for s in grp), max(s.p1 for s in grp),
                    )
                )
            prose = []

    for b in blocks:
        end = b.page_end if b.page_end is not None else b.page
        if b.kind == "heading":
            flush_prose()
            caption = ""
            m = HEADING.match(b.text)
            level, title = len(m.group(1)), m.group(2).strip()
            while path and path[-1][0] >= level:
                path.pop()
            path.append((level, title))
        elif b.kind == "prose":
            prose += [Sent(s, nu, b.page, end) for s, nu in sentences_with_breaks(b.text)]
            last = b.text.strip().splitlines()[-1].strip()
            caption = last if len(last) <= 200 and CAPTION.search(last) else ""
        else:
            flush_prose()
            pieces = (
                _split_table(b.text, p.max_table_tokens)
                if b.kind == "table"
                else _split_code(b.text, p.max_table_tokens)
            )
            if b.kind == "table" and caption:  # keep the table's label with its values
                pieces = [caption + "\n" + x for x in pieces]
            caption = ""
            out += [DynamicChunk(x, b.kind, section(), b.page, end) for x in pieces]
    flush_prose()
    return out


def chunk_markdown(md: str, params: ChunkParams | None = None, page: int = 0) -> list[DynamicChunk]:
    return chunk_blocks(markdown_blocks(md, page), params)
