"""Parent-child chunking for long structured documents (books, reports, annual reports).

  * Parent = a section (about 300-2500 tokens): what the LLM reads.
  * Child  = a small sentence-safe chunk inside a parent: what search matches.
Section structure comes from the PDF outline (TOC) when present, else from Markdown headings.
Tiny sections are merged into the next one, huge ones are split at block / sentence boundaries.
"""
import re
from dataclasses import dataclass, field

from .dynamic import (
    HEADING, Block, ChunkParams, DynamicChunk, Sent, chunk_blocks, est_tokens,
    join_sentences, markdown_blocks, pack_sentences,
)
from .sentences import sentences_with_breaks


@dataclass
class ParentParams:
    parent_max_tokens: int = 2500
    parent_min_tokens: int = 300
    child_target_tokens: int = 250
    child_dense_target_tokens: int = 200
    min_pages: int = 1  # router: documents shorter than this use plain dynamic chunking
    min_headings: int = 0  # router: headings needed when the PDF has no outline


@dataclass
class Parent:
    text: str
    section: str
    page_start: int
    page_end: int
    blocks: list[Block] = field(repr=False)
    base_path: list[tuple[int, str]] = field(repr=False)


@dataclass
class Unit:
    """One heading plus the blocks directly under it."""
    level: int
    path_before: list[tuple[int, str]]
    path: list[tuple[int, str]]
    blocks: list[Block] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return sum(est_tokens(b.text) for b in self.blocks)


TOC = list[tuple[int, str, int]]  # (level, title, page) like PyMuPDF get_toc()


def _norm(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", title.lower())


def _title_matches(a: str, b: str) -> bool:
    a, b = _norm(a), _norm(b)
    return bool(a and b and (a == b or a in b or b in a))


def looks_structured(pages_md: list[str], toc: TOC | None, pp: ParentParams) -> bool:
    """Router: long document with an outline or enough headings -> parent-child."""
    if len(pages_md) < pp.min_pages:
        return False
    if toc:
        return True
    n = sum(1 for md in pages_md for ln in md.splitlines() if HEADING.match(ln.strip()))
    return n >= pp.min_headings


def _stitch(blocks: list[Block]) -> list[Block]:
    """Join a sentence that continues across a page break (no end punctuation, lowercase start)."""
    out: list[Block] = []
    for b in blocks:
        prev = out[-1] if out else None
        if (
            prev and prev.kind == b.kind == "prose" and prev.page != b.page
            and not re.search(r"[.?!:][\"')\]]*\s*$", prev.text)
            and b.text.lstrip()[:1].islower()
        ):
            out[-1] = Block("prose", prev.text.rstrip() + " " + b.text.lstrip(), prev.page, b.page)
        else:
            out.append(b)
    return out


def _structure(pages: list[tuple[int, str]], toc: TOC | None) -> list[Block]:
    """Flatten pages into one block stream whose headings follow the TOC when there is one."""
    stream: list[Block] = []
    entries = list(toc or [])
    ptr = 0
    for page, md in pages:
        blocks = markdown_blocks(md, page)
        if not toc:
            stream += blocks
            continue
        due = []
        while ptr < len(entries) and entries[ptr][2] <= page:
            due.append(entries[ptr])
            ptr += 1
        anchors: dict[int, tuple[int, str, int]] = {}
        at_start = []
        for e in due:  # place each TOC heading where its text appears on the page
            idx = next(
                (i for i, b in enumerate(blocks)
                 if b.kind == "heading" and i not in anchors
                 and _title_matches(HEADING.match(b.text).group(2), e[1])),
                None,
            )
            if idx is None:
                at_start.append(e)
            else:
                anchors[idx] = e
        for lvl, title, _ in at_start:
            stream.append(Block("heading", f"{'#' * min(lvl, 6)} {title}", page))
        for i, b in enumerate(blocks):
            if i in anchors:
                lvl, title, _ = anchors[i]
                stream.append(Block("heading", f"{'#' * min(lvl, 6)} {title}", page))
            elif b.kind == "heading":  # not in the outline: plain text, not a section
                stream.append(Block("prose", HEADING.match(b.text).group(2), page))
            else:
                stream.append(b)
    return _stitch(stream)


def _units(stream: list[Block]) -> list[Unit]:
    units: list[Unit] = []
    path: list[tuple[int, str]] = []
    cur = Unit(0, [], [])
    for b in stream:
        if b.kind == "heading":
            if cur.blocks:
                units.append(cur)
            m = HEADING.match(b.text)
            level, title = len(m.group(1)), m.group(2).strip()
            before = list(path)
            while path and path[-1][0] >= level:
                path.pop()
            before = list(path)
            path.append((level, title))
            cur = Unit(level, before, list(path), [b])
        else:
            cur.blocks.append(b)
    if cur.blocks:
        units.append(cur)
    return units


def _is_content(blocks: list[Block]) -> bool:
    return any(b.kind != "heading" for b in blocks)


def _split_blocks(blocks: list[Block], max_tokens: int, pp: ParentParams) -> list[list[Block]]:
    """Split an oversized section into groups at block boundaries (and sentence boundaries)."""
    groups: list[list[Block]] = [[]]
    used = 0

    def add(b: Block) -> None:
        nonlocal used
        t = est_tokens(b.text)
        has_content = any(x.kind != "heading" for x in groups[-1])
        if has_content and used + t > max_tokens:  # a lone heading stays with the next block
            groups.append([])
            used = 0
        groups[-1].append(b)
        used += t

    for b in blocks:
        if b.kind == "prose" and est_tokens(b.text) > max_tokens:
            end = b.page_end if b.page_end is not None else b.page
            sents = [Sent(s, nu, b.page, end) for s, nu in sentences_with_breaks(b.text)]
            no_overlap = ChunkParams(overlap_sentences=0, min_tokens=0)
            for grp in pack_sentences(sents, max_tokens, no_overlap):
                add(Block("prose", join_sentences(grp), min(s.p0 for s in grp), max(s.p1 for s in grp)))
        else:
            add(b)
    return [g for g in groups if g]


def _make_parent(blocks: list[Block], path: list[tuple[int, str]], base: list[tuple[int, str]]) -> Parent:
    pages = [b.page for b in blocks] + [b.page_end for b in blocks if b.page_end]
    return Parent(
        text="\n\n".join(b.text for b in blocks),
        section=" > ".join(h for _, h in path),
        page_start=min(pages),
        page_end=max(pages),
        blocks=blocks,
        base_path=base,
    )


def _build_parents(units: list[Unit], pp: ParentParams) -> list[Parent]:
    merged: list[Unit] = []
    for u in units:
        acc = merged[-1] if merged else None
        if (
            acc and acc.tokens < pp.parent_min_tokens
            and acc.tokens + u.tokens <= pp.parent_max_tokens
            and u.level >= acc.level  # never merge into a higher-level (new chapter) heading
        ):
            acc.blocks += u.blocks
        else:
            merged.append(Unit(u.level, u.path_before, u.path, list(u.blocks)))

    parents: list[Parent] = []
    for u in merged:
        if not _is_content(u.blocks):
            continue
        groups = (
            [u.blocks] if u.tokens <= pp.parent_max_tokens
            else _split_blocks(u.blocks, pp.parent_max_tokens, pp)
        )
        for i, g in enumerate(groups):
            parents.append(_make_parent(g, u.path, u.path_before if i == 0 else u.path))
    return parents


def build_parent_child(
    pages: list[tuple[int, str]],
    toc: TOC | None = None,
    pp: ParentParams | None = None,
) -> tuple[list[Parent], list[tuple[int, DynamicChunk]]]:
    """Return parents and (parent_index, child_chunk) pairs."""
    pp = pp or ParentParams()
    parents = _build_parents(_units(_structure(pages, toc)), pp)
    child_params = ChunkParams(
        target_tokens=pp.child_target_tokens,
        dense_target_tokens=pp.child_dense_target_tokens,
        min_tokens=50,
        overlap_sentences=1,
        max_table_tokens=600,
    )
    children = [
        (i, c) for i, par in enumerate(parents) for c in chunk_blocks(par.blocks, child_params, par.base_path)
    ]
    return parents, children
