"""Answer reranking: pick the best of several drafted answers, and drop unsupported lines.

Every candidate is split into lines (paragraph / bullet / table row). Each line is scored by the
local cross-encoder against the context entries it cites (or the top entries if it cites none).

  support           0.7 * mean(line support) + 0.3 * weakest line
  number_grounding  share of numbers in the answer that appear verbatim in the context
  term_grounding    share of the answer's content words that appear in the context
  relevance         cross-encoder score of (question, answer)
  citation_validity share of cited lines whose cited context really supports them
  final = 0.35*support + 0.15*numbers + 0.20*terms + 0.15*relevance + 0.15*citations - length_penalty

The scorer is a relevance model, not an entailment model: treat the scores as a good proxy that
catches answers drifting away from the evidence, not as proof of correctness.
"""
import re
from dataclasses import dataclass, field
from typing import Callable

REFUSAL = "I don't know based on the provided records."

# scorer(pairs of (query, passage)) -> probabilities in 0-1
Scorer = Callable[[list[tuple[str, str]]], list[float]]

CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
LIST_MARK = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
TABLE_SEP = re.compile(r"^\s*\|[\s:|-]+\|?\s*$")

W_SUPPORT, W_NUMBERS, W_TERMS, W_RELEVANCE, W_CITES = 0.35, 0.15, 0.20, 0.15, 0.15
CITE_OK = 0.30  # a cited context must score at least this to count as supporting the line
DROP_BELOW = 0.25  # lines weaker than this are removed from the winning answer
MAX_WINDOWS = 12  # passages scored per context entry


@dataclass
class Candidate:
    text: str
    support: float = 0.0
    numbers: float = 1.0
    terms: float = 1.0
    relevance: float = 0.0
    cites: float = 0.0
    final: float = 0.0
    refused: bool = False
    dropped: list[str] = field(default_factory=list)


def is_refusal(text: str) -> bool:
    return text.strip().lower().startswith("i don't know based on the provided records")


def plain(line: str) -> str:
    """A line without list markers, table pipes, bold marks and citations."""
    line = CITE.sub("", LIST_MARK.sub("", line)).replace("**", "").replace("|", " ")
    return re.sub(r"\s+", " ", line).strip()


def windows(text: str, size: int = 900) -> list[str]:
    """Split a context entry into short passages (the cross-encoder only reads ~500 tokens)."""
    out, cur = [], ""
    for para in re.split(r"\n\s*\n|\n(?=\|)", text):
        para = para.strip()
        if not para:
            continue
        if cur and len(cur) + len(para) > size:
            out.append(cur)
            cur = ""
        while len(para) > size:  # one very long paragraph: cut at a sentence end when possible
            cut = max(para.rfind(". ", 0, size), para.rfind("; ", 0, size))
            cut = cut + 1 if cut > size // 3 else size
            out.append(para[:cut].strip())
            para = para[cut:].strip()
        cur = f"{cur}\n{para}".strip() if cur else para
    if cur:
        out.append(cur)
    return out[:MAX_WINDOWS] or [text[:size]]


STOP = set(
    "that this with from have been were will would which their there these those about into than then "
    "also does not provided context specify mentioned based document report according shows listed".split()
)
WORD = re.compile(r"[a-z0-9]{4,}")


def _terms(text: str) -> set[str]:
    """Content words, lightly normalised (lowercase, plural s removed)."""
    return {(w[:-1] if w.endswith("s") and len(w) > 4 else w) for w in WORD.findall(text.lower())} - STOP


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in NUMBER.findall(text)}


def evaluate(text: str, question: str, contexts: list[str], scorer: Scorer) -> Candidate:
    """Score one drafted answer. `contexts[n-1]` is the text cited as [n]."""
    cand = Candidate(text=text.strip())
    if not cand.text or is_refusal(cand.text):
        cand.refused = True
        return cand

    wins = [windows(c) for c in contexts]
    lines = []  # (original line, plain text, cited context indexes)
    for raw in cand.text.splitlines():
        body = plain(raw)
        if len(body) < 20 or TABLE_SEP.match(raw) or raw.lstrip().startswith("#"):
            continue
        cited = [int(n) - 1 for grp in CITE.findall(raw) for n in re.split(r"\s*,\s*", grp)
                 if 1 <= int(n) <= len(contexts)]
        lines.append((raw, body, cited))
    if not lines:
        return cand

    # one batch of (line, passage) pairs for the whole answer
    pairs, spans = [], []
    for _, body, cited in lines:
        idx = cited or list(range(min(3, len(contexts))))
        start = len(pairs)
        for i in idx:
            pairs += [(body, w) for w in wins[i]]
        spans.append((start, len(pairs), cited))
    scores = scorer(pairs) if pairs else []

    support, valid, cited_lines = [], 0, 0
    for (raw, _, _), (a, b, cited) in zip(lines, spans):
        best = max(scores[a:b], default=0.0)
        support.append(best)
        if cited:
            cited_lines += 1
            valid += best >= CITE_OK
        if best < DROP_BELOW and not raw.lstrip().startswith("|"):
            cand.dropped.append(raw)

    cand.support = 0.7 * sum(support) / len(support) + 0.3 * min(support)
    unmarked = "\n".join(LIST_MARK.sub("", ln) for ln in cand.text.splitlines())  # "1." is not a figure
    ans_numbers = _numbers(CITE.sub("", unmarked))
    ctx_numbers = _numbers(" ".join(contexts))
    cand.numbers = len(ans_numbers & ctx_numbers) / len(ans_numbers) if ans_numbers else 1.0
    ans_terms = _terms(CITE.sub("", cand.text))
    cand.terms = len(ans_terms & _terms(" ".join(contexts))) / len(ans_terms) if ans_terms else 1.0
    cand.relevance = scorer([(question, plain(cand.text))])[0]
    cand.cites = valid / cited_lines if cited_lines else 0.0
    penalty = min(0.1, max(0, len(cand.text.split()) - 250) / 2500)
    cand.final = (W_SUPPORT * cand.support + W_NUMBERS * cand.numbers + W_TERMS * cand.terms
                  + W_RELEVANCE * cand.relevance + W_CITES * cand.cites - penalty)
    return cand


# "[1]", "[1, 2]", "[1][2]" or "[1], [2], [3]": one group of citations after a claim
CLUSTER = re.compile(r"\[\d+(?:\s*,\s*\d+)*\](?:\s*,?\s*\[\d+(?:\s*,\s*\d+)*\])*")
SUFFICIENT = 0.5  # a cited context scoring at least this backs the claim by itself


def prune_citations(text: str, contexts: list[str], scorer: Scorer, sufficient: float = SUFFICIENT) -> str:
    """Keep ONE citation per claim when one source clearly backs it.

    Gemini often cites three entries for one fact. For every group of citations, each cited context is
    scored against the claim in front of it. If the best one is a direct match, only that one stays;
    if none is clear, the whole group is kept (better too many sources than a wrong one).
    """
    wins = [windows(c) for c in contexts]
    out_lines = []
    for raw in text.splitlines():
        if raw.lstrip().startswith(("|", "#")) or not CLUSTER.search(raw):
            out_lines.append(raw)
            continue
        pieces, pos = [], 0
        for m in CLUSTER.finditer(raw):
            claim = plain(raw[pos:m.start()])
            nums = sorted({int(n) for n in re.findall(r"\d+", m.group()) if 1 <= int(n) <= len(contexts)})
            keep = m.group()
            if len(nums) > 1 and len(claim) >= 12:
                pairs = [(claim, w) for n in nums for w in wins[n - 1]]
                scores = scorer(pairs)
                best, best_n, i = -1.0, nums[0], 0
                for n in nums:
                    s = max(scores[i : i + len(wins[n - 1])], default=0.0)
                    i += len(wins[n - 1])
                    if s > best:
                        best, best_n = s, n
                if best >= sufficient:
                    keep = f"[{best_n}]"
            pieces.append(raw[pos:m.start()] + keep)
            pos = m.end()
        out_lines.append("".join(pieces) + raw[pos:])
    return "\n".join(out_lines)


def pick_best(
    drafts: list[str], question: str, contexts: list[str], scorer: Scorer, min_support: float = 0.25
) -> tuple[str, bool, list[Candidate]]:
    """Return (answer text, refused, all scored candidates)."""
    scored = [evaluate(d, question, contexts, scorer) for d in drafts if d and d.strip()]
    real = [c for c in scored if not c.refused]
    if not real:
        return REFUSAL, True, scored
    best = max(real, key=lambda c: c.final)
    # refuse when the best draft is weakly supported, or when both its numbers and its wording
    # are mostly absent from the context (a fluent but invented answer)
    if best.support < min_support or (best.terms < 0.55 and best.numbers < 0.6):
        return REFUSAL, True, scored
    text = best.text
    for line in best.dropped:  # remove lines the evidence does not back up
        text = text.replace(line, "", 1)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return (text, False, scored) if text else (REFUSAL, True, scored)
