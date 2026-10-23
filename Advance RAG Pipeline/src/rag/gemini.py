"""Gemini client: embeddings, text, vision. Retries, rate limiting and free-tier quota handling.

The free tier has two kinds of limits, and both are handled here:
  * per MINUTE (e.g. 5 generate requests, 100 embedding texts): we pace our own calls with a
    RateLimiter so we stay under them, and if Google still answers 429 we wait as long as it asks.
  * per DAY (e.g. 20 generate requests): waiting does not help, so we stop at once (QuotaExhausted).
"""
import functools
import logging
import re
import threading
import time
from collections import deque

from google import genai
from google.genai import types
from tenacity import (
    before_sleep_log, retry, retry_if_not_exception_type, stop_after_attempt,
)

from .config import Config, gemini_key

log = logging.getLogger("rag.gemini")


class QuotaExhausted(RuntimeError):
    """The free-tier DAILY quota is used up. Retrying is pointless, so we stop immediately."""


class CandidatesUnsupported(RuntimeError):
    """The model does not allow more than one answer candidate per request."""


class RateLimiter:
    """Allows at most `limit` units in any sliding window of `window` seconds. limit 0 = no limit."""

    def __init__(self, limit: int, window: float = 60.0, clock=time.monotonic, sleep=time.sleep):
        self.limit, self.window, self._clock, self._sleep = limit, window, clock, sleep
        self._events: deque[tuple[float, int]] = deque()  # (time used, units)
        self._lock = threading.Lock()

    def acquire(self, units: int = 1) -> None:
        """Block until `units` can be used without going over the limit, then use them."""
        if self.limit <= 0:
            return
        units = min(units, self.limit)  # a single call bigger than the limit still has to go through
        while True:
            with self._lock:
                now = self._clock()
                while self._events and now - self._events[0][0] >= self.window:
                    self._events.popleft()
                used = sum(u for _, u in self._events)
                if used + units <= self.limit:
                    self._events.append((now, units))
                    return
                # wait until enough of the oldest usage has left the window
                freed, wait = used, 0.0
                for t, u in self._events:
                    freed -= u
                    wait = t + self.window - now
                    if freed + units <= self.limit:
                        break
            if wait > 5:
                log.info("Free-tier rate limit: waiting %.0fs before the next Gemini call", wait)
            self._sleep(max(wait, 0.05))


def _retry_wait(state) -> float:
    """How long to wait before retrying: as long as Gemini asks ("retry in 47s"), else back off."""
    text = str(state.outcome.exception())
    m = re.search(r"retry in ([\d.]+)\s*(ms|s)\b", text) or re.search(r"'retryDelay': '([\d.]+)()s'", text)
    if m:
        secs = float(m.group(1)) / (1000 if m.group(2) == "ms" else 1)
        return min(90.0, secs + 1.0)
    return min(30.0, 2.0 ** state.attempt_number)


def _guard(fn):
    # Turn a "per day" 429 into QuotaExhausted (per-minute 429s are still retried, after the wait Gemini asks for)
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except QuotaExhausted:
            raise
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) and "PerDay" in str(e):
                raise QuotaExhausted(
                    "Gemini free-tier daily quota reached (resets within 24h). "
                    "Wait, switch generation.model / vision_model in configs/default.yaml, or use a paid key."
                ) from e
            if "candidate" in str(e).lower() and ("400" in str(e) or "INVALID_ARGUMENT" in str(e)):
                raise CandidatesUnsupported(str(e)) from e
            raise

    return wrapper


_retry = retry(
    retry=retry_if_not_exception_type((QuotaExhausted, CandidatesUnsupported)),
    stop=stop_after_attempt(6),
    wait=_retry_wait,
    before_sleep=before_sleep_log(log, logging.WARNING),  # log retries
    reraise=True,
)


def drafts_from(candidates, n: int) -> list[str]:
    """One draft per candidate. Thought parts are ignored, and separate drafts are never glued together.

    Some models return the n drafts as n text parts of a SINGLE candidate. Joining those parts
    produced one answer with the same text repeated n times, so they are kept as separate drafts.
    """
    per_cand = [
        [p.text for p in c.content.parts if p.text and not getattr(p, "thought", False)]
        for c in candidates
        if c.content and c.content.parts
    ]
    if len(per_cand) == 1 and n > 1 and len(per_cand[0]) > 1:
        return [t.strip() for t in per_cand[0] if t.strip()]
    return [joined for parts in per_cand if (joined := "".join(parts).strip())]


class Gemini:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = genai.Client(api_key=gemini_key())
        # The limit is per model, so vision and answers share a limiter when they use the same model
        self._gen_limiters: dict[str, RateLimiter] = {}
        self._embed_limiter = RateLimiter(cfg.rate_limits.embed_texts_per_minute)
        self._embed_token_limiter = RateLimiter(cfg.rate_limits.embed_tokens_per_minute)

    def _gen_limiter(self, model: str) -> RateLimiter:
        if model not in self._gen_limiters:
            self._gen_limiters[model] = RateLimiter(self.cfg.rate_limits.generate_per_minute)
        return self._gen_limiters[model]

    # Embed a list of texts (task_type tells Gemini if it's a document or a query)
    @_retry
    @_guard
    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        self._embed_limiter.acquire(len(texts))  # the free tier counts every text, not every request
        self._embed_token_limiter.acquire(sum(len(t) // 4 + 1 for t in texts))  # and has a tokens-per-minute cap
        res = self.client.models.embed_content(
            model=self.cfg.embedding.model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=task_type, output_dimensionality=self.cfg.embedding.dim
            ),
        )
        return [e.values for e in res.embeddings]

    def embed_documents(self, texts: list[str], batch: int = 50) -> list[list[float]]:
        # Embed in batches to stay within API limits
        out: list[list[float]] = []
        for i in range(0, len(texts), batch):
            out += self._embed(texts[i : i + batch], "RETRIEVAL_DOCUMENT")
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]

    # Plain text generation (optionally JSON output)
    @_retry
    @_guard
    def generate(self, prompt: str, model: str | None = None, json: bool = False) -> str:
        model = model or self.cfg.generation.model
        self._gen_limiter(model).acquire()
        res = self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.cfg.generation.temperature,
                response_mime_type="application/json" if json else None,
            ),
        )
        return res.text or ""

    # Several drafts of the same answer in ONE request (one unit of free-tier request quota)
    @_retry
    @_guard
    def _candidates(self, prompt: str, n: int, temperature: float) -> list[str]:
        self._gen_limiter(self.cfg.generation.model).acquire()
        res = self.client.models.generate_content(
            model=self.cfg.generation.model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=temperature, candidate_count=n),
        )
        return drafts_from(res.candidates or [], n)

    def generate_candidates(self, prompt: str, n: int, temperature: float) -> list[str]:
        if n <= 1:
            return [self.generate(prompt)]
        try:
            drafts = [d for d in self._candidates(prompt, n, temperature) if d]
            if drafts:
                return drafts
        except CandidatesUnsupported as e:
            log.warning("Model rejected candidate_count=%d (%s); using one answer", n, e)
        return [self.generate(prompt)]

    # Send a page image to Gemini vision and get a text description
    @_retry
    @_guard
    def describe_image(self, png: bytes, prompt: str) -> str:
        model = self.cfg.generation.vision_model
        self._gen_limiter(model).acquire()
        res = self.client.models.generate_content(
            model=model,
            contents=[types.Part.from_bytes(data=png, mime_type="image/png"), prompt],
            config=types.GenerateContentConfig(temperature=0.0),
        )
        return res.text or ""
