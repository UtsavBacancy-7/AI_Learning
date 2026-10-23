import types

from rag.gemini import Gemini, RateLimiter, _retry_wait


class Clock:
    """A fake clock: sleeping just moves time forward."""

    def __init__(self):
        self.t, self.sleeps = 0.0, []

    def now(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def test_limiter_lets_calls_through_until_the_limit_then_waits_for_the_window():
    c = Clock()
    lim = RateLimiter(5, window=60, clock=c.now, sleep=c.sleep)
    for _ in range(5):
        lim.acquire()
    assert c.sleeps == []  # 5 calls fit in the first minute

    lim.acquire()  # the 6th has to wait until the first call is a minute old
    assert c.sleeps and abs(sum(c.sleeps) - 60) < 0.5
    assert c.t >= 60


def test_limiter_counts_units_and_never_exceeds_the_limit():
    c = Clock()
    lim = RateLimiter(100, window=60, clock=c.now, sleep=c.sleep)
    times = []
    for _ in range(6):  # six batches of 50 texts = 300 texts: needs at least 2 minutes at 100 per minute
        lim.acquire(50)
        times.append(c.t)
    assert times[:2] == [0.0, 0.0]  # the first 100 texts go straight through
    assert times[-1] >= 120
    for i, t in enumerate(times):  # in any 60 second window at most 100 texts were used
        in_window = [x for x in times if t - 60 < x <= t]
        assert 50 * len(in_window) <= 100, (i, times)


def test_no_limit_and_oversized_calls():
    c = Clock()
    RateLimiter(0, clock=c.now, sleep=c.sleep).acquire(1000)  # 0 = unlimited
    lim = RateLimiter(10, clock=c.now, sleep=c.sleep)
    lim.acquire(50)  # bigger than the limit: still goes through instead of waiting forever
    assert c.sleeps == []


def _state(message, attempt=1):
    exc = RuntimeError(message)
    return types.SimpleNamespace(outcome=types.SimpleNamespace(exception=lambda: exc), attempt_number=attempt)


def test_retry_waits_as_long_as_gemini_asks():
    assert abs(_retry_wait(_state("429 ... Please retry in 47.9168s.")) - 48.9168) < 0.01
    assert abs(_retry_wait(_state("Please retry in 34.27s.")) - 35.27) < 0.01
    assert abs(_retry_wait(_state("Please retry in 897.69ms.")) - 1.8977) < 0.01  # milliseconds
    assert abs(_retry_wait(_state("... 'retryDelay': '46s'}]}}")) - 47.0) < 0.01
    assert _retry_wait(_state("Please retry in 5000s.")) == 90.0  # capped
    assert _retry_wait(_state("some other error", attempt=3)) == 8.0  # plain back-off


def test_gemini_calls_acquire_before_every_api_request():
    calls = []

    class Spy:
        def __init__(self, name):
            self.name = name

        def acquire(self, units=1):
            calls.append((self.name, units))

    g = object.__new__(Gemini)
    g.cfg = types.SimpleNamespace(
        embedding=types.SimpleNamespace(model="m", dim=4),
        generation=types.SimpleNamespace(model="gen", vision_model="gen", temperature=0.0),
    )
    g._embed_limiter, g._embed_token_limiter = Spy("embed"), Spy("tokens")
    g._gen_limiters = {"gen": Spy("gen")}
    g.client = types.SimpleNamespace(models=types.SimpleNamespace(
        embed_content=lambda **k: types.SimpleNamespace(embeddings=[types.SimpleNamespace(values=[0.0] * 4)] * len(k["contents"])),
        generate_content=lambda **k: types.SimpleNamespace(text="ok", candidates=[]),
    ))
    assert len(g.embed_documents(["a"] * 120)) == 120  # batches of 50, 50, 20 texts
    g.generate("hello")
    g.describe_image(b"png", "describe")
    texts = [c for c in calls if c[0] == "embed"]
    tokens = [c for c in calls if c[0] == "tokens"]
    assert texts == [("embed", 50), ("embed", 50), ("embed", 20)]
    assert [u for _, u in tokens] == [50, 50, 20]  # a one-letter text counts as 1 token
    assert [c for c in calls if c[0] == "gen"] == [("gen", 1), ("gen", 1)]
