"""Log to the console and to logs/rag.log (rotating text file)."""
import logging
from logging.handlers import RotatingFileHandler

from .config import ROOT

LOG_FILE = ROOT / "logs" / "rag.log"


def setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger("rag")
    if root.handlers:  # already configured
        return
    root.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    console = logging.StreamHandler()
    for h in (file_handler, console):
        h.setFormatter(fmt)
        root.addHandler(h)
    root.propagate = False
