"""Usage:
    python scripts/cli.py index
    python scripts/cli.py ask "What is the leave policy?" [--source file.pdf]
    python scripts/cli.py eval eval/questions.jsonl [--no-answer] [--out eval/report.json]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.evaluate import evaluate, load_questions  # noqa: E402
from rag.logging_setup import setup_logging  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index")
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--source")
    ev = sub.add_parser("eval")
    ev.add_argument("questions", type=Path)
    ev.add_argument("--no-answer", action="store_true", help="retrieval + refusal only, no Gemini answers")
    ev.add_argument("--out", type=Path, help="write the full report as JSON")
    args = ap.parse_args()

    setup_logging()  # also writes logs/rag.log
    rag = RAGPipeline()
    if args.cmd == "index":
        print(f"Indexed {rag.index()} chunks")
    elif args.cmd == "ask":
        ans = rag.ask(args.question, source=args.source)
        print(ans.text, "\n\nSources:")
        for s in ans.sources:
            print(f"  [{s['n']}] {s['source']} > {s['section']} {s['pages']} ({s['type']})")
    else:
        results, summary = evaluate(rag, load_questions(args.questions), generate=not args.no_answer)
        print(json.dumps(summary, indent=2))
        if args.out:
            args.out.write_text(json.dumps({"summary": summary, "results": results}, indent=2))


if __name__ == "__main__":
    main()
