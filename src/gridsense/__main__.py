"""`python -m gridsense compare <baseline> <candidate>` (S7). Other commands live in their own modules (see rag-for-dropout/IMPLEMENTATION.md)."""
from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "compare":
        from gridsense.eval.compare import main as compare_main
        compare_main(sys.argv[2:])
        return
    raise SystemExit("usage: python -m gridsense compare <baseline snapshot.json> <candidate snapshot.json>")


if __name__ == "__main__":
    main()
