#!/usr/bin/env python3
"""CLI for the GREVIO capstone policy-brief pilot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from pilot.runner import dry_run, execute_paid_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate six independent policy briefs via OpenAI Responses API."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the six planned requests without calling the API or printing full passages.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Perform paid API requests (requires filled data/ and prompts/, and OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional existing empty run directory (default: new timestamp under runs/).",
    )
    args = parser.parse_args(argv)

    if args.dry_run and args.run:
        print("Use either --dry-run or --run, not both.", file=sys.stderr)
        return 2
    if not args.dry_run and not args.run:
        print("Specify --dry-run or --run.", file=sys.stderr)
        return 2

    load_dotenv()

    if args.dry_run:
        return dry_run()

    if args.run_dir is not None:
        args.run_dir.mkdir(parents=True, exist_ok=True)

    return execute_paid_run(run_dir=args.run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
