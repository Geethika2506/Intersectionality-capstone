#!/usr/bin/env python3
"""Human review of saved GREVIO policy briefs.

Reads response JSON already produced by run_pilot.py. Does not call the OpenAI API
and does not modify those response files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pilot.human_review import ReviewCheckError, create_templates, summary_path, write_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create human-review forms for saved briefs and summarize the labels."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    templates = subparsers.add_parser(
        "templates",
        help="Create one review form per saved response JSON file.",
    )
    summary = subparsers.add_parser(
        "summary",
        help="Check completed review forms and write a summary.",
    )
    for subparser in (templates, summary):
        subparser.add_argument(
            "--run-dir",
            type=Path,
            required=True,
            help="Run directory that contains the six response JSON files.",
        )
        subparser.add_argument(
            "--reference-dir",
            type=Path,
            default=None,
            help="Directory of coding tables (default: evaluation/).",
        )

    args = parser.parse_args(argv)
    try:
        if args.command == "templates":
            for message in create_templates(args.run_dir, args.reference_dir):
                print(message)
            return 0
        text = write_summary(args.run_dir, args.reference_dir)
        print(text, end="" if text.endswith("\n") else "\n")
        print(f"Wrote {summary_path(args.run_dir)}")
        return 0
    except ReviewCheckError as exc:
        print("Review checks failed:", file=sys.stderr)
        for error in exc.errors:
            print(f"- {error}", file=sys.stderr)
        if args.command == "summary":
            print("Summary was not written.", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
