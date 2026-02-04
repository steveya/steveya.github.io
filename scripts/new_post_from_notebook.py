#!/usr/bin/env python3
"""Create a stealth notebook draft for a new post.

Usage:
  python scripts/new_post_from_notebook.py --slug my-post --title "My Title" \
    --date 2026-02-04 --categories "Quantitative Finance, Time Series" \
    --description "One sentence" \
    [--in-posts]

Default behavior writes the notebook to drafts/<slug>.ipynb so it will never be
published. If --in-posts is passed, it writes posts/<slug>/draft.ipynb, but
.quartoignore keeps it out of renders.
"""

from __future__ import annotations

import argparse
import json
from datetime import date as date_type
from pathlib import Path


def _split_categories(raw: str) -> list[str]:
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]


def _today_iso() -> str:
    return date_type.today().isoformat()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--date", default=_today_iso())
    parser.add_argument("--description", default="")
    parser.add_argument("--categories", default="")
    parser.add_argument(
        "--in-posts",
        action="store_true",
        help="Write to posts/<slug>/draft.ipynb instead of drafts/<slug>.ipynb",
    )
    args = parser.parse_args()

    slug: str = args.slug.strip().strip("/")
    if not slug:
        raise SystemExit("--slug cannot be empty")

    categories = _split_categories(args.categories) if args.categories else []

    front_matter_lines = [
        "---",
        f'title: "{args.title}"',
        f"date: {args.date}",
    ]
    if args.description:
        front_matter_lines.append(f'description: "{args.description}"')
    if categories:
        cats = ", ".join(categories)
        front_matter_lines.append(f"categories: [{cats}]")
    front_matter_lines.extend(["", "---", ""])

    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {"language": "markdown"},
                "source": [line + "\n" for line in front_matter_lines]
                + [
                    "## Outline\n",
                    "- Motivation\n",
                    "- Data\n",
                    "- Method\n",
                    "- Results\n",
                    "- Notes / TODO\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {"language": "python"},
                "source": [
                    "# Scratchpad\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                ],
                "outputs": [],
                "execution_count": None,
            },
        ],
        "metadata": {
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    if args.in_posts:
        out_dir = Path("posts") / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "draft.ipynb"
    else:
        out_dir = Path("drafts")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{slug}.ipynb"

    out_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
