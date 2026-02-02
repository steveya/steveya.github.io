#!/usr/bin/env python3
"""Migrate Jekyll posts into Quarto posts structure."""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import re
import shutil
from pathlib import Path


POST_PATTERN = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})-(?P<slug>.+)\.md$")
MATHJAX_SCRIPT_RE = re.compile(
    r"^<script[^>]*mathjax[^>]*></script>$", re.IGNORECASE
)
POST_URL_RE = re.compile(r"\{\%\s*post_url\s+([^\s\%]+)\s*\%\}")


class MigrationWarning(Exception):
    """Custom warning for migration issues."""


def slug_from_filename(filename: str) -> str:
    match = POST_PATTERN.match(filename)
    if not match:
        raise MigrationWarning(f"Unrecognized post filename: {filename}")
    return match.group("slug")


def date_from_filename(filename: str) -> dt.date:
    match = POST_PATTERN.match(filename)
    if not match:
        raise MigrationWarning(f"Unrecognized post filename: {filename}")
    return dt.date.fromisoformat(match.group("date"))


def convert_post_urls(text: str, prefix: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        slug = match.group(1)
        if slug.count("-") >= 3:
            slug = "-".join(slug.split("-")[3:])
        return f"{prefix}{slug}/"

    return POST_URL_RE.sub(_replace, text)


def strip_mathjax_script(lines: list[str]) -> list[str]:
    while lines and not lines[0].strip():
        lines = lines[1:]
    if lines and MATHJAX_SCRIPT_RE.match(lines[0].strip()):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]
    return lines


def migrate_post(source: Path, dest_dir: Path) -> None:
    filename = source.name
    slug = slug_from_filename(filename)
    post_date = date_from_filename(filename)

    target_dir = dest_dir / slug
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / "index.qmd"

    content = source.read_text(encoding="utf-8")
    content = content.replace("\r\n", "\n")
    lines = content.split("\n")

    if lines and lines[0].strip() == "---":
        try:
            end_index = lines.index("---", 1)
        except ValueError as exc:
            raise MigrationWarning(f"Front matter not closed in {filename}") from exc

        front_matter = lines[: end_index + 1]
        body_lines = lines[end_index + 1 :]
    else:
        front_matter = ["---", f"title: {slug}", "---"]
        body_lines = lines

    body_lines = strip_mathjax_script(body_lines)
    body = "\n".join(body_lines).lstrip("\n")

    body = convert_post_urls(body, prefix="../")

    target_file.write_text("\n".join(front_matter) + "\n\n" + body + "\n", encoding="utf-8")

    logging.info("Migrated %s -> %s (date=%s)", source, target_file, post_date)


def migrate_tabs(source_dir: Path, dest_dir: Path) -> None:
    if not source_dir.exists():
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(source_dir.glob("*.md")):
        dest_file = dest_dir / source.name.replace(".md", ".qmd")
        content = source.read_text(encoding="utf-8").replace("\r\n", "\n")
        content = convert_post_urls(content, prefix="../posts/")
        dest_file.write_text(content + "\n", encoding="utf-8")
        logging.info("Migrated tab %s -> %s", source, dest_file)


def write_listing_pages(dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / "archives.qmd").write_text(
        "---\n"
        "title: \"Archives\"\n"
        "---\n\n"
        "```{listing}\n"
        "contents: ../posts/*/index.qmd\n"
        "sort: \"date desc\"\n"
        "```\n",
        encoding="utf-8",
    )
    (dest_dir / "categories.qmd").write_text(
        "---\n"
        "title: \"Categories\"\n"
        "---\n\n"
        "```{listing}\n"
        "contents: ../posts/*/index.qmd\n"
        "sort: \"date desc\"\n"
        "categories: true\n"
        "```\n",
        encoding="utf-8",
    )
    (dest_dir / "tags.qmd").write_text(
        "---\n"
        "title: \"Tags\"\n"
        "---\n\n"
        "```{listing}\n"
        "contents: ../posts/*/index.qmd\n"
        "sort: \"date desc\"\n"
        "tags: true\n"
        "```\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate Jekyll posts to Quarto format")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("_jekyll_legacy/_posts"),
        help="Path to the legacy Jekyll _posts directory",
    )
    parser.add_argument(
        "--tabs-source",
        type=Path,
        default=Path("_jekyll_legacy/_tabs"),
        help="Path to the legacy Jekyll _tabs directory",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("posts"),
        help="Destination posts directory",
    )
    parser.add_argument(
        "--tabs-dest",
        type=Path,
        default=Path("pages"),
        help="Destination pages directory",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete destination directories before migrating",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.clean:
        if args.dest.exists():
            shutil.rmtree(args.dest)
        if args.tabs_dest.exists():
            shutil.rmtree(args.tabs_dest)

    if not args.source.exists():
        raise SystemExit(f"Source posts directory not found: {args.source}")

    args.dest.mkdir(parents=True, exist_ok=True)
    for source in sorted(args.source.glob("*.md")):
        try:
            migrate_post(source, args.dest)
        except MigrationWarning as warning:
            logging.warning("%s", warning)

    migrate_tabs(args.tabs_source, args.tabs_dest)
    write_listing_pages(args.tabs_dest)


if __name__ == "__main__":
    main()
