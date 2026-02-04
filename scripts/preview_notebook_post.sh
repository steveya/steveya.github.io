#!/usr/bin/env bash
set -euo pipefail

# Preview a stealth notebook as a Quarto post.
#
# Usage:
#   scripts/preview_notebook_post.sh drafts/my-slug.ipynb my-slug
#   scripts/preview_notebook_post.sh posts/my-slug/draft.ipynb my-slug

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <path-to-notebook.ipynb> <slug>" >&2
  exit 2
fi

NOTEBOOK_PATH="$1"
SLUG="$2"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="posts/$SLUG"
mkdir -p "$OUT_DIR"

quarto convert "$NOTEBOOK_PATH" --output "$OUT_DIR/index.qmd"
quarto render "$OUT_DIR/index.qmd"

echo "Preview built: docs/posts/$SLUG/index.html"
