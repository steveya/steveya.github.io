# CLAUDE.md — steveya.github.io

## Project overview

This is a **Quarto-based static website** deployed to GitHub Pages. It serves as a personal research blog focused on quantitative finance, data science, and machine learning. Content is organized as multi-part series and standalone notes.

- **Site generator**: Quarto (1.5+)
- **Output directory**: `docs/` (GitHub Pages publishes from here)
- **Languages**: Quarto Markdown (`.qmd`), Jupyter notebooks (`.ipynb`), Python, JavaScript
- **Code execution is disabled** (`execute: enabled: false` in `_quarto.yml`). Posts contain static code blocks only.

## Repository structure

```
├── _quarto.yml              # Main Quarto config (theme, navbar, format)
├── index.qmd                # Homepage
├── styles.css               # Global CSS
├── references.bib           # Shared bibliography (BibTeX)
├── posts/                   # Published posts (each in posts/<slug>/index.qmd)
│   └── _metadata.yml        # Shared post metadata (bibliography path)
├── pages/                   # Navigation pages (series, notes, projects, categories, about, search)
├── scripts/                 # Python build utilities
│   ├── generate_indexes.py  # Pre-render: generates pages/ from post metadata
│   ├── new_post_from_notebook.py  # Creates draft notebooks
│   └── preview_notebook_post.sh   # Previews notebook as rendered post
├── blog_styles/             # Python plotting/table style library
├── data/                    # YAML data files (projects.yml)
├── assets/                  # Static images and media
├── _includes/               # HTML templates (nav.html)
├── drafts/                  # Private notebook drafts (ignored by Quarto)
├── tests/e2e/               # Playwright smoke tests
├── docs/                    # Rendered output (committed for GitHub Pages)
└── .github/workflows/       # CI and deploy pipelines
```

## Common commands

```bash
# Full site render (also runs generate_indexes.py as pre-render step)
quarto render

# Render a single post
quarto render posts/<slug>/index.qmd

# Render a single page
quarto render pages/<name>.qmd

# Live preview with hot reload
quarto preview

# Run E2E smoke tests (requires Playwright installed)
npm test

# Serve rendered docs locally
npm run serve:docs
```

## Build and test

**Build**: `npm run build` (alias for `quarto render`). The pre-render hook automatically runs `python3 scripts/generate_indexes.py` to regenerate index pages from post metadata.

**Test**: `npm test` runs Playwright E2E tests against the rendered `docs/` directory. Tests cover search filtering and bibliography rendering.

**CI**: On every PR and push to main, GitHub Actions (`.github/workflows/ci.yml`):
1. Renders the site with `quarto render` and **fails on any WARN-level messages**
2. Runs Playwright smoke tests

Always ensure `quarto render` produces zero warnings before pushing.

## Creating new posts

### Post structure

Each post lives in `posts/<slug>/index.qmd` with YAML frontmatter:

```yaml
---
title: "Post Title"
date: YYYY-MM-DD
description: "One sentence summary"
categories: [Category1, Category2]
content-type: "series"    # or "note"
series-id: "series-slug"  # if series
series-title: "Series Name"
series-part: 1
project-ids: [project-id] # optional
---
```

### Stealth notebook workflow

Draft privately, publish when ready:

```bash
# Create a draft notebook
python scripts/new_post_from_notebook.py --slug <slug> --title "<Title>" \
  --categories "Cat1, Cat2" --description "One sentence"

# Preview as rendered post
scripts/preview_notebook_post.sh drafts/<slug>.ipynb <slug>

# Publish: keep posts/<slug>/index.qmd, leave notebook in drafts/
```

Draft files are excluded via `.quartoignore`: `drafts/`, `posts/**/draft.ipynb`, `posts/**/_draft.ipynb`.

### Index generation

`scripts/generate_indexes.py` scans all `posts/*/index.qmd` files and auto-generates:
- `pages/series.qmd` — posts grouped by series
- `pages/notes.qmd` — standalone notes
- `pages/projects.qmd` — from `data/projects.yml`
- `pages/categories.qmd` — category listings

This runs automatically as a Quarto pre-render hook. Do not edit generated pages manually.

## Citations

- Bibliography: `references.bib` (BibTeX format, shared across all posts)
- Posts inherit bibliography via `posts/_metadata.yml`
- Use citekeys like `@taylor2004` in post body; a References section is generated automatically

## Styling

- **Theme**: Custom CSS only (`theme: none`), Charter/Georgia serif fonts
- **Color palette**: Primary `#8B4513` (saddle brown), background `#FDFCF9` (off-white), text `#292524`
- **Plot styles**: Use `blog_styles` Python package for consistent Matplotlib/table styling
- **Math**: KaTeX rendering

## Writing and prose requirements

These rules apply to all post content. Follow them strictly.

- **Tone**: Professional, academic. Precise definitions and careful caveats.
- **Structure**: Prefer paragraphs over bullet points. Bullets only when absolutely necessary.
- **Section headers**: Short, descriptive. Never phrased as questions.
- **Math notation**: Never use `:=`. Use `=` with phrases like "where" or "defined as".
- **LaTeX equations**: Display equations must NOT end with punctuation. Inline math should also avoid trailing punctuation.
- **Notation**: Include a short "Notation" paragraph early in the post (no bullet list).
- **Derivations**: Show mathematical derivations when they clarify the method. Do not skip steps.
- **Language**: No casual language, jokes, or informal asides.

## Key conventions

- Post slugs use kebab-case with numeric suffixes for series parts (e.g., `volatility-forecasts-1`)
- Listings target `posts/*/index.qmd` — publish as `.qmd`, not `.ipynb`
- The navbar is injected via `_includes/nav.html` (referenced in `_quarto.yml`)
- The search page (`pages/search.qmd`) uses custom JS filtering over Quarto listings
- The `docs/` directory is committed to the repo for GitHub Pages deployment
