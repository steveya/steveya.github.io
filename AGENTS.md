# Agent Runbook - steveya.github.io

## Purpose

This repo is a Quarto website deployed as a static site to GitHub Pages. It owns published posts, draft notebooks, narrative analysis, and publication-ready figures and tables. When a task touches a numbered family of posts, treat that family as one long-running workflow rather than as isolated posts.

## Scope

This file applies to the whole repo. Deeper `AGENTS.md` files override it for specific series or post areas.

## Repo Facts

- Source files live under `index.qmd`, `pages/`, and `posts/<slug>/index.qmd`.
- Output directory is `docs/`, which GitHub Pages publishes.
- Navbar HTML is injected through `_quarto.yml` with `_includes/nav.html`.
- Listings generally target `posts/*/index.qmd`.
- Bibliography lives at `references.bib`.
- Posts inherit the bibliography from `posts/_metadata.yml` using `bibliography: ../../references.bib`.
- Search UI lives in `pages/search.qmd` and expects listing contents from `../posts/*/index.qmd`.
- Deploy workflow is `.github/workflows/pages-deploy.yml`.
- PR and push checks are in `.github/workflows/ci.yml`.
- Smoke tests are Playwright-based under `tests/e2e/`.

## Common Commands

- Render the whole site: `quarto render`
- Render a single post: `quarto render posts/<slug>/index.qmd`
- Render a single page: `quarto render pages/<name>.qmd`

## Repo-Level Publishing Rules

- Preserve the existing site structure, front matter conventions, and content style.
- Prefer post-local changes inside a single `posts/<slug>/` directory unless a shared include, listing, or metadata file truly needs to change.
- Minimize disruption to unrelated posts, listings, navigation, and generated output.
- Keep figures and tables publication-quality:
  - axes labeled, units explicit, captions informative, and colors readable in print and on screen
  - table columns ordered intentionally and rounded consistently
- For cross-repo research work, reference outputs from sibling code repos without copying canonical source code into this repo.
- Keep the published source as Quarto where possible and keep notebooks as drafts or working artifacts.

## Series-Level Workflow Rules

- If a task touches a numbered series, load the matching shared workflow anchor at `posts/<series-stem>-workflow/AGENTS.md`.
- Treat the series as the primary unit of continuity:
  - locate the current frontier
  - identify the latest baseline and open branch
  - keep continuity with prior claims and prior negative results
- Use `../codex-workflows/<series-stem>-log.md` as the canonical series log.
- Define "incremental improvement" using the series-specific workflow file rather than assuming all series optimize the same benchmark or model family.
- If draft part labels drift from folder numbering, follow the series workflow file and current frontier rather than trusting numbering alone.

## Post-Level Execution Artifacts

- Draft notebooks may live in `drafts/` or `posts/<slug>/draft.ipynb`.
- Post-local execution artifacts such as figures, tables, CSV summaries, or notebook helpers should remain inside the relevant post directory when possible.
- Published source should remain in `posts/<slug>/index.qmd` unless the repo already uses another pattern intentionally.
- Generated HTML belongs under `docs/` and should not be hand-edited unless the task is explicitly about generated output.

## Stealth Workflow

Goal: work privately in a notebook, preview as a Quarto post when needed, and publish only when ready.

Draft notebooks are excluded from rendering and publishing via `.quartoignore`:

- `drafts/`
- `posts/**/draft.ipynb`
- `posts/**/_draft.ipynb`
- `posts/**/.ipynb_checkpoints/`

Create a new draft notebook with:

- `python scripts/new_post_from_notebook.py --slug <slug> --title "<Title>" --categories "Cat1, Cat2" --description "One sentence"`

Default output is `drafts/<slug>.ipynb`.

If you prefer the draft inside the post folder, use:

- `python scripts/new_post_from_notebook.py --in-posts --slug <slug> --title "<Title>" ...`

That writes `posts/<slug>/draft.ipynb`.

Preview a draft notebook as a post with:

- `scripts/preview_notebook_post.sh drafts/<slug>.ipynb <slug>`

This converts the notebook to `posts/<slug>/index.qmd` and renders `docs/posts/<slug>/index.html`.

When the post is ready:

- keep `posts/<slug>/index.qmd` as the published source
- keep the notebook in `drafts/` or `posts/<slug>/draft.ipynb` as the private working artifact

## What not to do

- Do not copy canonical package source or large experiment drivers from sibling code repos into this repo.
- Do not hand-edit generated HTML under `docs/` unless the task is explicitly about generated output.
- Do not refactor unrelated posts just because they are nearby in the filesystem.
- Do not accidentally publish drafts by moving notebook files out of the ignored draft paths without intent.
- Do not lower the prose standard for convenience.

## Validation

- Render the affected post with `quarto render posts/<slug>/index.qmd` whenever possible.
- Render the full site only when shared layout or global metadata changes make that necessary.
- For notebook-first posts, use `scripts/preview_notebook_post.sh` before treating the change as publication-ready.
- Verify citations resolve, listings still point to the right files, and figures and tables render cleanly.
- If shared UI or navigation changed, consider the existing Playwright smoke tests under `tests/e2e/`.

## Deliverables

- Post-local source changes whenever feasible.
- Publication-ready figures and tables with captions or surrounding explanatory text.
- An updated series log entry when the task advances a numbered series workflow.
- A clear note of the upstream result source when the content depends on a sibling code repo.
- Render status: completed, skipped, or blocked.

## Prose and Writing Requirements

- Tone and vocabulary must be professional and academic, with precise definitions and careful caveats.
- Avoid bullet points except when absolutely necessary; prefer paragraphs.
- Section headers must be short and must not be phrased as questions.
- Avoid the symbol `:=` entirely. Use `=` with a short phrase like "where" or "defined as".
- LaTeX display equations should not end with punctuation.
- Inline math should also avoid trailing punctuation when possible.
- Prefer consistent notation, and include a short "Notation" paragraph early rather than a notation bullet list.
- Avoid casual language, jokes, and informal asides.
- Mathematical explanations should be comprehensive. Do not avoid derivations when they clarify the method.
