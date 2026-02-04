# Agent Runbook — steveya.github.io

This repo is a Quarto website deployed as a static site to GitHub Pages.

## Key facts

- Source: Quarto `.qmd` under `index.qmd`, `pages/`, and `posts/<slug>/index.qmd`.
- Output directory: `docs/` (GitHub Pages publishes from here).
- Navbar is custom HTML injected via `_quarto.yml` → `format.html.include-before-body: _includes/nav.html`.

## Common commands

- Render the whole site: `quarto render`
- Render a single post: `quarto render posts/<slug>/index.qmd`
- Render a single page: `quarto render pages/<name>.qmd`

## “Stealth” workflow: Notebook → Post (Option A)

Goal: work privately in a notebook (not published), periodically preview as a Quarto post, and only publish when ready.

### Guardrails (do not publish drafts)

Draft notebooks are excluded from rendering/publishing via `.quartoignore`:

- `drafts/`
- `posts/**/draft.ipynb`
- `posts/**/_draft.ipynb`
- `posts/**/.ipynb_checkpoints/`

So you can safely keep drafts in the repo without them appearing under `docs/`.

### Create a new stealth notebook draft

Use the helper script:

- `python scripts/new_post_from_notebook.py --slug <slug> --title "<Title>" --categories "Cat1, Cat2" --description "One sentence"`

Default output:

- `drafts/<slug>.ipynb`

The notebook starts with a Markdown cell containing YAML front matter.

(Alternative) If you prefer the draft to live inside the post folder (still ignored by Quarto):

- `python scripts/new_post_from_notebook.py --in-posts --slug <slug> --title "<Title>" ...`

This writes:

- `posts/<slug>/draft.ipynb`

### Preview your progress as a real post

Convert + render into the standard published layout:

- `scripts/preview_notebook_post.sh drafts/<slug>.ipynb <slug>`

This will:

- Convert notebook → `posts/<slug>/index.qmd`
- Render the post → `docs/posts/<slug>/index.html`

### Publish

When the post is ready:

- Keep `posts/<slug>/index.qmd` as the published source.
- Leave the notebook in `drafts/` (or `posts/<slug>/draft.ipynb`) as a private working artifact.

## Listings note (important)

Listings/pages in this site generally target `posts/*/index.qmd`.

- If you publish notebook posts directly as `index.ipynb`, update the listing globs accordingly.
- The preferred approach here is to publish `index.qmd` and keep notebooks as drafts.

## Citations / References

- Bibliography lives at `references.bib`.
- Posts inherit bibliography via `posts/_metadata.yml` using `bibliography: ../../references.bib`.
- Use citekeys like `@taylor2004` in post body to generate a References section automatically.

## Search page (custom)

- `pages/search.qmd` is a custom listing + JS filter.
- Listing contents should use a glob: `../posts/*/index.qmd`.
- The JS waits for the Quarto listing to initialize and calls `list.search(q, SEARCH_COLUMNS)`.

## CI/CD

- Deploy workflow: `.github/workflows/pages-deploy.yml` (renders + runs smoke tests before deploy).
- PR/Push checks: `.github/workflows/ci.yml` (renders + runs smoke tests).
- Smoke tests are Playwright-based under `tests/e2e/`.

