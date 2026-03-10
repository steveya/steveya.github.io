# Steve Yang Blog (Quarto)

This site is now built with [Quarto](https://quarto.org/) and deployed to GitHub Pages.

## Local preview

```bash
python scripts/generate_indexes.py
quarto preview
```

The generated navigation pages live in `pages/series.qmd`, `pages/notes.qmd`,
`pages/projects.qmd`, and `pages/categories.qmd`. They are built from post
frontmatter plus `data/projects.yml`.

## Migration

Legacy Jekyll sources have been removed now that the site is fully on Quarto. To
regenerate the Quarto content from any restored legacy posts:

```bash
./scripts/migrate_posts.py --clean
```

See `migration-log.md` for files that require manual migration.
