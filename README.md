# Steve Yang Blog (Quarto)

This site is now built with [Quarto](https://quarto.org/) and deployed to GitHub Pages.

## Local preview

```bash
quarto preview
```

## Migration

Legacy Jekyll sources are preserved in `_jekyll_legacy/`. To regenerate the Quarto
content from the legacy posts:

```bash
./scripts/migrate_posts.py --clean
```

See `migration-log.md` for files that require manual migration.
