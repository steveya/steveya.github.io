# Steve Yang Blog (Quarto)

This site is now built with [Quarto](https://quarto.org/) and deployed to GitHub Pages.

## Local preview

```bash
quarto preview
```

## Migration

Legacy Jekyll sources have been removed now that the site is fully on Quarto. To
regenerate the Quarto content from any restored legacy posts:

```bash
./scripts/migrate_posts.py --clean
```

See `migration-log.md` for files that require manual migration.
