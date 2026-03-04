# blog_styles

Reusable notebook styling helpers for posts in this repository.

## Usage

```python
from blog_styles import style_results_table, inject_blog_table_css, apply_blog_plot_style

inject_blog_table_css()
apply_blog_plot_style()
display(style_results_table(results_df, precision=4))
```

## What's included

- `style_results_table(df, precision=6, index_col="variant")`
- `inject_blog_table_css()`
- `apply_blog_plot_style()`
- `blog_plot_context()`
