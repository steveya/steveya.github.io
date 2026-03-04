from __future__ import annotations

from typing import Any, cast

import pandas as pd
from IPython.display import HTML, display

BLOG_TABLE_STYLES: list[dict[str, Any]] = [
    {
        "selector": "table",
        "props": [
            ("border-collapse", "separate"),
            ("border-spacing", "0"),
            ("font-size", "0.95rem"),
            ("line-height", "1.4"),
        ],
    },
    {
        "selector": "thead th",
        "props": [
            ("text-align", "left"),
            ("font-weight", "600"),
            ("border-bottom", "1px solid #ddd"),
            ("padding", "8px 12px"),
        ],
    },
    {
        "selector": "tbody td",
        "props": [
            ("padding", "8px 12px"),
            ("border-bottom", "1px solid #eee"),
        ],
    },
    {
        "selector": "tbody tr:nth-child(even)",
        "props": [("background-color", "#f7f7f756")],
    },
]

BLOG_TABLE_CSS = """
<style>
.dataframe {
  border-collapse: separate !important;
  border-spacing: 0 !important;
  font-size: 0.98rem !important;
  line-height: 1.45 !important;
  margin: 0.75rem 0 1.25rem 0 !important;
  max-width: 100% !important;
  width: 100% !important;
  table-layout: auto !important;
}
.dataframe th, .dataframe td {
  padding: 0.6rem 0.9rem !important;
  text-align: right !important;
  border-bottom: 1px solid rgba(0, 0, 0, 0.08) !important;
}
.dataframe th {
  text-align: left !important;
  font-weight: 600 !important;
}
.dataframe tbody tr:nth-child(even) td,
.dataframe tbody tr:nth-child(even) th {
  background: rgba(0, 0, 0, 0.05) !important;
}
.dataframe {
  display: block !important;
  overflow-x: auto !important;
}
</style>
""".strip()


def style_results_table(
    df: pd.DataFrame,
    precision: int = 6,
    index_col: str = "variant",
) -> "pd.io.formats.style.Styler":
    out = df.set_index(index_col) if index_col in df.columns else df
    return out.style.set_table_styles(cast(Any, BLOG_TABLE_STYLES)).format(
        precision=precision
    )


def inject_blog_table_css() -> None:
    display(HTML(BLOG_TABLE_CSS))
