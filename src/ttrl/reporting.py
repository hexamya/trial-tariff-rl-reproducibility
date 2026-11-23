"""Small helpers for writing markdown reports without extra dependencies."""
from __future__ import annotations

import pandas as pd


def md_table(df: pd.DataFrame, floatfmt: str = "{:.2f}", index: bool = False) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table."""
    if index:
        df = df.reset_index()
    cols = [str(c) for c in df.columns]

    def fmt(v) -> str:
        if isinstance(v, float):
            return "" if pd.isna(v) else floatfmt.format(v)
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def ci(est: float, lo: float, hi: float, fmt: str = "{:+.1f}") -> str:
    """Format an estimate with its interval, e.g. '-5.8 [-7.2, -4.2]'."""
    return f"{fmt.format(est)} [{fmt.format(lo)}, {fmt.format(hi)}]"
