"""Reporting: console (tabulate), CSV, markdown."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
from tabulate import tabulate


CONSOLE_COLS = [
    "symbol", "sector", "score_total", "score_quality", "score_valuation",
    "score_accumulation", "score_momentum", "f_score", "drawdown_ath",
    "pe", "divergence", "mom_12_1", "dist_sma200",
]


def _fmt_pct(v):
    if v is None or pd.isna(v):
        return "-"
    return f"{v*100:+.1f}%"


def _fmt_num(v, digits=1):
    if v is None or pd.isna(v):
        return "-"
    return f"{v:.{digits}f}"


def _format_for_console(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["score_total", "score_quality", "score_valuation",
              "score_accumulation", "score_momentum"]:
        if c in out.columns:
            out[c] = out[c].apply(lambda v: _fmt_num(v, 1))
    if "f_score" in out.columns:
        out["f_score"] = out["f_score"].apply(lambda v: "-" if pd.isna(v) else f"{int(v)}/9")
    for c in ["drawdown_ath", "mom_12_1", "dist_sma200"]:
        if c in out.columns:
            out[c] = out[c].apply(_fmt_pct)
    if "pe" in out.columns:
        out["pe"] = out["pe"].apply(lambda v: _fmt_num(v, 1))
    return out


def to_console(df: pd.DataFrame, top_n: int = 30) -> None:
    if df.empty:
        print("(empty result)")
        return

    processed = df[df["skipped"].isna()].head(top_n)
    if processed.empty:
        print("No tickers passed the filters.")
        return

    cols = [c for c in CONSOLE_COLS if c in processed.columns]
    formatted = _format_for_console(processed[cols])
    print(tabulate(formatted, headers="keys", tablefmt="github", showindex=False))

    skipped_count = df["skipped"].notna().sum()
    total = len(df)
    print(f"\nProcessed: {total - skipped_count}/{total}    Skipped: {skipped_count}")


def to_csv(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def to_markdown(df: pd.DataFrame, path: str | Path, top_n: int = 30) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    processed = df[df["skipped"].isna()].head(top_n)
    cols = [c for c in CONSOLE_COLS if c in processed.columns]
    formatted = _format_for_console(processed[cols])

    md = []
    md.append(f"# Screener results — top {top_n}\n")
    md.append(tabulate(formatted, headers="keys", tablefmt="github", showindex=False))
    md.append("")
    skipped_count = df["skipped"].notna().sum()
    total = len(df)
    md.append(f"\nProcessed: {total - skipped_count}/{total}    Skipped: {skipped_count}\n")

    Path(path).write_text("\n".join(md), encoding="utf-8")
