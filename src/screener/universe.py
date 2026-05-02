"""Carga de universos de tickers.

Por defecto S&P 500 desde Wikipedia. Cache local en data/universe_*.csv.
"""
from __future__ import annotations

from io import StringIO
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import requests

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
DEFAULT_CACHE_DIR = Path("data")
USER_AGENT = "quant-engine/0.1 (educational research)"


def load_sp500(
    cache_path: str | Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Carga la lista del S&P 500 con cache local.

    Returns DataFrame con columnas:
        Symbol, Security, GICS_Sector, GICS_Sub_Industry
    """
    cache_path = Path(cache_path) if cache_path else DEFAULT_CACHE_DIR / "universe_sp500.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path)

    resp = requests.get(SP500_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    df = tables[0].copy()

    # Wikipedia a veces cambia los nombres de columnas; normalizamos.
    rename_map = {
        "Symbol": "Symbol",
        "Security": "Security",
        "GICS Sector": "GICS_Sector",
        "GICS Sub-Industry": "GICS_Sub_Industry",
    }
    df = df.rename(columns=rename_map)
    keep = ["Symbol", "Security", "GICS_Sector", "GICS_Sub_Industry"]
    df = df[[c for c in keep if c in df.columns]]

    # Yahoo usa '-' en lugar de '.' (BRK.B -> BRK-B)
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)

    df.to_csv(cache_path, index=False)
    _write_meta(cache_path)
    return df


def load_custom(tickers: list[str]) -> pd.DataFrame:
    """Universo manual desde una lista de tickers."""
    return pd.DataFrame({
        "Symbol": tickers,
        "Security": tickers,
        "GICS_Sector": ["Unknown"] * len(tickers),
        "GICS_Sub_Industry": ["Unknown"] * len(tickers),
    })


def _write_meta(cache_path: Path) -> None:
    meta_path = cache_path.with_suffix(".meta.txt")
    meta_path.write_text(f"refreshed: {datetime.now(timezone.utc).isoformat()}\n")
