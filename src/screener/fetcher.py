"""Wrapper de yfinance con cache en disco.

Politicas de refresh:
- OHLCV: refresh diario (incremental cuando es posible)
- .info y financials trimestrales: 1x/semana
- insider_transactions: 1x/semana

Cache en data/screener/{ohlcv,fundamentals}/<TICKER>.{parquet,json}
"""
from __future__ import annotations

import json
import logging
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DATA_ROOT = Path("data/screener")
OHLCV_DIR = DATA_ROOT / "ohlcv"
FUND_DIR = DATA_ROOT / "fundamentals"
META_DIR = DATA_ROOT / "meta"

OHLCV_MAX_AGE_DAYS = 1
FUND_MAX_AGE_DAYS = 7
INSIDER_MAX_AGE_DAYS = 7


def _ensure_dirs() -> None:
    for d in (OHLCV_DIR, FUND_DIR, META_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Aplana MultiIndex columns y deja todo en lowercase.

    yfinance a veces devuelve columnas como ('AAPL', 'Open') (especialmente con
    group_by='ticker' o cuando hay un solo ticker en una lista). Normalizamos
    a flat ['open', 'high', 'low', 'close', 'volume'].
    """
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    df.columns = [str(c).lower() for c in df.columns]
    return df


def _meta_path(ticker: str) -> Path:
    return META_DIR / f"{ticker}.json"


def _read_meta(ticker: str) -> dict:
    p = _meta_path(ticker)
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _write_meta(ticker: str, key: str) -> None:
    meta = _read_meta(ticker)
    meta[key] = datetime.now(timezone.utc).isoformat()
    _meta_path(ticker).write_text(json.dumps(meta, indent=2))


def _is_stale(ticker: str, key: str, max_age_days: int) -> bool:
    meta = _read_meta(ticker)
    ts_str = meta.get(key)
    if not ts_str:
        return True
    ts = datetime.fromisoformat(ts_str)
    age = datetime.now(timezone.utc) - ts
    return age > timedelta(days=max_age_days)


# ----------------------------------------------------------------------
# OHLCV
# ----------------------------------------------------------------------

def fetch_ohlcv(ticker: str, period: str = "5y", refresh: bool = False) -> pd.DataFrame | None:
    """Devuelve OHLCV diario ajustado. Cache local en parquet."""
    _ensure_dirs()
    cache_path = OHLCV_DIR / f"{ticker}.parquet"

    if cache_path.exists() and not refresh and not _is_stale(ticker, "ohlcv", OHLCV_MAX_AGE_DAYS):
        return pd.read_parquet(cache_path)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    except Exception as e:
        logger.warning("OHLCV fetch failed for %s: %s", ticker, e)
        return None

    if df is None or df.empty:
        logger.warning("OHLCV empty for %s", ticker)
        return None

    df = _normalize_ohlcv(df)
    if df.empty or "close" not in df.columns:
        logger.warning("OHLCV normalized to empty/invalid for %s", ticker)
        return None
    df.to_parquet(cache_path)
    _write_meta(ticker, "ohlcv")
    return df


def fetch_ohlcv_bulk(
    tickers: list[str],
    period: str = "5y",
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Bulk download. Mucho mas rapido que ticker-por-ticker.

    Para tickers ya cacheados (no stale), los lee desde parquet sin tocar la API.
    """
    _ensure_dirs()
    out: dict[str, pd.DataFrame] = {}
    to_download: list[str] = []

    for t in tickers:
        cache_path = OHLCV_DIR / f"{t}.parquet"
        if cache_path.exists() and not refresh and not _is_stale(t, "ohlcv", OHLCV_MAX_AGE_DAYS):
            cached = pd.read_parquet(cache_path)
            if not cached.empty:
                out[t] = cached
                continue
            # cache vacio (download anterior fallido) -> re-bajar
        to_download.append(t)

    if not to_download:
        return out

    logger.info("Downloading OHLCV for %d tickers...", len(to_download))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_all = yf.download(
                to_download,
                period=period,
                interval="1d",
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False,
            )
    except Exception as e:
        logger.warning("Bulk OHLCV download failed: %s", e)
        return out

    failed_in_bulk: list[str] = []
    for t in to_download:
        try:
            if len(to_download) == 1:
                df_t = df_all
            else:
                df_t = df_all[t] if t in df_all.columns.get_level_values(0) else None
            if df_t is None or df_t.empty:
                failed_in_bulk.append(t)
                continue
            df_t = _normalize_ohlcv(df_t.dropna(how="all"))
            if df_t.empty or "close" not in df_t.columns:
                failed_in_bulk.append(t)
                continue
            cache_path = OHLCV_DIR / f"{t}.parquet"
            df_t.to_parquet(cache_path)
            _write_meta(t, "ohlcv")
            out[t] = df_t
        except Exception as e:
            logger.warning("OHLCV slice failed for %s: %s", t, e)
            failed_in_bulk.append(t)

    # Reintento individual para los que fallaron en bulk (evita el bug de
    # 'database is locked' que aparece a veces en yfinance bulk).
    for t in failed_in_bulk:
        logger.info("Retrying %s individually...", t)
        df_t = fetch_ohlcv(t, period=period, refresh=True)
        if df_t is not None and not df_t.empty:
            out[t] = df_t

    return out


# ----------------------------------------------------------------------
# Fundamentals (.info, quarterly_*)
# ----------------------------------------------------------------------

def fetch_info(ticker: str, refresh: bool = False) -> dict[str, Any] | None:
    """Snapshot de fundamentals desde .info."""
    _ensure_dirs()
    cache_path = FUND_DIR / f"{ticker}_info.json"

    if cache_path.exists() and not refresh and not _is_stale(ticker, "info", FUND_MAX_AGE_DAYS):
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            info = yf.Ticker(ticker).info
    except Exception as e:
        logger.warning(".info fetch failed for %s: %s", ticker, e)
        return None

    if not info:
        return None

    # Limpiamos campos no serializables
    clean = {k: v for k, v in info.items() if _is_json_serializable(v)}
    cache_path.write_text(json.dumps(clean, indent=2, default=str))
    _write_meta(ticker, "info")
    return clean


def fetch_financials(ticker: str, refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Devuelve dict con quarterly_income_stmt, quarterly_balance_sheet, quarterly_cashflow.

    Cada DataFrame puede estar vacio si yfinance no tiene los datos.
    """
    _ensure_dirs()
    keys = {
        "qis": "quarterly_income_stmt",
        "qbs": "quarterly_balance_sheet",
        "qcf": "quarterly_cashflow",
    }
    out: dict[str, pd.DataFrame] = {}

    needs_fetch = False
    for short_key in keys:
        cache_path = FUND_DIR / f"{ticker}_{short_key}.parquet"
        if cache_path.exists() and not refresh and not _is_stale(ticker, short_key, FUND_MAX_AGE_DAYS):
            try:
                out[short_key] = pd.read_parquet(cache_path)
                continue
            except Exception:
                pass
        needs_fetch = True
        break

    if not needs_fetch and len(out) == len(keys):
        return out

    out = {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = yf.Ticker(ticker)
            for short_key, attr in keys.items():
                try:
                    df = getattr(t, attr)
                except Exception:
                    df = pd.DataFrame()
                if df is None:
                    df = pd.DataFrame()
                if not df.empty:
                    # Yahoo usa columnas como timestamps; normalizamos a string para parquet
                    df = df.copy()
                    df.columns = [str(c) for c in df.columns]
                    cache_path = FUND_DIR / f"{ticker}_{short_key}.parquet"
                    df.to_parquet(cache_path)
                    _write_meta(ticker, short_key)
                out[short_key] = df
    except Exception as e:
        logger.warning("Financials fetch failed for %s: %s", ticker, e)
        for short_key in keys:
            out.setdefault(short_key, pd.DataFrame())

    return out


def fetch_insider(ticker: str, refresh: bool = False) -> pd.DataFrame:
    """Insider transactions. Puede estar vacio si yfinance no los tiene."""
    _ensure_dirs()
    cache_path = FUND_DIR / f"{ticker}_insider.parquet"

    if cache_path.exists() and not refresh and not _is_stale(ticker, "insider", INSIDER_MAX_AGE_DAYS):
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = yf.Ticker(ticker).insider_transactions
    except Exception as e:
        logger.warning("insider fetch failed for %s: %s", ticker, e)
        return pd.DataFrame()

    if df is None:
        df = pd.DataFrame()

    if not df.empty:
        df = df.copy()
        df.columns = [str(c) for c in df.columns]
        df.to_parquet(cache_path)
    _write_meta(ticker, "insider")
    return df


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _is_json_serializable(v: Any) -> bool:
    try:
        json.dumps(v, default=str)
        return True
    except Exception:
        return False
