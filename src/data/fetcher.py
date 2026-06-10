"""
Data fetching module.
Handles OHLCV (Binance via ccxt), Open Interest (Bybit v5 REST),
and Funding Rate (Binance Futures REST). All with Parquet cache.
"""

import time
import ccxt
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone


def _ts_to_ms(date_str: str) -> int:
    """Convert 'YYYY-MM-DD' string to UTC milliseconds timestamp."""
    return int(
        datetime.strptime(date_str, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp() * 1000
    )


def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    since_date: str,
    exchange_id: str,
    cache_dir: Path,
) -> pd.DataFrame:
    """
    Download OHLCV data with incremental Parquet cache.

    Parameters
    ----------
    symbol      : ccxt format, e.g. "BTC/USDT"
    timeframe   : "1h", "4h", "1d", etc.
    since_date  : "YYYY-MM-DD" — only used on first download
    exchange_id : ccxt exchange id, e.g. "binance"
    cache_dir   : Path to cache directory
    """
    cache_file = cache_dir / f"{symbol.replace('/', '_')}_{timeframe}.parquet"

    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})

    if cache_file.exists():
        df_cached = pd.read_parquet(cache_file)
        last_ts   = df_cached.index[-1]
        since_ts  = int(last_ts.timestamp() * 1000) + 1
        print(f"  [CACHE] {symbol} {timeframe}: {len(df_cached)} candles "
              f"(up to {last_ts.date()})")
    else:
        df_cached = None
        since_ts  = _ts_to_ms(since_date)
        print(f"  [FETCH] {symbol} {timeframe}: downloading from {since_date}...")

    all_ohlcv = []
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since_ts, limit=1000)
        if not batch:
            break
        all_ohlcv.extend(batch)
        since_ts = batch[-1][0] + 1
        if len(batch) < 1000:
            break

    if all_ohlcv:
        df_new = pd.DataFrame(
            all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], unit="ms", utc=True)
        df_new.set_index("timestamp", inplace=True)
        print(f"  [FETCH] {symbol} {timeframe}: {len(df_new)} new candles.")
    else:
        df_new = None
        print(f"  [CACHE] {symbol} {timeframe}: up to date.")

    frames = [f for f in [df_cached, df_new] if f is not None]
    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_parquet(cache_file)
    return df


def fetch_open_interest(
    symbol: str,
    since_date: str,
    timeframe: str,
    cache_dir: Path,
) -> pd.DataFrame:
    """
    Download Open Interest history from Bybit REST API v5.

    No API key required. Full history available.
    Returns columns: oi_coins (BTC), oi_usdt (set to oi_coins,
    multiplied by close price in merge step).

    Parameters
    ----------
    symbol    : Bybit format, e.g. "BTCUSDT"
    since_date: "YYYY-MM-DD"
    timeframe : "1h" | "4h" | "1d"
    cache_dir : Path to cache directory
    """
    cache_file = cache_dir / f"oi_{symbol}_{timeframe}.parquet"
    BASE_URL   = "https://api.bybit.com/v5/market/open-interest"

    tf_ms = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
    if timeframe not in tf_ms:
        raise ValueError(f"Unsupported timeframe for OI: {timeframe}")

    interval  = tf_ms[timeframe]
    block_ms  = 200 * interval
    now_ts    = int(datetime.now(timezone.utc).timestamp() * 1000)

    if cache_file.exists():
        df_cached  = pd.read_parquet(cache_file)
        last_ts    = df_cached.index[-1]
        fetch_from = int(last_ts.timestamp() * 1000) + interval
        print(f"  [CACHE] OI {symbol}: {len(df_cached)} rows (up to {last_ts.date()})")
    else:
        df_cached  = None
        fetch_from = _ts_to_ms(since_date)
        print(f"  [FETCH] OI {symbol}: downloading from {since_date} (Bybit v5)...")

    all_rows = []
    cursor   = fetch_from

    while cursor < now_ts:
        end_ts = min(cursor + block_ms - interval, now_ts)
        params = {
            "category":    "linear",
            "symbol":      symbol,
            "intervalTime": timeframe,
            "startTime":   cursor,
            "endTime":     end_ts,
            "limit":       200,
        }
        try:
            resp = requests.get(BASE_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  [ERROR] OI {symbol}: {e}")
            break

        if data.get("retCode") != 0:
            print(f"  [ERROR] OI {symbol} API: {data.get('retMsg', 'unknown')}")
            break

        rows = data.get("result", {}).get("list", [])
        if rows:
            all_rows.extend(rows)
        cursor = end_ts + interval
        time.sleep(0.05)

    if not all_rows:
        if df_cached is not None:
            print(f"  [CACHE] OI {symbol}: no new data.")
            return df_cached
        raise RuntimeError(f"Could not download OI for {symbol}.")

    df_new = pd.DataFrame(all_rows).rename(columns={
        "openInterest": "oi_coins",
        "timestamp":    "timestamp",
    })
    df_new["timestamp"] = pd.to_datetime(
        pd.to_numeric(df_new["timestamp"]), unit="ms", utc=True
    )
    df_new.set_index("timestamp", inplace=True)
    df_new["oi_coins"] = pd.to_numeric(df_new["oi_coins"], errors="coerce")
    df_new["oi_usdt"]  = df_new["oi_coins"]   # multiplied by close in merge

    print(f"  [FETCH] OI {symbol}: {len(df_new)} new rows.")

    frames = [f for f in [df_cached, df_new] if f is not None]
    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    # Normalize to ns for consistent merge with OHLCV
    df.index = df.index.astype("datetime64[ns, UTC]")
    df.to_parquet(cache_file)
    return df


def fetch_funding_rate(
    symbol: str,
    since_date: str,
    cache_dir: Path,
) -> pd.DataFrame:
    """
    Download funding rate history from Binance Futures REST.

    No API key required. Full history since 2019.
    Frequency: every 8 hours. Forward-filled to 1H in merge step.

    Parameters
    ----------
    symbol    : Binance format, e.g. "BTCUSDT"
    since_date: "YYYY-MM-DD"
    cache_dir : Path to cache directory
    """
    cache_file = cache_dir / f"funding_{symbol}.parquet"
    BASE_URL   = "https://fapi.binance.com/fapi/v1/fundingRate"
    now_ts     = int(datetime.now(timezone.utc).timestamp() * 1000)

    if cache_file.exists():
        df_cached  = pd.read_parquet(cache_file)
        last_ts    = df_cached.index[-1]
        fetch_from = int(last_ts.timestamp() * 1000) + 1
        print(f"  [CACHE] FR {symbol}: {len(df_cached)} rows (up to {last_ts.date()})")
    else:
        df_cached  = None
        fetch_from = _ts_to_ms(since_date)
        print(f"  [FETCH] FR {symbol}: downloading from {since_date}...")

    all_rows = []
    cursor   = fetch_from

    while cursor < now_ts:
        params = {"symbol": symbol, "startTime": cursor, "limit": 1000}
        try:
            resp = requests.get(BASE_URL, params=params, timeout=15)
            resp.raise_for_status()
            rows = resp.json()
        except Exception as e:
            print(f"  [ERROR] FR {symbol}: {e}")
            break

        if not rows or not isinstance(rows, list):
            break

        all_rows.extend(rows)
        cursor = rows[-1]["fundingTime"] + 1
        if len(rows) < 1000:
            break
        time.sleep(0.1)

    if not all_rows:
        if df_cached is not None:
            print(f"  [CACHE] FR {symbol}: no new data.")
            return df_cached
        raise RuntimeError(f"Could not download funding rate for {symbol}.")

    df_new = pd.DataFrame(all_rows)
    df_new["timestamp"] = pd.to_datetime(df_new["fundingTime"], unit="ms", utc=True)
    df_new.set_index("timestamp", inplace=True)
    df_new = df_new[["fundingRate"]].rename(
        columns={"fundingRate": "funding_rate"}
    ).astype(float)

    print(f"  [FETCH] FR {symbol}: {len(df_new)} new rows.")

    frames = [f for f in [df_cached, df_new] if f is not None]
    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_parquet(cache_file)
    return df


def merge_derivatives(
    df_ohlcv: pd.DataFrame,
    df_oi: pd.DataFrame,
    df_fr: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge OI and funding rate into OHLCV DataFrame.

    Uses merge_asof for tolerance on timestamp misalignment between
    exchanges. oi_usdt = oi_coins × close (Bybit returns contracts only).
    Funding rate is forward-filled from 8H to 1H frequency.
    """
    df = df_ohlcv.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)

    oi = df_oi[["oi_coins"]].copy()
    if not isinstance(oi.index, pd.DatetimeIndex):
        oi.index = pd.to_datetime(oi.index, utc=True)

    df_reset = df.reset_index()
    oi_reset = oi.reset_index()

    ts_col_df = df_reset.columns[0]
    ts_col_oi = oi_reset.columns[0]

    df_reset[ts_col_df] = df_reset[ts_col_df].astype("datetime64[us, UTC]")
    oi_reset[ts_col_oi] = oi_reset[ts_col_oi].astype("datetime64[us, UTC]")

    merged = pd.merge_asof(
        df_reset.sort_values(ts_col_df),
        oi_reset.sort_values(ts_col_oi),
        left_on   = ts_col_df,
        right_on  = ts_col_oi,
        direction = "backward",
        tolerance = pd.Timedelta("90min"),
    )
    merged.set_index(ts_col_df, inplace=True)
    df = merged

    df["oi_coins"] = df["oi_coins"].ffill().fillna(0)
    df["oi_usdt"]  = df["oi_coins"] * df["close"]

    fr = df_fr[["funding_rate"]].copy()
    if not isinstance(fr.index, pd.DatetimeIndex):
        fr.index = pd.to_datetime(fr.index, utc=True)
    fr_1h = fr["funding_rate"].resample("1h").last().ffill()
    df = df.join(fr_1h, how="left")
    df["funding_rate"] = df["funding_rate"].fillna(0)

    return df
