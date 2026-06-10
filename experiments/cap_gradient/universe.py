"""
Universo del experimento: top N perpetuos USDT de Bybit rankeados por
open interest en USD (no por volumen, que está inflado por wash trading).

Limitación documentada: el snapshot es el universo ACTUAL, no point-in-time.
Perpetuos deslistados no aparecen → sesgo de supervivencia que favorece a
los pares chicos sobrevivientes.

Cada candidato se valida contra las tres fuentes de datos del motor:
  - OHLCV spot en Binance (ccxt) con >= min_history_months de historia
  - OI en Bybit (launchTime del contrato)
  - Funding en Binance Futures (onboardDate en fapi exchangeInfo)
"""

import re
import time
from datetime import datetime, timezone

import ccxt
import pandas as pd
import requests
from dateutil.relativedelta import relativedelta

from .config import UNIVERSE_PATH, ensure_dirs, exp_cfg, tier_for_rank

BYBIT_INSTRUMENTS = "https://api.bybit.com/v5/market/instruments-info"
BYBIT_TICKERS     = "https://api.bybit.com/v5/market/tickers"
BINANCE_FAPI_INFO = "https://fapi.binance.com/fapi/v1/exchangeInfo"

# Cuántos candidatos rankear como máximo para llenar el top_n tras exclusiones
MAX_SCAN = 60


def _bybit_get(url: str, params: dict) -> dict:
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if data.get("retCode") != 0:
        raise RuntimeError(f"Bybit API error: {data.get('retMsg')}")
    return data["result"]


def fetch_bybit_perps() -> pd.DataFrame:
    """Perpetuos lineales USDT activos en Bybit, con launchTime."""
    rows, cursor = [], ""
    while True:
        params = {"category": "linear", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        result = _bybit_get(BYBIT_INSTRUMENTS, params)
        rows.extend(result.get("list", []))
        cursor = result.get("nextPageCursor", "")
        if not cursor:
            break
        time.sleep(0.1)

    df = pd.DataFrame(rows)
    df = df[
        (df["contractType"] == "LinearPerpetual")
        & (df["status"] == "Trading")
        & (df["quoteCoin"] == "USDT")
    ].copy()
    df["launch_time"] = pd.to_datetime(
        pd.to_numeric(df["launchTime"]), unit="ms", utc=True
    )
    return df[["symbol", "baseCoin", "launch_time"]]


def fetch_bybit_oi_values() -> pd.DataFrame:
    """openInterestValue (USD) por símbolo desde tickers."""
    result = _bybit_get(BYBIT_TICKERS, {"category": "linear"})
    df = pd.DataFrame(result["list"])
    df["oi_usd"] = pd.to_numeric(df["openInterestValue"], errors="coerce")
    return df[["symbol", "oi_usd"]]


def _base_variants(base: str) -> list[str]:
    """Variantes del baseCoin para mapear entre exchanges.

    Bybit usa prefijo/sufijo 1000 (o 10000) para tokens de precio chico
    (1000PEPE, SHIB1000); Binance spot lista el token sin multiplicador,
    Binance futures a veces con prefijo 1000.
    """
    variants = [base]
    m = re.match(r"^(\d{3,6})(.+)$", base)
    if m and m.group(1) in {"100", "1000", "10000", "1000000"}:
        variants.append(m.group(2))
    m = re.match(r"^(.+?)(\d{3,6})$", base)
    if m and m.group(2) in {"1000", "10000"}:
        variants.append(m.group(1))
    return variants


def build_universe() -> pd.DataFrame:
    ensure_dirs()
    exp = exp_cfg()
    top_n   = exp["top_n"]
    cutoff  = datetime.now(timezone.utc) - relativedelta(months=exp["min_history_months"])
    excl    = set(exp["exclude_bases"])

    print("  [UNIVERSE] Bybit instruments + tickers...")
    perps = fetch_bybit_perps()
    ois   = fetch_bybit_oi_values()
    cand  = perps.merge(ois, on="symbol").sort_values("oi_usd", ascending=False)
    cand  = cand.head(MAX_SCAN).reset_index(drop=True)

    print("  [UNIVERSE] Binance spot markets + fapi exchangeInfo...")
    binance = ccxt.binance({"enableRateLimit": True})
    spot_markets = set(binance.load_markets().keys())

    fapi = requests.get(BINANCE_FAPI_INFO, timeout=20).json()
    fapi_info = {
        s["symbol"]: pd.to_datetime(s["onboardDate"], unit="ms", utc=True)
        for s in fapi["symbols"]
        if s.get("status") == "TRADING" and s.get("quoteAsset") == "USDT"
    }

    cutoff_ms = int(cutoff.timestamp() * 1000)
    records, n_included = [], 0

    for _, row in cand.iterrows():
        base = row["baseCoin"]
        rec = {
            "symbol_bybit": row["symbol"],
            "base":         base,
            "oi_usd":       row["oi_usd"],
            "launch_time":  row["launch_time"],
            "ccxt_symbol":  None,
            "binance_fut_symbol": None,
            "included":     False,
            "exclusion_reason": None,
        }

        def _exclude(reason: str) -> None:
            rec["exclusion_reason"] = reason
            records.append(rec)
            print(f"    - {row['symbol']:<14} EXCLUIDO: {reason}")

        if n_included >= top_n:
            rec["exclusion_reason"] = "universo completo (top_n alcanzado)"
            records.append(rec)
            continue
        if base in excl:
            _exclude("stablecoin/sintético")
            continue
        if row["launch_time"] > cutoff:
            _exclude(f"contrato Bybit listado {row['launch_time'].date()} (<18m de OI)")
            continue

        # Spot en Binance (fuente OHLCV del motor)
        spot = next(
            (f"{v}/USDT" for v in _base_variants(base) if f"{v}/USDT" in spot_markets),
            None,
        )
        if spot is None:
            _exclude("sin par spot en Binance")
            continue

        # Funding en Binance Futures
        fut = next(
            (f"{v}USDT" for v in _base_variants(base) if f"{v}USDT" in fapi_info),
            None,
        )
        if fut is None:
            _exclude("sin perpetuo USDT en Binance Futures")
            continue
        if fapi_info[fut] > cutoff:
            _exclude(f"perp Binance onboard {fapi_info[fut].date()} (<18m de funding)")
            continue

        # Historia spot real: primera vela 1h <= cutoff (+7 días de tolerancia)
        try:
            first = binance.fetch_ohlcv(spot, "1h", since=cutoff_ms, limit=1)
        except Exception as e:
            _exclude(f"error consultando historia spot: {e}")
            continue
        if not first:
            _exclude("sin velas spot en el período requerido")
            continue
        first_ts = pd.to_datetime(first[0][0], unit="ms", utc=True)
        if first_ts > pd.Timestamp(cutoff) + pd.Timedelta(days=7):
            _exclude(f"historia spot arranca {first_ts.date()} (<18m)")
            continue

        n_included += 1
        rec.update({
            "ccxt_symbol":        spot,
            "binance_fut_symbol": fut,
            "included":           True,
        })
        records.append(rec)
        print(f"    + {row['symbol']:<14} OI ${row['oi_usd']/1e6:,.0f}M  → rank {n_included}")

    df = pd.DataFrame(records)
    df["snapshot_ts"] = pd.Timestamp.now(tz="UTC")

    # Rank y tier solo sobre los incluidos (orden por OI desc ya garantizado)
    df["rank"] = pd.NA
    df.loc[df["included"], "rank"] = range(1, int(df["included"].sum()) + 1)
    tiers = df.loc[df["included"], "rank"].map(lambda r: tier_for_rank(int(r), exp))
    df["tier"]      = pd.NA
    df["cost_mult"] = pd.NA
    df.loc[df["included"], "tier"]      = tiers.map(lambda t: t[0])
    df.loc[df["included"], "cost_mult"] = tiers.map(lambda t: t[1])

    df.to_parquet(UNIVERSE_PATH)
    n_exc = (~df["included"]).sum()
    print(f"  [UNIVERSE] {n_included} incluidos / {n_exc} candidatos excluidos "
          f"→ {UNIVERSE_PATH}")
    return df


def load_universe() -> pd.DataFrame:
    if not UNIVERSE_PATH.exists():
        raise FileNotFoundError(
            f"No existe {UNIVERSE_PATH}. Corré primero: "
            "python -m experiments.cap_gradient --stage universe"
        )
    return pd.read_parquet(UNIVERSE_PATH)
