"""
Descarga y QA de datos por símbolo. Reutiliza el fetcher del motor
(caché Parquet incremental en data/). La descarga es secuencial con el
throttling propio del fetcher; el paralelismo queda para el backtest.

Por símbolo se construye y cachea el frame mergeado (OHLCV + OI + funding
+ tendencia diaria) en data/experiment_cap_gradient/{BASE}.parquet, que es
lo que consume el runner en las 3 pasadas.
"""

import traceback

import pandas as pd

from src.data.fetcher import (
    fetch_funding_rate, fetch_ohlcv, fetch_open_interest, merge_derivatives,
)
from src.data.preparation import apply_daily_trend, build_daily_trend

from .config import DATA_DIR, PREPARED_DIR, QA_PATH, ensure_dirs, exp_cfg, load_cfg
from .universe import load_universe


def build_frame(
    df_1h: pd.DataFrame,
    df_1d: pd.DataFrame,
    df_oi: pd.DataFrame,
    df_fr: pd.DataFrame,
    since_date: str,
    ma_trend_1d: int,
) -> pd.DataFrame:
    """Frame de backtest: OHLCV 1h + derivados + tendencia diaria laggeada.

    Misma secuencia que main.py; factorizada para reuso en el test
    anti look-ahead (que necesita reconstruir el frame con data truncada).
    """
    daily_trend = build_daily_trend(df_1d, ma_period=ma_trend_1d)
    df_bt = df_1h[df_1h.index >= pd.Timestamp(since_date, tz="UTC")].copy()
    df_bt = merge_derivatives(df_bt, df_oi, df_fr)
    df_bt = apply_daily_trend(df_bt, daily_trend)
    return df_bt


def _qa_metrics(df_bt: pd.DataFrame, gap_alert: int) -> dict:
    """Calidad del frame mergeado: huecos OHLCV, gaps largos, cobertura OI."""
    idx = df_bt.index
    span_hours = (idx[-1] - idx[0]) / pd.Timedelta(hours=1) + 1
    missing_pct = (1 - len(idx) / span_hours) * 100

    deltas = idx.to_series().diff().dropna() / pd.Timedelta(hours=1)
    n_gaps_alert = int((deltas > gap_alert).sum())
    max_gap = float(deltas.max()) if len(deltas) else 0.0

    oi_coverage = float((df_bt["oi_usdt"] > 0).mean() * 100)
    fr_coverage = float((df_bt["funding_rate"] != 0).mean() * 100)

    return {
        "start":          idx[0],
        "end":            idx[-1],
        "n_bars":         len(idx),
        "span_months":    round(span_hours / 730, 1),
        "missing_pct":    round(missing_pct, 3),
        "n_gaps_alert":   n_gaps_alert,
        "max_gap_hours":  round(max_gap, 1),
        "oi_coverage_pct": round(oi_coverage, 1),
        "fr_coverage_pct": round(fr_coverage, 1),
    }


def prepared_path(base: str):
    return PREPARED_DIR / f"{base}.parquet"


def download_and_prepare(refresh_qa_only: bool = False) -> pd.DataFrame:
    """Descarga secuencial + QA de todos los símbolos incluidos del universo.

    Símbolos que fallan o no pasan QA quedan registrados con excluded=True;
    no tiran abajo la corrida.
    """
    ensure_dirs()
    cfg, exp = load_cfg(), exp_cfg()
    g = cfg["global"]
    ma_1d = cfg["strategy"]["liquidity_sweep"]["ma_trend_1d"]

    uni = load_universe()
    uni = uni[uni["included"]].sort_values("rank")

    qa_records = []
    for _, row in uni.iterrows():
        base = row["base"]
        print(f"\n  ── [{int(row['rank']):>2}] {base} "
              f"({row['ccxt_symbol']} | OI {row['symbol_bybit']} | FR {row['binance_fut_symbol']})")
        rec = {"base": base, "rank": int(row["rank"]), "excluded": False, "reason": None}
        try:
            df_1h = fetch_ohlcv(row["ccxt_symbol"], g["timeframe"],
                                exp["since_date"], g["exchange_id"], DATA_DIR)
            df_1d = fetch_ohlcv(row["ccxt_symbol"], g["tf_trend"],
                                g["since_date_trend"], g["exchange_id"], DATA_DIR)
            df_oi = fetch_open_interest(row["symbol_bybit"],
                                        g["since_date_deriv"], g["timeframe"], DATA_DIR)
            df_fr = fetch_funding_rate(row["binance_fut_symbol"],
                                       g["since_date_deriv"], DATA_DIR)

            df_bt = build_frame(df_1h, df_1d, df_oi, df_fr, exp["since_date"], ma_1d)
            qa = _qa_metrics(df_bt, exp["gap_alert_candles"])
            rec.update(qa)

            if qa["span_months"] < exp["min_history_months"]:
                rec.update(excluded=True,
                           reason=f"historia efectiva {qa['span_months']}m < {exp['min_history_months']}m")
            elif qa["missing_pct"] > exp["max_missing_pct"]:
                rec.update(excluded=True,
                           reason=f"OHLCV faltante {qa['missing_pct']:.2f}% > {exp['max_missing_pct']}%")
            elif (100 - qa["oi_coverage_pct"]) > exp["max_oi_missing_pct"]:
                rec.update(excluded=True,
                           reason=f"cobertura OI {qa['oi_coverage_pct']:.1f}% insuficiente")
            else:
                df_bt.to_parquet(prepared_path(base))

            status = f"EXCLUIDO: {rec['reason']}" if rec["excluded"] else "OK"
            print(f"     QA: {qa['n_bars']} velas | faltantes {qa['missing_pct']:.2f}% | "
                  f"OI {qa['oi_coverage_pct']:.1f}% | {status}")

        except Exception as e:
            rec.update(excluded=True, reason=f"error de descarga: {e}")
            print(f"     [ERROR] {base}: {e}")
            traceback.print_exc(limit=2)

        qa_records.append(rec)

    df_qa = pd.DataFrame(qa_records)
    df_qa.to_parquet(QA_PATH)
    n_ok = int((~df_qa["excluded"]).sum())
    print(f"\n  [DATA] {n_ok}/{len(df_qa)} símbolos pasaron QA → {QA_PATH}")
    return df_qa


def load_qa() -> pd.DataFrame:
    if not QA_PATH.exists():
        raise FileNotFoundError(
            f"No existe {QA_PATH}. Corré primero: "
            "python -m experiments.cap_gradient --stage data"
        )
    return pd.read_parquet(QA_PATH)
