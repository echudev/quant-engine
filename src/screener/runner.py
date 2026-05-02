"""Orquesta la corrida end-to-end del screener.

Pipeline:
  1. Cargar universo
  2. Bulk download de OHLCV
  3. Por cada ticker (en paralelo): fetch info + financials, aplicar filtros,
     calcular scores, computar composite
  4. Retornar DataFrame ordenado por score compuesto descendente
"""
from __future__ import annotations

import logging
from typing import Any
import pandas as pd
from joblib import Parallel, delayed

from . import fetcher, filters, universe
from .composite import composite_score, DEFAULT_WEIGHTS
from .scores import quality, valuation, accumulation, momentum

logger = logging.getLogger(__name__)


def _process_ticker(
    symbol: str,
    sector: str,
    ohlcv: pd.DataFrame | None,
    market_cap_min: float,
    adv_min_usd: float,
    history_min_days: int,
    momentum_min: float,
    sma200_distance_min: float,
    weights: dict,
) -> dict[str, Any] | None:
    """Procesa un ticker. Devuelve dict con scores o None si fue excluido."""
    if ohlcv is None or ohlcv.empty:
        return {"symbol": symbol, "sector": sector, "skipped": "no_ohlcv"}

    info = fetcher.fetch_info(symbol)
    if info is None:
        return {"symbol": symbol, "sector": sector, "skipped": "no_info"}

    passes, reason = filters.passes_hard_filters(
        info, ohlcv,
        market_cap_min=market_cap_min,
        adv_min_usd=adv_min_usd,
        history_min_days=history_min_days,
        momentum_min=momentum_min,
        sma200_distance_min=sma200_distance_min,
    )
    if not passes:
        return {"symbol": symbol, "sector": sector, "skipped": reason}

    # Scores que solo necesitan OHLCV
    acc_res = accumulation.accumulation_score(ohlcv)
    mom_res = momentum.momentum_score(ohlcv)
    val_res = valuation.valuation_score(info, ohlcv)

    # Score de calidad — necesita financials
    fin = fetcher.fetch_financials(symbol)
    qual_res = quality.piotroski_score(fin.get("qis"), fin.get("qbs"), fin.get("qcf"))

    scores = {
        "quality":      qual_res.score,
        "valuation":    val_res.score,
        "accumulation": acc_res.score,
        "momentum":     mom_res.score,
        "insider":      None,  # no implementado en MVP
    }
    composite = composite_score(scores, weights)

    return {
        "symbol": symbol,
        "sector": sector,
        "skipped": None,
        "market_cap": info.get("marketCap"),
        "price": float(ohlcv["close"].iloc[-1]),
        # Scores
        "score_total": composite,
        "score_quality": qual_res.score,
        "score_valuation": val_res.score,
        "score_accumulation": acc_res.score,
        "score_momentum": mom_res.score,
        # Detalles utiles para revision
        "f_score": qual_res.f_score,
        "drawdown_ath": val_res.drawdown_ath,
        "pe": val_res.pe,
        "pb": val_res.pb,
        "ev_ebitda": val_res.ev_ebitda,
        "divergence": acc_res.divergence_kind,
        "obv_slope": acc_res.obv_slope,
        "price_slope_60d": acc_res.price_slope,
        "cmf_30d": acc_res.cmf_avg_30d,
        "vol_ratio": acc_res.volume_ratio,
        "mom_12_1": mom_res.momentum_12_1,
        "dist_sma200": mom_res.distance_to_sma200,
        "dividend_yield": info.get("dividendYield"),
        "roe": info.get("returnOnEquity"),
    }


def run_screener(
    universe_name: str = "sp500",
    custom_tickers: list[str] | None = None,
    market_cap_min: float = 10e9,
    adv_min_usd: float = 50e6,
    history_min_days: int = 504,
    momentum_min: float = -0.35,
    sma200_distance_min: float = -0.30,
    weights: dict | None = None,
    n_jobs: int = 4,
    refresh: bool = False,
) -> pd.DataFrame:
    """Corre el screener sobre el universo elegido.

    Devuelve DataFrame con todos los tickers procesados (incluidos skipped),
    ordenado por score_total descendente.
    """
    weights = weights or DEFAULT_WEIGHTS

    # 1. Universo
    if universe_name == "custom" and custom_tickers:
        univ_df = universe.load_custom(custom_tickers)
    else:
        univ_df = universe.load_sp500(refresh=refresh)

    tickers = univ_df["Symbol"].tolist()
    sectors = dict(zip(univ_df["Symbol"], univ_df["GICS_Sector"]))
    logger.info("Universe: %d tickers", len(tickers))

    # 2. Bulk OHLCV
    ohlcv_map = fetcher.fetch_ohlcv_bulk(tickers, period="5y", refresh=refresh)
    logger.info("OHLCV ready for %d/%d tickers", len(ohlcv_map), len(tickers))

    # 3. Procesamiento paralelo
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_process_ticker)(
            t,
            sectors.get(t, "Unknown"),
            ohlcv_map.get(t),
            market_cap_min, adv_min_usd, history_min_days,
            momentum_min, sma200_distance_min,
            weights,
        )
        for t in tickers
    )

    df = pd.DataFrame([r for r in results if r is not None])
    if df.empty:
        return df

    # Ordenamos: primero los procesados con score, despues los skipped
    processed = df[df["skipped"].isna()].sort_values("score_total", ascending=False)
    skipped = df[df["skipped"].notna()]

    return pd.concat([processed, skipped], ignore_index=True)
