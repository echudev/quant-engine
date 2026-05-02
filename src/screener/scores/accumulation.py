"""Score de Acumulacion (Wyckoff).

Componentes (suman a peso total):
  - OBV slope normalizado 60d         25%
  - Divergencia OBV vs precio 60d     25%
  - CMF(20) promedio 30d              20%
  - Volume ratio 20d / 252d           15%
  - A/D Line slope normalizado 60d    15%
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import pandas as pd

from .. import indicators as ind


@dataclass
class AccumulationResult:
    score: float
    obv_slope: float            # normalizado: % por dia
    price_slope: float          # normalizado: % por dia
    divergence_kind: str        # bullish_div | bullish_confirm | distribution | none
    cmf_avg_30d: float
    volume_ratio: float
    ad_slope: float
    components: dict = field(default_factory=dict)


def _sigmoid_score(x: float, scale: float = 0.005) -> float:
    """Mapea un slope normalizado (% por dia) a score 0-100 via sigmoide.

    scale controla cuan rapido satura. Default 0.5%/dia ya da score ~73.
    """
    return 100.0 / (1.0 + math.exp(-x / scale))


def _bucket_score(value: float, thresholds: list[float], scores: list[float]) -> float:
    for thr, sc in zip(thresholds, scores):
        if value <= thr:
            return sc
    return scores[-1]


def accumulation_score(ohlcv: pd.DataFrame) -> AccumulationResult:
    if ohlcv is None or ohlcv.empty or len(ohlcv) < 60:
        return AccumulationResult(
            score=0.0, obv_slope=0.0, price_slope=0.0,
            divergence_kind="none", cmf_avg_30d=0.0,
            volume_ratio=0.0, ad_slope=0.0,
            components={"reason": "insufficient_history"},
        )

    obv_series = ind.obv(ohlcv)
    ad_series = ind.ad_line(ohlcv)
    cmf_series = ind.cmf(ohlcv, period=20)

    # Slopes normalizados sobre los ultimos 60 dias
    obv_slope = ind.normalized_slope(obv_series, 60)
    ad_slope = ind.normalized_slope(ad_series, 60)
    price_slope = ind.normalized_slope(ohlcv["close"], 60)

    obv_slope_score = _sigmoid_score(obv_slope, scale=0.005)
    ad_slope_score = _sigmoid_score(ad_slope, scale=0.005)

    # Divergencia OBV vs precio
    if obv_slope > 0 and price_slope <= 0:
        div_kind = "bullish_divergence"
        div_score = 100.0
    elif obv_slope > 0 and price_slope > 0:
        div_kind = "bullish_confirm"
        div_score = 60.0
    elif obv_slope <= 0 and price_slope <= 0:
        div_kind = "distribution"
        div_score = 20.0
    else:
        # OBV cayendo, precio subiendo — divergencia bearish
        div_kind = "bearish_divergence"
        div_score = 0.0

    # CMF promedio ultimos 30 dias
    cmf_avg = float(cmf_series.tail(30).mean()) if cmf_series.notna().sum() >= 30 else 0.0
    # Mapeamos: -0.1 -> 0, +0.1 -> 100
    cmf_score = max(0.0, min(100.0, (cmf_avg + 0.1) / 0.2 * 100.0))

    # Volume ratio 20d / 252d
    if len(ohlcv) >= 252:
        vol_20 = float(ohlcv["volume"].tail(20).mean())
        vol_252 = float(ohlcv["volume"].tail(252).mean())
        vol_ratio = (vol_20 / vol_252) if vol_252 > 0 else 0.0
    else:
        vol_20 = float(ohlcv["volume"].tail(20).mean())
        vol_full = float(ohlcv["volume"].mean())
        vol_ratio = (vol_20 / vol_full) if vol_full > 0 else 0.0

    vol_score = _bucket_score(vol_ratio, [0.8, 1.0, 1.2, 1.5], [0, 30, 60, 85])
    if vol_ratio > 1.5:
        vol_score = 100.0

    final = (
        0.25 * obv_slope_score +
        0.25 * div_score +
        0.20 * cmf_score +
        0.15 * vol_score +
        0.15 * ad_slope_score
    )

    return AccumulationResult(
        score=float(final),
        obv_slope=float(obv_slope),
        price_slope=float(price_slope),
        divergence_kind=div_kind,
        cmf_avg_30d=cmf_avg,
        volume_ratio=float(vol_ratio),
        ad_slope=float(ad_slope),
        components={
            "obv_slope_score": obv_slope_score,
            "divergence_score": div_score,
            "cmf_score": cmf_score,
            "vol_score": vol_score,
            "ad_slope_score": ad_slope_score,
        },
    )
