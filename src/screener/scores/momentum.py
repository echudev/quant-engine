"""Score de Momentum confirmador.

Componentes:
  - Momentum 12-1 (Jegadeesh-Titman)        50%
  - Distancia a SMA200                       30%
  - Slope SMA50 30d                          20%
"""
from __future__ import annotations

from dataclasses import dataclass, field
import pandas as pd

from .. import indicators as ind


@dataclass
class MomentumResult:
    score: float
    momentum_12_1: float
    distance_to_sma200: float
    sma50_slope_30d: float
    components: dict = field(default_factory=dict)


def _bucket_score(value: float, thresholds: list[float], scores: list[float]) -> float:
    for thr, sc in zip(thresholds, scores):
        if value <= thr:
            return sc
    return scores[-1]


def momentum_score(ohlcv: pd.DataFrame) -> MomentumResult:
    if ohlcv is None or ohlcv.empty or len(ohlcv) < 50:
        return MomentumResult(
            score=0.0, momentum_12_1=0.0,
            distance_to_sma200=0.0, sma50_slope_30d=0.0,
            components={"reason": "insufficient_history"},
        )

    close = ohlcv["close"]

    # 1. Momentum 12-1
    mom = ind.momentum_12_1(close)
    # buckets: <-20% -> 0, -20% a 0 -> 40, 0 a 20% -> 80, >20% -> 100
    mom_score = _bucket_score(mom, [-0.20, 0.0, 0.20], [0, 40, 80])
    if mom > 0.20:
        mom_score = 100.0

    # 2. Distancia a SMA200
    if len(close) >= 200:
        dist = ind.distance_to_sma(close, 200)
        # < -30%: 20 | -30% a -10%: 50 | -10% a 0: 80 | 0 a 20%: 100 | >20%: 60 (estirado)
        if dist <= -0.30:
            dist_score = 20.0
        elif dist <= -0.10:
            dist_score = 50.0
        elif dist <= 0.0:
            dist_score = 80.0
        elif dist <= 0.20:
            dist_score = 100.0
        else:
            dist_score = 60.0
    else:
        dist = 0.0
        dist_score = 50.0  # neutral sin datos

    # 3. SMA50 slope sobre 30d (% del valor de SMA50 actual)
    if len(close) >= 80:
        sma50 = ind.sma(close, 50)
        slope = ind.normalized_slope(sma50, 30)  # % por dia
        # buckets: <-0.0017 -> 0 | -0.0017 a 0 -> 40 | 0 a 0.0017 -> 80 | >0.0017 -> 100
        # (~0.17%/dia ~ 5%/mes)
        if slope <= -0.0017:
            sma50_score = 0.0
        elif slope <= 0.0:
            sma50_score = 40.0
        elif slope <= 0.0017:
            sma50_score = 80.0
        else:
            sma50_score = 100.0
    else:
        slope = 0.0
        sma50_score = 50.0

    final = 0.50 * mom_score + 0.30 * dist_score + 0.20 * sma50_score

    return MomentumResult(
        score=float(final),
        momentum_12_1=float(mom),
        distance_to_sma200=float(dist),
        sma50_slope_30d=float(slope),
        components={
            "mom_score": mom_score,
            "dist_score": dist_score,
            "sma50_score": sma50_score,
        },
    )
