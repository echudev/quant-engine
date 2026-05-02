"""Score de Valoracion — drawdown desde ATH + multiples actuales.

Para el MVP usamos:
  - Drawdown desde ATH (peso 50%)
  - P/E, P/B, EV/EBITDA absolutos vs buckets fijos (peso 50% combinado)

En fase 2 incorporamos percentiles vs propia historia 5y.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

from .. import indicators as ind


@dataclass
class ValuationResult:
    score: float
    drawdown_ath: float
    pe: Optional[float]
    pb: Optional[float]
    ev_ebitda: Optional[float]
    pfcf: Optional[float]
    components: dict = field(default_factory=dict)


def _bucket_score(value: float, thresholds: list[float], scores: list[float]) -> float:
    """Devuelve el score correspondiente al primer threshold que value <= threshold.

    thresholds debe estar ordenado ascendente.
    Si value > ultimo threshold, devuelve scores[-1].
    """
    for thr, sc in zip(thresholds, scores):
        if value <= thr:
            return sc
    return scores[-1]


def valuation_score(info: dict, ohlcv: pd.DataFrame) -> ValuationResult:
    """Score 0-100. Combina drawdown desde ATH + buckets de multiples."""
    # 1. Drawdown desde ATH (peso 50%)
    dd_ath = ind.drawdown_from_ath(ohlcv["close"]) if not ohlcv.empty else 0.0
    # cap el drawdown a 0.7 (mas alla de eso no necesariamente es mejor)
    dd_capped = min(dd_ath, 0.70)
    dd_score = (dd_capped / 0.70) * 100.0

    # 2. Multiples (peso 50% combinado)
    pe = info.get("trailingPE")
    pb = info.get("priceToBook")
    ev_ebitda = info.get("enterpriseToEbitda")
    pfcf = None
    # P/FCF aproximado: market_cap / FCF ttm si esta disponible
    market_cap = info.get("marketCap")
    fcf = info.get("freeCashflow")
    if market_cap and fcf and fcf > 0:
        pfcf = market_cap / fcf

    # Buckets (lower = mejor score). Negativos = excluir.
    pe_score = None
    if pe is not None and pe > 0:
        pe_score = _bucket_score(pe, [10, 15, 20, 30, 50], [100, 80, 60, 40, 20])

    pb_score = None
    if pb is not None and pb > 0:
        pb_score = _bucket_score(pb, [1.0, 1.5, 2.5, 4.0, 6.0], [100, 80, 60, 40, 20])

    ev_ebitda_score = None
    if ev_ebitda is not None and ev_ebitda > 0:
        ev_ebitda_score = _bucket_score(ev_ebitda, [6, 10, 14, 20, 30], [100, 80, 60, 40, 20])

    pfcf_score = None
    if pfcf is not None and pfcf > 0:
        pfcf_score = _bucket_score(pfcf, [10, 15, 20, 30, 50], [100, 80, 60, 40, 20])

    # Promedio de los multiples disponibles
    multiples_scores = [s for s in (pe_score, pb_score, ev_ebitda_score, pfcf_score) if s is not None]
    if multiples_scores:
        multiples_avg = sum(multiples_scores) / len(multiples_scores)
    else:
        multiples_avg = 50.0  # neutral si no hay multiples disponibles

    # Combinamos 50/50
    final = 0.5 * dd_score + 0.5 * multiples_avg

    return ValuationResult(
        score=float(final),
        drawdown_ath=float(dd_ath),
        pe=float(pe) if pe is not None else None,
        pb=float(pb) if pb is not None else None,
        ev_ebitda=float(ev_ebitda) if ev_ebitda is not None else None,
        pfcf=float(pfcf) if pfcf is not None else None,
        components={
            "dd_score": dd_score,
            "pe_score": pe_score,
            "pb_score": pb_score,
            "ev_ebitda_score": ev_ebitda_score,
            "pfcf_score": pfcf_score,
            "multiples_avg": multiples_avg,
        },
    )
