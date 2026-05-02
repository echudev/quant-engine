"""Filtros duros aplicados antes de calcular scores."""
from __future__ import annotations

import pandas as pd

from . import indicators as ind


def passes_hard_filters(
    info: dict | None,
    ohlcv: pd.DataFrame | None,
    market_cap_min: float = 10e9,
    adv_min_usd: float = 50e6,
    history_min_days: int = 504,
    momentum_min: float = -0.35,
    sma200_distance_min: float = -0.30,
) -> tuple[bool, str | None]:
    """Devuelve (pasa, razon_si_no_pasa).

    Filtros aplicados:
      - market_cap >= market_cap_min
      - ADV (avg daily volume * close) ultimos 60 dias >= adv_min_usd
      - historia disponible >= history_min_days
      - momentum_12_1 > momentum_min (anti falling knife)
      - distancia a SMA200 > sma200_distance_min (descarta caidas confirmadas)
    """
    if ohlcv is None or ohlcv.empty:
        return False, "no_ohlcv"

    if len(ohlcv) < history_min_days:
        return False, f"history_too_short:{len(ohlcv)}<{history_min_days}"

    if info is None:
        return False, "no_info"

    market_cap = info.get("marketCap")
    if not market_cap or market_cap < market_cap_min:
        return False, f"market_cap_too_small:{market_cap}"

    # ADV en USD
    tail = ohlcv.tail(60)
    adv_usd = float((tail["volume"] * tail["close"]).mean())
    if adv_usd < adv_min_usd:
        return False, f"adv_too_low:{adv_usd:.0f}<{adv_min_usd:.0f}"

    # Anti-falling-knife: momentum 12-1 catastrofico
    mom = ind.momentum_12_1(ohlcv["close"])
    if mom < momentum_min:
        return False, f"momentum_too_negative:{mom:.2f}<{momentum_min}"

    # Distancia a SMA200: si esta muy abajo, todavia no recupero
    if len(ohlcv) >= 200:
        dist = ind.distance_to_sma(ohlcv["close"], 200)
        if dist < sma200_distance_min:
            return False, f"below_sma200:{dist:.2f}<{sma200_distance_min}"

    return True, None
