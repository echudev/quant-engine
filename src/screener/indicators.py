"""Indicadores tecnicos y metricas de precio.

Funciones puras sobre DataFrames OHLCV. Asume columnas en lowercase:
open, high, low, close, volume.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume.

    OBV[t] = OBV[t-1] + volume[t]   if close[t] > close[t-1]
           = OBV[t-1] - volume[t]   if close[t] < close[t-1]
           = OBV[t-1]               if close[t] == close[t-1]
    """
    direction = np.sign(df["close"].diff().fillna(0.0))
    signed_vol = direction * df["volume"]
    return signed_vol.cumsum().rename("obv")


def ad_line(df: pd.DataFrame) -> pd.Series:
    """Accumulation/Distribution Line.

    A/D = sum( CLV * volume ) where CLV = ((C-L)-(H-C))/(H-L)
    """
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / rng
    clv = clv.fillna(0.0)
    return (clv * df["volume"]).cumsum().rename("ad_line")


def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Chaikin Money Flow.

    CMF = sum_N( CLV * volume ) / sum_N( volume )
    """
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / rng
    mfm = mfm.fillna(0.0)
    mfv = mfm * df["volume"]
    cmf_series = mfv.rolling(period).sum() / df["volume"].rolling(period).sum()
    return cmf_series.rename(f"cmf_{period}")


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def linear_slope(series: pd.Series, n: int | None = None) -> float:
    """Pendiente lineal (regresion OLS) sobre los ultimos n puntos.

    Devuelve la pendiente en unidades-Y / unidad-de-indice.
    Si la serie tiene <2 puntos validos, devuelve 0.0.
    """
    s = series.dropna() if n is None else series.dropna().tail(n)
    if len(s) < 2:
        return 0.0
    x = np.arange(len(s), dtype=float)
    y = s.to_numpy(dtype=float)
    # cov(x,y) / var(x)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)


def normalized_slope(series: pd.Series, n: int | None = None) -> float:
    """Slope dividido por el promedio reciente, para hacer la magnitud
    comparable entre series con escalas distintas (precio vs OBV vs A/D).

    Devuelve el % de cambio por unidad de indice (~ por dia).
    """
    s = series.dropna() if n is None else series.dropna().tail(n)
    if len(s) < 2:
        return 0.0
    slope = linear_slope(s)
    base = abs(s.mean())
    if base == 0:
        return 0.0
    return slope / base


def momentum_12_1(close: pd.Series) -> float:
    """Return 12 meses excluyendo el ultimo mes (~21 dias).

    Jegadeesh-Titman 1993. Necesita >= 252 dias de historia.
    """
    if len(close) < 252:
        return 0.0
    end = close.iloc[-21]
    start = close.iloc[-252]
    if start <= 0:
        return 0.0
    return float(end / start - 1.0)


def drawdown_from_ath(close: pd.Series) -> float:
    """% de drawdown desde el maximo historico de la serie."""
    if close.empty:
        return 0.0
    ath = close.max()
    if ath <= 0:
        return 0.0
    return float((ath - close.iloc[-1]) / ath)


def drawdown_from_52w_high(close: pd.Series) -> float:
    if len(close) < 2:
        return 0.0
    window = close.tail(252) if len(close) >= 252 else close
    high = window.max()
    if high <= 0:
        return 0.0
    return float((high - close.iloc[-1]) / high)


def distance_to_sma(close: pd.Series, period: int) -> float:
    """% que el precio actual esta por encima/debajo de la SMA de N periodos.

    Devuelve 0.0 si no hay suficiente historia.
    """
    if len(close) < period:
        return 0.0
    sma_val = close.tail(period).mean()
    if sma_val <= 0:
        return 0.0
    return float(close.iloc[-1] / sma_val - 1.0)
