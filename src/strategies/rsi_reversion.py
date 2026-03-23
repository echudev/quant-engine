"""
RSI Mean Reversion Strategy.

Signal logic:
  1. RSI crosses upward through oversold level
     → exhaustion reversal signal
  2. 1H trend (close > MA200) AND daily trend both bullish
     → trading with the market
  3. Exit when RSI rises above exit_level OR SL/TP triggered
"""

from dataclasses import dataclass
import pandas as pd
import numpy as np
from .base import BaseStrategy, StrategyParams


@dataclass
class RSIReversionParams(StrategyParams):
    rsi_period  : int   = 14
    oversold    : float = 35.0
    exit_level  : float = 55.0
    ma_trend    : int   = 200


class RSIReversionStrategy(BaseStrategy):

    exit_column = "exit_rsi"

    def __init__(self, params: RSIReversionParams):
        super().__init__(params)
        self.p = params

    @staticmethod
    def _compute_rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["rsi"]      = self._compute_rsi(df["close"], self.p.rsi_period)
        df["ma200"]    = df["close"].rolling(self.p.ma_trend).mean()
        df["trend_1h"] = df["close"] > df["ma200"]
        df["atr_sl"]   = self.atr_sl_series(df)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        rsi_was_below  = df["rsi"].shift(1) < self.p.oversold
        rsi_now_above  = df["rsi"] >= self.p.oversold
        trend_ok       = df["trend_1h"] & df["trend_daily"]
        df["entry"]    = rsi_was_below & rsi_now_above & trend_ok
        df["exit_rsi"] = df["rsi"] > self.p.exit_level
        return df
