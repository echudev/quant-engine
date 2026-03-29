"""
RSI Mean Reversion Strategy.

Signal logic:
  1. RSI crosses upward through oversold level (default: 30)
     → exhaustion reversal signal (tighter threshold = higher quality)
  2. Volume spike on the dip: max volume in last vol_lookback candles
     must exceed vol_mult × vol_ma
     → distinguishes real capitulation (panic selling) from slow bleed
  3. 1H trend (close > MA200) AND daily trend both bullish
     → trading with the market
  4. Exit when RSI rises above exit_level OR SL/TP triggered

Changes vs original:
  - oversold 35 → 30: fewer but higher-conviction entries
  - Added volume confirmation: eliminates low-volume dips that fail
    to attract buyers (the main cause of RSI mean reversion losses)
"""

from dataclasses import dataclass
import pandas as pd
import numpy as np
from .base import BaseStrategy, StrategyParams


@dataclass
class RSIReversionParams(StrategyParams):
    rsi_period   : int   = 14
    oversold     : float = 30.0
    exit_level   : float = 55.0
    ma_trend     : int   = 200
    vol_mult     : float = 1.5    # volume spike multiplier vs rolling MA
    vol_ma_period: int   = 20     # lookback for volume baseline MA
    vol_lookback : int   = 3      # candles to look back for the volume spike


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
        df["rsi"]       = self._compute_rsi(df["close"], self.p.rsi_period)
        df["ma200"]     = df["close"].rolling(self.p.ma_trend).mean()
        df["trend_1h"]  = df["close"] > df["ma200"]
        df["vol_ma"]    = df["volume"].rolling(self.p.vol_ma_period).mean()
        # highest volume in the dip window (panic candles before recovery)
        df["vol_max"]   = df["volume"].rolling(self.p.vol_lookback).max()
        df["atr_sl"]    = self.atr_sl_series(df)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        rsi_was_below = df["rsi"].shift(1) < self.p.oversold
        rsi_now_above = df["rsi"] >= self.p.oversold
        trend_ok      = df["trend_1h"] & df["trend_daily"]
        # Volume spike: the highest volume candle in the dip window
        # must have been above the baseline (real capitulation)
        vol_spike     = df["vol_max"].shift(1) >= (df["vol_ma"].shift(1) * self.p.vol_mult)
        df["entry"]   = rsi_was_below & rsi_now_above & trend_ok & vol_spike
        df["exit_rsi"] = df["rsi"] > self.p.exit_level
        return df
