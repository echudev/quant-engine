"""
Liquidity Sweep OI Strategy.

Signal logic:
  1. OI dropped > oi_drop_pct% accumulated over oi_window candles
     → proxy for forced long liquidations
  2. Funding rate > funding_min before the sweep
     → market was loaded with longs, vulnerable to flush
  3. Price dropped during the sweep candle
     → OI drop coincides with price decline = liquidations (not voluntary exit)
  4. Current candle is bullish with minimum body
     → first sign of recovery
  5. 1H trend (close > MA50) AND daily trend (close > MA50 daily) both bullish
     → trading with the market
"""

from dataclasses import dataclass
import pandas as pd
from .base import BaseStrategy, StrategyParams


@dataclass
class LiquiditySweepParams(StrategyParams):
    oi_drop_pct  : float = 4.0
    oi_window    : int   = 3
    funding_min  : float = 0.00005
    confirm_body : float = 0.2
    ma_trend_1h  : int   = 50


class LiquiditySweepStrategy(BaseStrategy):

    def __init__(self, params: LiquiditySweepParams):
        super().__init__(params)
        self.p = params

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # OI drop: accumulate negative OI changes over window
        df["oi_pct_change"] = df["oi_usdt"].pct_change() * 100
        oi_drops = df["oi_pct_change"].clip(upper=0)
        df["oi_drop_roll"]  = oi_drops.rolling(self.p.oi_window).sum().abs()

        # Funding rate rolling mean
        df["funding_ma"] = df["funding_rate"].rolling(3).mean()

        # Candle body size for confirmation
        df["body_pct"] = abs(df["close"] - df["open"]) / df["open"] * 100

        # 1H trend filter
        df["ma50"]     = df["close"].rolling(self.p.ma_trend_1h).mean()
        df["trend_1h"] = df["close"] > df["ma50"]

        # Dynamic ATR-based stop loss
        df["atr_sl"] = self.atr_sl_series(df)

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        oi_swept    = df["oi_drop_roll"].shift(1) >= self.p.oi_drop_pct
        funded_long = df["funding_ma"].shift(1)   >= self.p.funding_min
        price_drop  = df["close"].shift(1) < df["open"].shift(1)
        bullish_ok  = (df["close"] > df["open"]) & (df["body_pct"] >= self.p.confirm_body)
        trend_ok    = df["trend_1h"] & df["trend_daily"]

        df["entry"] = oi_swept & funded_long & price_drop & bullish_ok & trend_ok
        return df
