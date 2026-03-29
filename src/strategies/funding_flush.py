"""
Funding Rate Flush Strategy.

Concept: When perpetual futures funding rate is elevated relative to
its recent history (crowded longs), the market is vulnerable to a
long squeeze. When OI then drops sharply (forced liquidations), a
high-volume flush exhausts sellers and creates a high-probability
recovery entry.

Key difference from LiquiditySweep:
  - Uses rolling PERCENTILE rank for funding (adaptive to each asset's
    funding volatility — e.g. SOL swings -0.93% to +0.12% vs BTC's
    tighter range; a fixed threshold would miss or over-fire).
  - Adds VOLUME spike confirmation to distinguish real capitulation
    from gradual OI bleed (low-volume OI drift ≠ liquidation flush).
  - More selective → fewer but higher-quality signals.

Signal logic:
  1. Funding rate percentile (rolling funding_rank_window hours) >=
     funding_pct_rank threshold
     → market has been crowded with longs, vulnerable to squeeze
  2. OI dropped >= oi_drop_pct% accumulated over oi_window candles
     → long liquidations are actively happening
  3. Price declined during the sweep window
     → confirms involuntary exits, not profit-taking
  4. Volume spike: max volume in sweep window >= vol_mult × vol_ma
     → panic selling confirms flush is real (not slow bleed)
  5. Current candle is bullish with minimum body size
     → first sign of recovery / demand absorption
  6. 1H price > MA50 AND daily price > MA50
     → trade only in the direction of the dominant trend
"""

from dataclasses import dataclass
import pandas as pd
from .base import BaseStrategy, StrategyParams


@dataclass
class FundingFlushParams(StrategyParams):
    oi_drop_pct       : float = 3.0    # min OI drop % accumulated in window
    oi_window         : int   = 3      # candles over which OI drop is measured
    funding_pct_rank  : float = 65.0   # funding percentile threshold (0-100)
    funding_rank_window: int  = 168    # lookback for percentile (168h = 7 days)
    confirm_body      : float = 0.3    # min body % for recovery candle
    vol_mult          : float = 1.3    # volume spike: max_vol_in_window >= mult × vol_ma
    vol_ma_period     : int   = 24     # period for volume baseline MA
    ma_trend_1h       : int   = 50     # 1H MA for trend filter


class FundingFlushStrategy(BaseStrategy):

    def __init__(self, params: FundingFlushParams):
        super().__init__(params)
        self.p = params

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # ── OI drop: accumulate negative OI pct changes over window
        df["oi_pct_change"] = df["oi_usdt"].pct_change() * 100
        oi_drops = df["oi_pct_change"].clip(upper=0)
        df["oi_drop_roll"] = oi_drops.rolling(self.p.oi_window).sum().abs()

        # ── Funding rate rolling percentile rank (adaptive per asset)
        # rank(pct=True) gives values 0..1 → multiply by 100 for % scale
        df["funding_pct"] = (
            df["funding_rate"]
            .rolling(self.p.funding_rank_window, min_periods=24)
            .rank(pct=True) * 100
        )

        # ── Volume baseline and sweep-window spike
        df["vol_ma"] = df["volume"].rolling(self.p.vol_ma_period).mean()
        # highest volume in the last oi_window candles (the flush window)
        df["vol_max_window"] = df["volume"].rolling(self.p.oi_window).max()

        # ── Candle body for recovery confirmation
        df["body_pct"] = abs(df["close"] - df["open"]) / df["open"] * 100

        # ── 1H trend filter
        df["ma50"]     = df["close"].rolling(self.p.ma_trend_1h).mean()
        df["trend_1h"] = df["close"] > df["ma50"]

        # ── Dynamic ATR-based stop loss
        df["atr_sl"] = self.atr_sl_series(df)

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # All sweep conditions reference the candle BEFORE current (shift 1)
        # so entry triggers on the open of the recovery candle.

        # 1. Funding was elevated during the flush window
        funding_elevated = df["funding_pct"].shift(1) >= self.p.funding_pct_rank

        # 2. OI dropped significantly in sweep window
        oi_swept = df["oi_drop_roll"].shift(1) >= self.p.oi_drop_pct

        # 3. Price fell during the sweep window (confirms liquidations, not exits)
        price_drop = df["close"].shift(1) < df["open"].shift(1)

        # 4. Volume spike during flush (panic selling = exhaustion signal)
        vol_spike = (
            df["vol_max_window"].shift(1) >= df["vol_ma"].shift(1) * self.p.vol_mult
        )

        # 5. Recovery candle: bullish with minimum body
        bullish_recovery = (
            (df["close"] > df["open"]) &
            (df["body_pct"] >= self.p.confirm_body)
        )

        # 6. Trend alignment (1H and daily)
        trend_ok = df["trend_1h"] & df["trend_daily"]

        df["entry"] = (
            funding_elevated &
            oi_swept &
            price_drop &
            vol_spike &
            bullish_recovery &
            trend_ok
        )
        return df
