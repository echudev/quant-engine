"""
Shared data preparation: daily trend filter applied to intraday frames.

Used by main.py and experiments. The daily trend is lagged one day so that
intraday bars of day D only see the trend computed with day D-1's close —
using day D's own close would be look-ahead (the close isn't known intraday).
"""

import pandas as pd


def build_daily_trend(df_daily: pd.DataFrame, ma_period: int = 50) -> pd.Series:
    ma    = df_daily["close"].rolling(ma_period).mean()
    trend = (df_daily["close"] > ma)
    trend.index = trend.index.normalize()
    return trend


def apply_daily_trend(df: pd.DataFrame, daily_trend: pd.Series) -> pd.DataFrame:
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    # Lag: bar of day D reads the trend of day D-1 (last fully closed day)
    prev_day = df.index.normalize() - pd.Timedelta(days=1)
    df["trend_daily"] = (
        daily_trend.reindex(prev_day, method="ffill")
        .fillna(False)
        .values
    )
    return df
