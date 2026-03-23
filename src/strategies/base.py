"""
Base strategy interface.
All strategies implement compute_indicators() and generate_signals().
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


@dataclass
class StrategyParams:
    """Base parameter container. Subclasses add strategy-specific fields."""
    take_profit : float = 4.0
    atr_mult    : float = 2.0
    atr_period  : int   = 14
    ma_trend_1d : int   = 50


class BaseStrategy(ABC):
    """
    Abstract base for all trading strategies.

    Subclasses must implement:
        compute_indicators(df) -> pd.DataFrame
        generate_signals(df)   -> pd.DataFrame  (adds 'entry' bool column)

    Optionally override:
        exit_column  : name of exit signal column (default: no signal exit)
    """

    exit_column: Optional[str] = None

    def __init__(self, params: StrategyParams):
        self.params = params

    @abstractmethod
    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicator columns to df. Returns new DataFrame."""
        ...

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 'entry' boolean column to df. Returns new DataFrame."""
        ...

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full pipeline: indicators → signals. Convenience method."""
        df = self.compute_indicators(df)
        df = self.generate_signals(df)
        return df

    @staticmethod
    def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"]  - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def atr_sl_series(self, df: pd.DataFrame) -> pd.Series:
        """Dynamic stop-loss as fraction of price: atr_mult × ATR / close."""
        atr = self.compute_atr(df, self.params.atr_period)
        return (self.params.atr_mult * atr / df["close"]).bfill()
