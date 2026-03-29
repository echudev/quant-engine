# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the full backtest pipeline
python main.py

# Install dependencies (uses uv)
uv pip install -e .

# Install from scratch
python -m venv .venv
source .venv/Scripts/activate  # Windows
uv pip install -e .
```

No test suite exists currently.

## Architecture

This is a multi-strategy crypto backtesting engine. The data flow is:

```
config/settings.toml
       ↓
main.py (orchestrator)
       ↓
src/data/fetcher.py          → Fetches OHLCV (Binance/CCXT), OI (Bybit v5), Funding Rates (Binance Futures)
                               Caches data in data/*.parquet (incremental — only re-downloads new bars)
       ↓
src/strategies/base.py       → BaseStrategy abstract class (compute_indicators → generate_signals)
src/strategies/liquidity_sweep.py  → Entry: OI drop + funding rate + bullish confirmation + trend
src/strategies/rsi_reversion.py    → Entry: RSI oversold recovery + trend filter; Exit: RSI exit level
       ↓
src/backtest/engine.py       → Wraps vectorbt.Portfolio.from_signals(); runs grid search and walk-forward
       ↓
src/reporting/metrics.py     → Console output (print_report, print_comparison, print_walkforward_summary)
src/reporting/plots.py       → Equity curves, drawdowns, multi-symbol bar charts → plots/*.png
       ↓
results/run_{timestamp}.csv  → All strategy/symbol combinations
results/wf_{symbol}_{ts}.csv → Walk-forward validation results
```

### Key Design Points

**Strategy pattern**: Each strategy is a `BaseStrategy` subclass with a `StrategyParams` dataclass. To add a new strategy: subclass `BaseStrategy`, define a params dataclass, implement `compute_indicators()` and `generate_signals()` (must write an `entry` column; optionally an `exit_rsi` column for manual exits).

**Backtest engine** (`src/backtest/engine.py`):
- `run_backtest()` — calls `vectorbt.Portfolio.from_signals()` with `atr_sl` column for dynamic stop-loss and `take_profit` % from params
- `optimize_strategy()` — parallelized grid search (joblib threads, n_jobs=-1)
- `walk_forward()` — rolling train/test windows; parallelized at window level; inner grid search is sequential per window

**Data merging**: `merge_derivatives()` in fetcher uses `pd.merge_asof()` with a 90-minute tolerance to handle timestamp misalignment between exchanges.

**Configuration**: All symbols, timeframes, strategy parameters, and optimization thresholds live in `config/settings.toml`. Per-symbol parameter overrides can be set under each symbol's table (e.g., `[symbols.SOL.liquidity_sweep]`).

**Multi-exchange data sources**:
- OHLCV: Binance via CCXT
- Open Interest: Bybit v5 REST API (no auth required)
- Funding Rates: Binance Futures REST API (no auth required)

**Walk-forward validation** runs only for ETH and SOL (configurable in `main.py`). Uses 12-month train / 6-month test rolling windows.
