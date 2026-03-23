"""
Quant Engine — main entry point.

Runs Liquidity Sweep OI strategy across multiple symbols.
Config: config/settings.toml
"""

import tomllib
import pandas as pd
from pathlib import Path

from src.data.fetcher import (
    fetch_ohlcv, fetch_open_interest, fetch_funding_rate, merge_derivatives
)
from src.strategies.liquidity_sweep import LiquiditySweepStrategy, LiquiditySweepParams
from src.strategies.rsi_reversion import RSIReversionStrategy, RSIReversionParams
from src.backtest.engine import run_backtest, extract_metrics, optimize_strategy
from src.reporting.metrics import print_report, print_comparison
from src.reporting.plots import plot_equity_curves, plot_multi_symbol

# ──────────────────────────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────────────────────────

with open("config/settings.toml", "rb") as f:
    CFG = tomllib.load(f)

GLOBAL     = CFG["global"]
CACHE_DIR   = Path("data")
PLOTS_DIR   = Path("plots")
RESULTS_DIR = Path("results")
for d in (CACHE_DIR, PLOTS_DIR, RESULTS_DIR):
    d.mkdir(exist_ok=True)

TIMEFRAME  = GLOBAL["timeframe"]
TF_TREND   = GLOBAL["tf_trend"]
SINCE_DATE = GLOBAL["since_date"]
INIT_CASH  = GLOBAL["init_cash"]
EXCHANGE   = GLOBAL["exchange_id"]

LS_CFG  = CFG["strategy"]["liquidity_sweep"]
RSI_CFG = CFG["strategy"]["rsi_reversion"]


def build_ls_params(symbol_name: str) -> LiquiditySweepParams:
    """Build LiquiditySweepParams with per-symbol overrides."""
    base = {**LS_CFG}
    # Remove nested override dicts before building params
    for key in list(base.keys()):
        if isinstance(base[key], dict):
            base.pop(key)
    # Apply per-symbol overrides if present
    override = LS_CFG.get(symbol_name, {})
    base.update(override)
    return LiquiditySweepParams(**base)


def build_rsi_params() -> RSIReversionParams:
    return RSIReversionParams(**RSI_CFG)


# ──────────────────────────────────────────────────────────────────
#  DAILY TREND — shared across all symbols
# ──────────────────────────────────────────────────────────────────

def build_daily_trend(df_daily: pd.DataFrame, ma_period: int = 50) -> pd.Series:
    ma    = df_daily["close"].rolling(ma_period).mean()
    trend = (df_daily["close"] > ma)
    trend.index = trend.index.normalize()
    return trend


def apply_daily_trend(df: pd.DataFrame, daily_trend: pd.Series) -> pd.DataFrame:
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    df["trend_daily"] = daily_trend.reindex(
        df.index.normalize(), method="ffill"
    ).values
    return df


# ──────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    symbols    = [s for s in CFG["symbols"] if s["enabled"]]
    all_results = []
    all_pf_ls   = []
    all_pf_rsi  = []
    all_labels  = []

    dfs = {}  # store prepared DataFrames for grid search
    for sym in symbols:
        name   = sym["name"]
        print(f"\n{'═'*62}")
        print(f"  {name}")
        print(f"{'═'*62}")

        # ── 1. OHLCV
        df_1h = fetch_ohlcv(
            sym["ccxt_symbol"], TIMEFRAME, SINCE_DATE, EXCHANGE, CACHE_DIR
        )
        df_1d = fetch_ohlcv(
            sym["ccxt_symbol"], TF_TREND,
            GLOBAL["since_date_trend"], EXCHANGE, CACHE_DIR
        )

        # ── 2. Derivatives
        df_oi = fetch_open_interest(
            sym["bybit_symbol"],
            GLOBAL["since_date_deriv"], TIMEFRAME, CACHE_DIR
        )
        df_fr = fetch_funding_rate(
            sym["binance_symbol"],
            GLOBAL["since_date_deriv"], CACHE_DIR
        )

        # ── 3. Prepare base DataFrame
        daily_trend = build_daily_trend(df_1d, ma_period=LS_CFG["ma_trend_1d"])
        df_bt = df_1h[df_1h.index >= pd.Timestamp(SINCE_DATE, tz="UTC")].copy()
        df_bt = merge_derivatives(df_bt, df_oi, df_fr)
        df_bt = apply_daily_trend(df_bt, daily_trend)

        oi_cov = (df_bt["oi_usdt"] > 0).sum() / len(df_bt) * 100
        print(f"  OI coverage: {oi_cov:.1f}%  |  "
              f"FR range: {df_bt['funding_rate'].min():.4f} → "
              f"{df_bt['funding_rate'].max():.4f}")

        # ── 4. Liquidity Sweep
        print(f"\n  ── Liquidity Sweep OI [{name}]")
        ls_params = build_ls_params(name)
        ls_strat  = LiquiditySweepStrategy(ls_params)
        df_ls     = ls_strat.prepare(df_bt)
        print(f"  Signals: {df_ls['entry'].sum()}")

        pf_ls = run_backtest(df_ls, ls_strat, INIT_CASH, timeframe=TIMEFRAME)
        r_ls  = extract_metrics(pf_ls)
        r_ls.update({"name": "Liquidity Sweep", "symbol": name})
        all_results.append(r_ls)
        all_pf_ls.append(pf_ls)
        all_labels.append(f"LS {name}")

        print(f"  Return: {r_ls['return_%']:.1f}%  Sharpe: {r_ls['sharpe']:.2f}  "
              f"WR: {r_ls['win_rate_%']:.1f}%  Trades: {r_ls['trades']}")

        # ── 5. RSI Reversion
        print(f"\n  ── RSI Mean Reversion [{name}]")
        rsi_params = build_rsi_params()
        rsi_strat  = RSIReversionStrategy(rsi_params)
        df_rsi     = rsi_strat.prepare(df_bt)
        print(f"  Signals: {df_rsi['entry'].sum()}")

        dfs[name] = df_bt  # store for optional grid search

        pf_rsi = run_backtest(df_rsi, rsi_strat, INIT_CASH, timeframe=TIMEFRAME)
        r_rsi  = extract_metrics(pf_rsi)
        r_rsi.update({"name": "RSI Reversion", "symbol": name})
        all_results.append(r_rsi)
        all_pf_rsi.append(pf_rsi)

        # ── 6. Per-symbol equity curve
        plot_equity_curves(
            portfolios  = [pf_ls, pf_rsi],
            labels      = [f"Liquidity Sweep", f"RSI Reversion"],
            df_price    = df_bt,
            init_cash   = INIT_CASH,
            title       = f"Equity Curve — {name} {TIMEFRAME}+{TF_TREND}  |  {SINCE_DATE} → today",
            output_path = str(PLOTS_DIR / f"equity_{name.lower()}.png"),
        )

    # ── 7. Summary
    print_comparison(all_results)

    # ── 8. Multi-symbol comparison (Liquidity Sweep only)
    # ════════════════════════════════════════
    #  GRID SEARCH — per symbol (uncomment to run, takes ~5 min)
    # ════════════════════════════════════════
    # LS_PARAM_GRID = {
    #     "oi_drop_pct":  [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0],
    #     "funding_min":  [0.0, 0.00005, 0.0001, 0.0002],
    #     "take_profit":  [3.0, 4.0, 5.0, 6.0, 8.0],
    #     "atr_mult":     [1.5, 2.0, 2.5, 3.0],
    # }
    # opt_results = {}
    # for sym in symbols:
    #     name = sym["name"]
    #     print(f"\n── Grid Search: Liquidity Sweep [{name}]")
    #     opt_results[name] = optimize_strategy(
    #         dfs[name], LiquiditySweepStrategy,
    #         build_ls_params(name), LS_PARAM_GRID,
    #     )
    #     if not opt_results[name].empty:
    #         ts  = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    #         opt_results[name].to_csv(
    #             RESULTS_DIR / f"grid_{name.lower()}_{ts}.csv", index=False
    #         )

    ls_results = [r for r in all_results if r["name"] == "Liquidity Sweep"]
    plot_multi_symbol(ls_results, output_path=str(PLOTS_DIR / "multi_symbol.png"))

    # ── 9. Persist results to CSV
    import json
    from datetime import datetime as dt
    run_ts  = dt.now().strftime("%Y%m%d_%H%M%S")
    results_df = pd.DataFrame(all_results)
    csv_path   = RESULTS_DIR / f"run_{run_ts}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"  [RESULTS] Saved: {csv_path}")
    print(f"  [DONE] Plots: {PLOTS_DIR}/  |  Results: {RESULTS_DIR}/")
