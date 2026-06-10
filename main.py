"""
Liquidity Sweep Engine — main entry point.

Runs Liquidity Sweep OI strategy across multiple symbols.
Config: config/settings.toml
"""

import tomllib
import pandas as pd
from pathlib import Path

from src.data.fetcher import (
    fetch_ohlcv, fetch_open_interest, fetch_funding_rate, merge_derivatives
)
from src.data.preparation import build_daily_trend, apply_daily_trend
from src.strategies.liquidity_sweep import LiquiditySweepStrategy, LiquiditySweepParams
from src.strategies.rsi_reversion import RSIReversionStrategy, RSIReversionParams
from src.strategies.funding_flush import FundingFlushStrategy, FundingFlushParams
from src.backtest.engine import (
    run_backtest, extract_metrics, buy_hold_metrics, optimize_strategy, walk_forward
)
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
FF_CFG  = CFG["strategy"]["funding_flush"]


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


def build_ff_params(symbol_name: str) -> FundingFlushParams:
    """Build FundingFlushParams with per-symbol overrides."""
    base = {**FF_CFG}
    for key in list(base.keys()):
        if isinstance(base[key], dict):
            base.pop(key)
    override = FF_CFG.get(symbol_name, {})
    base.update(override)
    return FundingFlushParams(**base)


# ──────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    symbols    = [s for s in CFG["symbols"] if s["enabled"]]
    all_results = []
    all_pf_ls   = []
    all_pf_rsi  = []
    all_pf_ff   = []
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

        # ── Buy & Hold benchmark over the exact same period
        bh = buy_hold_metrics(df_bt, timeframe=TIMEFRAME)
        print(f"  Buy & Hold:  {bh['bh_return_%']:.1f}%  "
              f"Sharpe: {bh['bh_sharpe']:.2f}  MaxDD: {bh['bh_max_dd_%']:.1f}%")

        # ── 4. Liquidity Sweep
        print(f"\n  ── Liquidity Sweep OI [{name}]")
        ls_params = build_ls_params(name)
        ls_strat  = LiquiditySweepStrategy(ls_params)
        df_ls     = ls_strat.prepare(df_bt)
        print(f"  Signals: {df_ls['entry'].sum()}")

        pf_ls = run_backtest(df_ls, ls_strat, INIT_CASH, timeframe=TIMEFRAME)
        r_ls  = extract_metrics(pf_ls)
        r_ls.update({"name": "Liquidity Sweep", "symbol": name,
                     "bh_%": bh["bh_return_%"],
                     "alpha_%": round(r_ls["return_%"] - bh["bh_return_%"], 2)})
        all_results.append(r_ls)
        all_pf_ls.append(pf_ls)
        all_labels.append(f"LS {name}")

        print(f"  Return: {r_ls['return_%']:.1f}%  Sharpe: {r_ls['sharpe']:.2f}  "
              f"WR: {r_ls['win_rate_%']:.1f}%  Trades: {r_ls['trades']}")

        # ── 5. RSI Reversion (v2 — with volume confirmation)
        print(f"\n  ── RSI Mean Reversion [{name}]")
        rsi_params = build_rsi_params()
        rsi_strat  = RSIReversionStrategy(rsi_params)
        df_rsi     = rsi_strat.prepare(df_bt)
        print(f"  Signals: {df_rsi['entry'].sum()}")

        dfs[name] = df_bt  # store for optional grid search

        pf_rsi = run_backtest(df_rsi, rsi_strat, INIT_CASH, timeframe=TIMEFRAME)
        r_rsi  = extract_metrics(pf_rsi)
        r_rsi.update({"name": "RSI Reversion", "symbol": name,
                      "bh_%": bh["bh_return_%"],
                      "alpha_%": round(r_rsi["return_%"] - bh["bh_return_%"], 2)})
        all_results.append(r_rsi)
        all_pf_rsi.append(pf_rsi)

        print(f"  Return: {r_rsi['return_%']:.1f}%  Sharpe: {r_rsi['sharpe']:.2f}  "
              f"WR: {r_rsi['win_rate_%']:.1f}%  Trades: {r_rsi['trades']}")

        # ── 6. Funding Flush
        print(f"\n  ── Funding Rate Flush [{name}]")
        ff_params = build_ff_params(name)
        ff_strat  = FundingFlushStrategy(ff_params)
        df_ff     = ff_strat.prepare(df_bt)
        print(f"  Signals: {df_ff['entry'].sum()}")

        pf_ff = run_backtest(df_ff, ff_strat, INIT_CASH, timeframe=TIMEFRAME)
        r_ff  = extract_metrics(pf_ff)
        r_ff.update({"name": "Funding Flush", "symbol": name,
                     "bh_%": bh["bh_return_%"],
                     "alpha_%": round(r_ff["return_%"] - bh["bh_return_%"], 2)})
        all_results.append(r_ff)
        all_pf_ff.append(pf_ff)

        print(f"  Return: {r_ff['return_%']:.1f}%  Sharpe: {r_ff['sharpe']:.2f}  "
              f"WR: {r_ff['win_rate_%']:.1f}%  Trades: {r_ff['trades']}")

        # ── 7. Per-symbol equity curve
        plot_equity_curves(
            portfolios  = [pf_ls, pf_rsi, pf_ff],
            labels      = ["Liquidity Sweep", "RSI Reversion", "Funding Flush"],
            df_price    = df_bt,
            init_cash   = INIT_CASH,
            title       = f"Equity Curve — {name} {TIMEFRAME}+{TF_TREND}  |  {SINCE_DATE} → today",
            output_path = str(PLOTS_DIR / f"equity_{name.lower()}.png"),
        )

    # ── 7. Summary
    print_comparison(all_results)

    ls_results = [r for r in all_results if r["name"] == "Liquidity Sweep"]
    plot_multi_symbol(ls_results, output_path=str(PLOTS_DIR / "multi_symbol.png"))

    from datetime import datetime as dt
    run_ts = dt.now().strftime("%Y%m%d_%H%M%S")

    # ════════════════════════════════════════
    #  WALK-FORWARD: Liquidity Sweep [BTC]
    #
    #  BTC LS params were hand-tuned on the full history ("Perfil A — BTC óptimo").
    #  With only 29 total trades, in-sample optimization risk is real.
    #  WF validates that the edge holds across different market regimes,
    #  not just the 2023-2026 bull run.
    #  min_trades=5: BTC LS is selective (~7-8 trades per 12-month window).
    # ════════════════════════════════════════
    WF_BTC_LS_GRID = {
        "oi_drop_pct": [1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
        "funding_min": [0.00001, 0.00002, 0.00005, 0.0001],
        "take_profit": [4.0, 6.0, 8.0],
        "atr_mult":    [2.0, 2.5, 3.0, 3.5],
    }
    print("\n── Walk-Forward: Liquidity Sweep [BTC]")
    wf_btc_ls = walk_forward(
        df             = dfs["BTC"],
        strategy_class = LiquiditySweepStrategy,
        base_params    = build_ls_params("BTC"),
        param_grid     = WF_BTC_LS_GRID,
        train_months   = 12,
        test_months    = 6,
        min_trades     = 5,
        init_cash      = INIT_CASH,
        timeframe      = TIMEFRAME,
        n_jobs         = -1,
    )
    if not wf_btc_ls.empty:
        wf_btc_ls.to_csv(RESULTS_DIR / f"wf_BTC_LS_{run_ts}.csv", index=False)

    # ════════════════════════════════════════
    #  WALK-FORWARD: Funding Flush [BTC]
    #
    #  BTC FF has 22 trades, Sharpe 1.28 in-sample — but also unvalidated.
    #  Same concern as LS: params tuned on full history.
    #  If 3+/4 windows are profitable, the edge is real and BTC FF should
    #  be the primary BTC strategy (replacing the curve-fitted LS).
    # ════════════════════════════════════════
    WF_BTC_FF_GRID = {
        "funding_pct_rank": [40.0, 45.0, 50.0, 55.0, 60.0, 65.0],
        "oi_drop_pct":      [1.0, 1.5, 2.0, 3.0, 4.0],
        "vol_mult":         [1.0, 1.3, 1.5],
        "take_profit":      [4.0, 5.0, 6.0, 8.0],
    }
    print("\n── Walk-Forward: Funding Flush [BTC]")
    wf_btc_ff = walk_forward(
        df             = dfs["BTC"],
        strategy_class = FundingFlushStrategy,
        base_params    = build_ff_params("BTC"),
        param_grid     = WF_BTC_FF_GRID,
        train_months   = 12,
        test_months    = 6,
        min_trades     = 5,
        init_cash      = INIT_CASH,
        timeframe      = TIMEFRAME,
        n_jobs         = -1,
    )
    if not wf_btc_ff.empty:
        wf_btc_ff.to_csv(RESULTS_DIR / f"wf_BTC_FF_{run_ts}.csv", index=False)

    # ════════════════════════════════════════
    #  GRID SEARCH: Funding Flush [ETH]
    #
    #  FF ETH has 35 trades at Sharpe 1.05 with zero optimization.
    #  Find the best full-history params (use WF below to validate OOS).
    #  320 combinations: fast on parallel cores.
    # ════════════════════════════════════════
    FF_ETH_GRID = {
        "funding_pct_rank": [55.0, 60.0, 65.0, 70.0, 75.0],
        "oi_drop_pct":      [2.0, 3.0, 4.0, 5.0],
        "vol_mult":         [1.0, 1.3, 1.5, 2.0],
        "take_profit":      [4.0, 5.0, 6.0, 7.0],
    }
    print("\n── Grid Search: Funding Flush [ETH]")
    gs_ff_eth = optimize_strategy(
        df             = dfs["ETH"],
        strategy_class = FundingFlushStrategy,
        base_params    = build_ff_params("ETH"),
        param_grid     = FF_ETH_GRID,
        min_trades     = 15,
        min_wr         = 50.0,
        max_dd         = -20.0,
        init_cash      = INIT_CASH,
        timeframe      = TIMEFRAME,
        top_n          = 10,
    )
    if not gs_ff_eth.empty:
        gs_ff_eth.to_csv(RESULTS_DIR / f"gs_ff_eth_{run_ts}.csv", index=False)

    # ════════════════════════════════════════
    #  WALK-FORWARD: Funding Flush [ETH]
    #
    #  Validates FF ETH edge out-of-sample across 4 rolling windows.
    #  35 trades over 3 years → ~8-9 per 6-month test window.
    #  Uses same grid as above for inner per-window optimization.
    # ════════════════════════════════════════
    WF_FF_ETH_GRID = {
        "funding_pct_rank": [55.0, 60.0, 65.0, 70.0, 75.0],
        "oi_drop_pct":      [2.0, 3.0, 4.0, 5.0],
        "vol_mult":         [1.0, 1.3, 1.5, 2.0],
        "take_profit":      [4.0, 5.0, 6.0, 7.0],
    }
    print("\n── Walk-Forward: Funding Flush [ETH]")
    wf_ff_eth = walk_forward(
        df             = dfs["ETH"],
        strategy_class = FundingFlushStrategy,
        base_params    = build_ff_params("ETH"),
        param_grid     = WF_FF_ETH_GRID,
        train_months   = 12,
        test_months    = 6,
        min_trades     = 8,
        init_cash      = INIT_CASH,
        timeframe      = TIMEFRAME,
        n_jobs         = -1,
    )
    if not wf_ff_eth.empty:
        wf_ff_eth.to_csv(RESULTS_DIR / f"wf_FF_ETH_{run_ts}.csv", index=False)

    # ════════════════════════════════════════
    #  PORTFOLIO COMBINADO (all enabled symbols, LS + FF)
    # ════════════════════════════════════════
    sym_names = [s["name"] for s in symbols]
    combined_pf  = all_pf_ls + all_pf_ff
    combined_lbl = [f"LS {n}" for n in sym_names] + [f"FF {n}" for n in sym_names]

    if combined_pf:
        plot_equity_curves(
            portfolios  = combined_pf,
            labels      = combined_lbl,
            df_price    = dfs[sym_names[0]],
            init_cash   = INIT_CASH,
            title       = f"Portfolio {'+'.join(sym_names)}  LS+FF  |  {SINCE_DATE} → today",
            output_path = str(PLOTS_DIR / "portfolio_combined.png"),
        )

    # ── Persist run summary
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_DIR / f"run_{run_ts}.csv", index=False)
    print(f"  [RESULTS] Saved: {RESULTS_DIR}/")
    print(f"  [DONE] Plots: {PLOTS_DIR}/  |  Results: {RESULTS_DIR}/")
