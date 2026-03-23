"""
Backtest engine.
Runs vectorbt backtests and walk-forward validation.
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
import itertools
from typing import Optional
from ..strategies.base import BaseStrategy, StrategyParams


def run_backtest(
    df: pd.DataFrame,
    strategy: BaseStrategy,
    init_cash: float = 10_000,
    fees: float = 0.001,
    timeframe: str = "1h",
) -> vbt.Portfolio:
    """
    Run a single backtest for a strategy on a prepared DataFrame.

    df must already have 'entry', 'atr_sl' columns from strategy.prepare().
    """
    exits = (
        df[strategy.exit_column]
        if strategy.exit_column and strategy.exit_column in df.columns
        else pd.Series(False, index=df.index)
    )

    return vbt.Portfolio.from_signals(
        close     = df["close"],
        entries   = df["entry"],
        exits     = exits,
        sl_stop   = df["atr_sl"],
        tp_stop   = strategy.params.take_profit / 100,
        init_cash = init_cash,
        fees      = fees,
        size      = 1.0,
        freq      = timeframe,
    )


def extract_metrics(portfolio: vbt.Portfolio) -> dict:
    """Extract key performance metrics from a vectorbt Portfolio."""
    stats = portfolio.stats()
    return {
        "return_%":    round(portfolio.total_return() * 100, 2),
        "sharpe":      round(portfolio.sharpe_ratio(), 3),
        "max_dd_%":    round(portfolio.max_drawdown() * 100, 2),
        "trades":      stats.get("Total Trades", 0),
        "win_rate_%":  round(stats.get("Win Rate [%]", float("nan")), 1),
        "prof_factor": round(stats.get("Profit Factor", float("nan")), 2),
    }


def walk_forward(
    df: pd.DataFrame,
    strategy_class: type,
    base_params: StrategyParams,
    param_grid: dict,
    train_months: int = 12,
    test_months: int  = 6,
    min_trades: int   = 10,
    init_cash: float  = 10_000,
    timeframe: str    = "1h",
) -> pd.DataFrame:
    """
    Walk-forward validation.

    Splits the full period into overlapping train/test windows.
    For each window: optimizes params on train, evaluates on test.
    Returns a DataFrame with out-of-sample results per window.

    Parameters
    ----------
    df              : full prepared DataFrame (with trend_daily, oi columns, etc.)
    strategy_class  : uninstantiated strategy class
    base_params     : default params (used as base for grid search)
    param_grid      : dict of {param_name: [values]} to optimize
    train_months    : months in each training window
    test_months     : months in each test window
    min_trades      : minimum trades to consider a result valid
    """
    from dateutil.relativedelta import relativedelta

    start = df.index[0].to_pydatetime()
    end   = df.index[-1].to_pydatetime()
    results = []
    window_n = 0

    cursor = start
    while True:
        train_end = cursor + relativedelta(months=train_months)
        test_end  = train_end + relativedelta(months=test_months)

        if test_end > end:
            break

        df_train = df[(df.index >= pd.Timestamp(cursor,   tz="UTC")) &
                      (df.index <  pd.Timestamp(train_end, tz="UTC"))]
        df_test  = df[(df.index >= pd.Timestamp(train_end, tz="UTC")) &
                      (df.index <  pd.Timestamp(test_end,  tz="UTC"))]

        window_n += 1
        print(f"  [WF] Window {window_n}: "
              f"train {cursor.date()}→{train_end.date()} | "
              f"test {train_end.date()}→{test_end.date()}")

        # ── Optimize on train set
        best_sharpe = -np.inf
        best_params = None

        keys   = list(param_grid.keys())
        values = list(param_grid.values())

        for combo in itertools.product(*values):
            params_dict = dict(zip(keys, combo))
            params = base_params.__class__(**{
                **vars(base_params), **params_dict
            })
            strategy = strategy_class(params)
            df_prepared = strategy.prepare(df_train)

            if df_prepared["entry"].sum() < min_trades:
                continue
            try:
                pf  = run_backtest(df_prepared, strategy, init_cash, timeframe=timeframe)
                sh  = pf.sharpe_ratio()
                dd  = pf.max_drawdown() * 100
                wr  = pf.stats().get("Win Rate [%]", 0)
                if pf.stats().get("Total Trades", 0) < min_trades:
                    continue
                if dd < -35 or wr < 40:
                    continue
                if sh > best_sharpe:
                    best_sharpe = sh
                    best_params = params
            except Exception:
                continue

        if best_params is None:
            print(f"  [WF] Window {window_n}: no valid params found on train set.")
            cursor += relativedelta(months=test_months)
            continue

        # ── Evaluate on test set with best params
        strategy  = strategy_class(best_params)
        df_test_p = strategy.prepare(df_test)

        try:
            pf_test = run_backtest(df_test_p, strategy, init_cash, timeframe=timeframe)
            m = extract_metrics(pf_test)
            results.append({
                "window":       window_n,
                "train_start":  cursor.date(),
                "train_end":    train_end.date(),
                "test_start":   train_end.date(),
                "test_end":     test_end.date(),
                "best_params":  str(vars(best_params)),
                **m,
            })
            print(f"  [WF] Window {window_n} test: "
                  f"ret={m['return_%']:.1f}% sharpe={m['sharpe']:.2f} "
                  f"trades={m['trades']}")
        except Exception as e:
            print(f"  [WF] Window {window_n} test failed: {e}")

        cursor += relativedelta(months=test_months)

    if not results:
        print("  [WF] No valid windows found.")
        return pd.DataFrame()

    return pd.DataFrame(results)


def optimize_strategy(
    df: pd.DataFrame,
    strategy_class: type,
    base_params,
    param_grid: dict,
    min_trades: int   = 20,
    min_wr: float     = 45.0,
    max_dd: float     = -30.0,
    init_cash: float  = 10_000,
    timeframe: str    = "1h",
    top_n: int        = 10,
) -> pd.DataFrame:
    """
    Grid search optimization for any strategy.

    Tries all combinations in param_grid, runs a backtest for each,
    filters by quality thresholds, returns top_n sorted by Sharpe.

    Parameters
    ----------
    df            : prepared DataFrame with trend_daily and derivatives
    strategy_class: uninstantiated strategy class
    base_params   : default params object (used as base for overrides)
    param_grid    : dict of {param_name: [values_to_try]}
    min_trades    : discard combos with fewer trades
    min_wr        : discard combos with win rate below this (%)
    max_dd        : discard combos with drawdown worse than this (%)
    top_n         : how many results to return
    """
    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))

    print(f"  [OPT] {len(combos)} combinations (min {min_trades} trades, "
          f"WR>{min_wr}%, DD>{max_dd}%)...")

    results = []

    for combo in combos:
        overrides   = dict(zip(keys, combo))
        params_dict = {**vars(base_params), **overrides}
        params      = base_params.__class__(**params_dict)
        strategy    = strategy_class(params)
        df_sig      = strategy.prepare(df)

        if df_sig["entry"].sum() < min_trades:
            continue

        try:
            pf     = run_backtest(df_sig, strategy, init_cash, timeframe=timeframe)
            stats  = pf.stats()
            trades = stats.get("Total Trades", 0)
            wr     = stats.get("Win Rate [%]", 0)
            dd     = pf.max_drawdown() * 100

            if trades < min_trades or dd < max_dd or wr < min_wr:
                continue

            row = {**overrides}
            row.update({
                "trades":    trades,
                "win_rate":  round(wr, 1),
                "sharpe":    round(pf.sharpe_ratio(), 3),
                "return_%":  round(pf.total_return() * 100, 2),
                "max_dd_%":  round(dd, 2),
                "pf":        round(stats.get("Profit Factor", 0), 2),
            })
            results.append(row)
        except Exception:
            continue

    if not results:
        print(f"  [OPT] No combinations passed quality filters.")
        return pd.DataFrame()

    df_res = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    print(f"  [OPT] {len(results)} valid combinations. Top {top_n}:")
    print(df_res.head(top_n).to_string(index=False))
    return df_res
