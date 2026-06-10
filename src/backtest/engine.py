"""
Backtest engine.
Runs vectorbt backtests, grid search optimization, and walk-forward validation.
Grid search and walk-forward inner loops are parallelized with joblib.
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
import itertools
from typing import Optional
from joblib import Parallel, delayed, cpu_count
from ..strategies.base import BaseStrategy, StrategyParams


# ──────────────────────────────────────────────────────────────────
#  CORE BACKTEST
# ──────────────────────────────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    strategy: BaseStrategy,
    init_cash: float = 10_000,
    fees: float      = 0.001,
    slippage: float  = 0.0005,
    timeframe: str   = "1h",
) -> vbt.Portfolio:
    """
    Run a single backtest for a prepared DataFrame.
    df must already have 'entry' and 'atr_sl' columns from strategy.prepare().
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
        slippage  = slippage,
        size      = np.inf,   # deploy all available cash per entry (vbt default)
        freq      = timeframe,
    )


def buy_hold_metrics(df: pd.DataFrame, timeframe: str = "1h") -> dict:
    """
    Buy & hold benchmark over the exact backtested period.
    Buys the close on the first bar, holds to the last bar.
    Returns total return, Sharpe and max drawdown for fair comparison.
    """
    close = df["close"].dropna()
    bh_return = (close.iloc[-1] / close.iloc[0] - 1) * 100

    rets = close.pct_change().dropna()
    ann = {"1h": 24 * 365, "4h": 6 * 365, "1d": 365}.get(timeframe, 24 * 365)
    sharpe = (rets.mean() / rets.std() * (ann ** 0.5)) if rets.std() > 0 else float("nan")

    cummax = close.cummax()
    max_dd = ((close - cummax) / cummax).min() * 100

    return {
        "bh_return_%": round(bh_return, 2),
        "bh_sharpe":   round(sharpe, 3),
        "bh_max_dd_%": round(max_dd, 2),
    }


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


# ──────────────────────────────────────────────────────────────────
#  GRID SEARCH — parallelized with joblib
# ──────────────────────────────────────────────────────────────────

def _eval_combo(
    combo: tuple,
    keys: list,
    base_params: StrategyParams,
    strategy_class: type,
    df: pd.DataFrame,
    min_trades: int,
    min_wr: float,
    max_dd: float,
    init_cash: float,
    timeframe: str,
    fees: float     = 0.001,
    slippage: float = 0.0005,
) -> Optional[dict]:
    """
    Evaluate a single parameter combination.
    Returns a result dict or None if it doesn't pass filters.
    Designed to run in parallel — no shared mutable state.
    """
    overrides   = dict(zip(keys, combo))
    params_dict = {**vars(base_params), **overrides}
    params      = base_params.__class__(**params_dict)
    strategy    = strategy_class(params)
    df_sig      = strategy.prepare(df)

    if df_sig["entry"].sum() < min_trades:
        return None

    try:
        pf     = run_backtest(df_sig, strategy, init_cash,
                              fees=fees, slippage=slippage, timeframe=timeframe)
        stats  = pf.stats()
        trades = stats.get("Total Trades", 0)
        wr     = stats.get("Win Rate [%]", 0)
        dd     = pf.max_drawdown() * 100

        if trades < min_trades or dd < max_dd or wr < min_wr:
            return None

        return {
            **overrides,
            "trades":   trades,
            "win_rate": round(wr, 1),
            "sharpe":   round(pf.sharpe_ratio(), 3),
            "return_%": round(pf.total_return() * 100, 2),
            "max_dd_%": round(dd, 2),
            "pf":       round(stats.get("Profit Factor", 0), 2),
        }
    except Exception:
        return None


def optimize_strategy(
    df: pd.DataFrame,
    strategy_class: type,
    base_params: StrategyParams,
    param_grid: dict,
    min_trades: int  = 20,
    min_wr: float    = 45.0,
    max_dd: float    = -30.0,
    init_cash: float = 10_000,
    timeframe: str   = "1h",
    top_n: int       = 10,
    n_jobs: int      = -1,
    fees: float      = 0.001,
    slippage: float  = 0.0005,
) -> pd.DataFrame:
    """
    Parallelized grid search for any strategy.

    Parameters
    ----------
    df             : prepared DataFrame (with trend_daily, oi, etc.)
    strategy_class : uninstantiated strategy class
    base_params    : default params object used as base for overrides
    param_grid     : {param_name: [values]} — all combinations are tested
    min_trades     : discard combos with fewer trades than this
    min_wr         : discard combos with win rate below this (%)
    max_dd         : discard combos with drawdown worse than this (%)
    top_n          : number of top results to print
    n_jobs         : parallel workers (-1 = all CPU cores)
    """
    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))
    cores  = cpu_count() if n_jobs == -1 else n_jobs

    print(f"  [OPT] {len(combos)} combinations on {cores} cores "
          f"(min {min_trades} trades, WR>{min_wr}%, DD>{max_dd}%)...")

    raw = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_eval_combo)(
            combo, keys, base_params, strategy_class,
            df, min_trades, min_wr, max_dd, init_cash, timeframe,
            fees, slippage,
        )
        for combo in combos
    )

    results = [r for r in raw if r is not None]

    if not results:
        print(f"  [OPT] No combinations passed quality filters.")
        return pd.DataFrame()

    df_res = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    print(f"  [OPT] {len(results)} valid combinations. Top {top_n}:")
    print(df_res.head(top_n).to_string(index=False))
    return df_res


# ──────────────────────────────────────────────────────────────────
#  WALK-FORWARD VALIDATION — parallelized inner loop
# ──────────────────────────────────────────────────────────────────

def _eval_window(
    window_n: int,
    cursor,
    train_end,
    test_end,
    df: pd.DataFrame,
    strategy_class: type,
    base_params: StrategyParams,
    param_grid: dict,
    min_trades: int,
    init_cash: float,
    timeframe: str,
    fees: float         = 0.001,
    slippage: float     = 0.0005,
    train_max_dd: float = -35.0,
    train_min_wr: float = 40.0,
) -> Optional[dict]:
    """
    Evaluate a single walk-forward window.
    Optimizes on train set, evaluates on test set.
    Returns result dict or None if no valid params found.
    """
    from dateutil.relativedelta import relativedelta

    # Normalize datetimes to UTC. Accepts tz-aware or tz-naive inputs.
    def _to_utc(ts):
        t = pd.Timestamp(ts)
        # pd.Timestamp(..., tz=...) errors if ts already has tzinfo
        if t.tz is None:
            return t.tz_localize("UTC")
        return t.tz_convert("UTC")

    ts_cursor    = _to_utc(cursor)
    ts_train_end = _to_utc(train_end)
    ts_test_end  = _to_utc(test_end)

    df_train = df[(df.index >= ts_cursor) & (df.index < ts_train_end)]
    df_test  = df[(df.index >= ts_train_end) & (df.index < ts_test_end)]

    keys   = list(param_grid.keys())
    values = list(param_grid.values())

    # ── Optimize on train (sequential inside the window — already parallel at window level)
    best_sharpe = -np.inf
    best_params = None

    for combo in itertools.product(*values):
        overrides   = dict(zip(keys, combo))
        params_dict = {**vars(base_params), **overrides}
        params      = base_params.__class__(**params_dict)
        strategy    = strategy_class(params)
        df_sig      = strategy.prepare(df_train)

        if df_sig["entry"].sum() < min_trades:
            continue
        try:
            pf  = run_backtest(df_sig, strategy, init_cash,
                               fees=fees, slippage=slippage, timeframe=timeframe)
            sh  = pf.sharpe_ratio()
            dd  = pf.max_drawdown() * 100
            wr  = pf.stats().get("Win Rate [%]", 0)
            trades = pf.stats().get("Total Trades", 0)

            if trades < min_trades or dd < train_max_dd or wr < train_min_wr:
                continue
            if sh > best_sharpe:
                best_sharpe = sh
                best_params = params
        except Exception:
            continue

    if best_params is None:
        return None

    # ── Evaluate on test set
    strategy  = strategy_class(best_params)
    df_test_p = strategy.prepare(df_test)

    try:
        pf_test = run_backtest(df_test_p, strategy, init_cash,
                               fees=fees, slippage=slippage, timeframe=timeframe)
        m = extract_metrics(pf_test)
        return {
            "window":      window_n,
            "train_start": cursor.date(),
            "train_end":   train_end.date(),
            "test_start":  train_end.date(),
            "test_end":    test_end.date(),
            "sharpe_is":   round(float(best_sharpe), 3),
            "best_params": str({k: getattr(best_params, k) for k in param_grid}),
            **m,
        }
    except Exception:
        return None


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
    n_jobs: int       = -1,
    fees: float         = 0.001,
    slippage: float     = 0.0005,
    train_max_dd: float = -35.0,
    train_min_wr: float = 40.0,
) -> pd.DataFrame:
    """
    Parallelized walk-forward validation.

    Splits the full period into rolling train/test windows.
    Windows are evaluated in parallel (each window on its own core).
    For each window: optimizes params on train, evaluates on test.

    Parameters
    ----------
    df              : full DataFrame (with trend_daily, oi, etc.)
    strategy_class  : uninstantiated strategy class
    base_params     : default params (base for grid overrides)
    param_grid      : {param_name: [values]} to optimize per window
    train_months    : size of each training window in months
    test_months     : size of each test window in months
    min_trades      : minimum trades per window to be considered valid
    n_jobs          : parallel workers (-1 = all CPU cores)
    """
    from dateutil.relativedelta import relativedelta

    start = df.index[0].to_pydatetime()
    end   = df.index[-1].to_pydatetime()
    cores = cpu_count() if n_jobs == -1 else n_jobs

    # Build list of windows
    windows = []
    cursor  = start
    n       = 0
    while True:
        train_end = cursor + relativedelta(months=train_months)
        test_end  = train_end + relativedelta(months=test_months)
        if test_end > end:
            break
        n += 1
        windows.append((n, cursor, train_end, test_end))
        cursor += relativedelta(months=test_months)

    if not windows:
        print("  [WF] Not enough data for walk-forward windows.")
        return pd.DataFrame()

    print(f"  [WF] {len(windows)} windows × "
          f"(train {train_months}m / test {test_months}m) on {cores} cores...")

    raw = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_eval_window)(
            n, cursor, train_end, test_end,
            df, strategy_class, base_params, param_grid,
            min_trades, init_cash, timeframe,
            fees, slippage, train_max_dd, train_min_wr,
        )
        for n, cursor, train_end, test_end in windows
    )

    results = [r for r in raw if r is not None]

    if not results:
        print("  [WF] No valid windows found.")
        empty = pd.DataFrame()
        empty.attrs["n_windows_total"] = len(windows)
        return empty

    df_wf = pd.DataFrame(results).sort_values("window")
    df_wf.attrs["n_windows_total"] = len(windows)

    print(f"\n  [WF] Results ({len(results)}/{len(windows)} windows valid):")
    cols = ["window", "test_start", "test_end",
            "return_%", "sharpe", "max_dd_%", "win_rate_%", "trades"]
    print(df_wf[cols].to_string(index=False))

    profitable = (df_wf["return_%"] > 0).sum()
    print(f"\n  [WF] Summary: {profitable}/{len(df_wf)} profitable windows | "
          f"avg return {df_wf['return_%'].mean():.1f}% | "
          f"avg Sharpe {df_wf['sharpe'].mean():.2f}")

    return df_wf
