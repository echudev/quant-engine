"""
Reporting utilities: console reports and comparison tables.
"""

import vectorbt as vbt


def print_report(
    portfolio: vbt.Portfolio,
    name: str,
    symbol: str,
    timeframe: str,
    tf_trend: str,
    since_date: str,
    init_cash: float,
) -> dict:
    """Print performance report and return metrics dict."""
    stats         = portfolio.stats()
    total_return  = portfolio.total_return() * 100
    sharpe        = portfolio.sharpe_ratio()
    max_dd        = portfolio.max_drawdown() * 100
    win_rate      = stats.get("Win Rate [%]", float("nan"))
    total_trades  = stats.get("Total Trades", 0)
    profit_factor = stats.get("Profit Factor", float("nan"))

    print(f"\n{'='*62}")
    print(f"  {name}  [{symbol}]")
    print(f"{'='*62}")
    print(f"  {symbol} {timeframe}+{tf_trend}  |  {since_date} → today")
    print(f"  Capital ${init_cash:,.0f}  |  TP {stats.get('take_profit', '?')}%  "
          f"|  SL ATR-dynamic")
    print(f"{'-'*62}")
    print(f"  Return       : {total_return:.2f}%")
    print(f"  Sharpe       : {sharpe:.2f}")
    print(f"  Max DD       : {max_dd:.2f}%")
    print(f"  Trades       : {total_trades}")
    print(f"  Win rate     : {win_rate:.1f}%")
    print(f"  Prof. factor : {profit_factor:.2f}")
    print(f"{'='*62}\n")

    return {
        "name": name, "symbol": symbol,
        "return_%": total_return, "sharpe": sharpe,
        "max_dd_%": max_dd, "trades": total_trades,
        "win_rate_%": win_rate, "prof_factor": profit_factor,
    }


def print_comparison(results: list) -> None:
    """Print a comparison table of multiple strategy/symbol results."""
    print(f"\n{'━'*72}")
    print(f"  COMPARISON")
    print(f"{'━'*72}")
    print(f"  {'Strategy':<22} {'Symbol':<6} {'Return':>8} {'B&H':>8} "
          f"{'Alpha':>8} {'Sharpe':>7} {'MaxDD':>8} {'WR':>7} {'Trades':>7}")
    print(f"  {'-'*86}")
    for r in results:
        bh    = r.get("bh_%", float("nan"))
        alpha = r.get("alpha_%", float("nan"))
        print(f"  {r['name']:<22} {r['symbol']:<6} {r['return_%']:>7.1f}% "
              f"{bh:>7.1f}% {alpha:>7.1f}% "
              f"{r['sharpe']:>7.2f} {r['max_dd_%']:>7.1f}% "
              f"{r['win_rate_%']:>6.1f}% {r['trades']:>7}")
    print(f"{'━'*72}\n")


def print_walkforward_summary(wf_results) -> None:
    """Print walk-forward validation summary."""
    if wf_results.empty:
        print("No walk-forward results to display.")
        return

    print(f"\n{'━'*72}")
    print(f"  WALK-FORWARD VALIDATION SUMMARY")
    print(f"{'━'*72}")
    print(f"  Windows    : {len(wf_results)}")
    print(f"  Profitable : {(wf_results['return_%'] > 0).sum()} / {len(wf_results)}")
    print(f"  Avg return : {wf_results['return_%'].mean():.1f}%")
    print(f"  Avg Sharpe : {wf_results['sharpe'].mean():.2f}")
    print(f"  Avg Max DD : {wf_results['max_dd_%'].mean():.1f}%")
    print(f"\n  Per-window results:")
    cols = ["window", "test_start", "test_end", "return_%",
            "sharpe", "max_dd_%", "win_rate_%", "trades"]
    print(wf_results[cols].to_string(index=False))
    print(f"{'━'*72}\n")
