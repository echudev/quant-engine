"""
Visualization module.
Equity curves, drawdown panels, multi-symbol comparisons.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import vectorbt as vbt
import pandas as pd
from pathlib import Path


DARK_BG   = "#0f1117"
GRID_COL  = "#1f2937"
SPINE_COL = "#374151"
TEXT_COL  = "#e5e7eb"
MUTED_COL = "#9ca3af"
COLORS    = ["#3b82f6", "#f59e0b", "#10b981", "#ef4444",
             "#8b5cf6", "#06b6d4", "#f97316", "#ec4899"]


def _style_ax(ax) -> None:
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=MUTED_COL)
    ax.xaxis.label.set_color(MUTED_COL)
    ax.yaxis.label.set_color(MUTED_COL)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_COL)


def plot_equity_curves(
    portfolios: list,
    labels: list,
    df_price: pd.DataFrame,
    init_cash: float,
    title: str,
    output_path: str = "equity_curve.png",
) -> None:
    """
    Plot equity curves vs Buy & Hold with drawdown panel.

    Parameters
    ----------
    portfolios  : list of vbt.Portfolio objects
    labels      : list of strategy names
    df_price    : DataFrame with 'close' column (OHLCV)
    init_cash   : initial capital for B&H normalization
    title       : chart title
    output_path : file path to save PNG
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    fig.patch.set_facecolor(DARK_BG)
    _style_ax(ax1)
    _style_ax(ax2)

    # ── Buy & Hold
    price = df_price["close"].copy()
    price = price[price.index >= portfolios[0].value().index[0]]
    bh    = (price / price.iloc[0]) * init_cash
    ax1.plot(bh.index, bh.values, color="#6b7280",
             linewidth=1, linestyle="--", label="Buy & Hold", alpha=0.7)

    # ── Strategy curves
    for pf, label, color in zip(portfolios, labels, COLORS):
        value = pf.value()
        dd    = pf.drawdown() * 100
        ax1.plot(value.index, value.values,
                 color=color, linewidth=1.5, label=label)
        ax2.fill_between(dd.index, dd.values, 0, color=color, alpha=0.25)
        ax2.plot(dd.index, dd.values, color=color, linewidth=0.8)

    # ── Formatting
    ax1.set_ylabel("Capital (USDT)", fontsize=11)
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    ax1.legend(facecolor="#1f2937", edgecolor=SPINE_COL,
               labelcolor=TEXT_COL, fontsize=10)
    ax1.set_title(title, color=TEXT_COL, fontsize=13, pad=12)
    ax1.grid(axis="y", color=GRID_COL, linewidth=0.5)

    ax2.set_ylabel("Drawdown (%)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax2.grid(axis="y", color=GRID_COL, linewidth=0.5)
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [PLOT] Saved: {output_path}")


def plot_multi_symbol(
    results: list,
    output_path: str = "multi_symbol.png",
) -> None:
    """
    Bar chart comparing returns across symbols.

    Parameters
    ----------
    results     : list of dicts with 'symbol', 'name', 'return_%', 'sharpe'
    output_path : file path to save PNG
    """
    if not results:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(DARK_BG)
    _style_ax(ax1)
    _style_ax(ax2)

    symbols = [r["symbol"] for r in results]
    returns = [r["return_%"] for r in results]
    sharpes = [r["sharpe"] for r in results]
    colors  = [COLORS[i % len(COLORS)] for i in range(len(results))]

    # Returns bar
    bars1 = ax1.bar(symbols, returns, color=colors, alpha=0.85)
    ax1.axhline(0, color=SPINE_COL, linewidth=0.8)
    ax1.set_ylabel("Total Return (%)", fontsize=11)
    ax1.set_title("Return by Symbol", color=TEXT_COL, fontsize=12)
    ax1.bar_label(bars1, fmt="%.1f%%", color=TEXT_COL, fontsize=9, padding=3)

    # Sharpe bar
    bars2 = ax2.bar(symbols, sharpes, color=colors, alpha=0.85)
    ax2.axhline(0, color=SPINE_COL, linewidth=0.8)
    ax2.axhline(1, color="#10b981", linewidth=0.8, linestyle="--", alpha=0.6)
    ax2.set_ylabel("Sharpe Ratio", fontsize=11)
    ax2.set_title("Sharpe by Symbol", color=TEXT_COL, fontsize=12)
    ax2.bar_label(bars2, fmt="%.2f", color=TEXT_COL, fontsize=9, padding=3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [PLOT] Saved: {output_path}")
