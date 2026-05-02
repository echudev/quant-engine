"""Entrypoint del stock screener.

Uso:
    python main_screener.py
    python main_screener.py --refresh                # ignorar cache
    python main_screener.py --tickers AAPL,MSFT,INTC # universo custom (smoke test)
    python main_screener.py --top 50

Lee config/screener.toml para weights y filtros por defecto.
"""
from __future__ import annotations

import argparse
import logging
import sys
import tomllib
from datetime import datetime
from pathlib import Path

from src.screener.runner import run_screener
from src.screener.report import to_csv, to_console, to_markdown


CONFIG_PATH = Path("config/screener.toml")


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "rb") as f:
            return tomllib.load(f)
    return {}


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Stock screener")
    parser.add_argument("--refresh", action="store_true", help="Ignorar cache, re-descargar todo")
    parser.add_argument("--tickers", type=str, default=None, help="Lista custom: AAPL,MSFT,INTC")
    parser.add_argument("--top", type=int, default=None, help="Cantidad de tickers a mostrar")
    parser.add_argument("--no-console", action="store_true", help="No imprimir tabla en consola")
    args = parser.parse_args()

    cfg = load_config()
    weights = cfg.get("weights", None)
    filters_cfg = cfg.get("filters", {})
    output_cfg = cfg.get("output", {})
    exec_cfg = cfg.get("execution", {})

    top_n = args.top or output_cfg.get("top_n", 30)

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        universe_name = "custom"
    else:
        tickers = None
        universe_name = cfg.get("universe", {}).get("source", "sp500")

    df = run_screener(
        universe_name=universe_name,
        custom_tickers=tickers,
        market_cap_min=filters_cfg.get("market_cap_min", 10e9),
        adv_min_usd=filters_cfg.get("adv_min_usd", 50e6),
        history_min_days=filters_cfg.get("history_min_days", 504),
        momentum_min=filters_cfg.get("momentum_min", -0.35),
        sma200_distance_min=filters_cfg.get("sma200_distance_min", -0.30),
        weights=weights,
        n_jobs=exec_cfg.get("n_jobs", 4),
        refresh=args.refresh or exec_cfg.get("refresh", False),
    )

    if df.empty:
        print("Sin resultados.")
        return 1

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_cfg.get("csv_path", "results/screener_{ts}.csv").replace("{ts}", ts)
    md_path = output_cfg.get("markdown_path", "results/screener_{ts}.md").replace("{ts}", ts)

    to_csv(df, csv_path)
    to_markdown(df, md_path, top_n=top_n)

    print_console = output_cfg.get("print_console", True) and not args.no_console
    if print_console:
        to_console(df, top_n=top_n)

    print(f"\nCSV: {csv_path}")
    print(f"MD:  {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
