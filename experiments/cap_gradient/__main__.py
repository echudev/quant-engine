"""
Experimento: gradiente de capitalización para Liquidity Sweep OI.

Lanzamiento completo:   python -m experiments.cap_gradient
Por etapa:              python -m experiments.cap_gradient --stage universe
Etapas: universe → data → lookahead → baseline → backtest → analyze → report

El gate de baseline corre BTC y ETH primero (pasada main): si ninguno de
los dos muestra ventanas OOS rentables tras los fixes del motor, el
experimento se detiene (la hipótesis perdió su ancla) salvo --skip-gate.
"""

import argparse
import random
import sys

import numpy as np

from . import analysis as analysis_mod
from . import report as report_mod
from .config import ensure_dirs, exp_cfg, write_run_meta
from .data import download_and_prepare
from .runner import PASSES, run_pass, save_results
from .universe import build_universe

STAGES = ("universe", "data", "lookahead", "baseline", "backtest",
          "analyze", "report", "all")


def stage_lookahead() -> None:
    from tests.test_no_lookahead import main as lookahead_main
    lookahead_main()


def stage_baseline(skip_gate: bool) -> bool:
    """WF de BTC/ETH con fixes. Devuelve True si el experimento debe seguir."""
    print("\n══ BASELINE GATE: BTC + ETH (pasada main, costos tier 1) ══")
    df = run_pass("main", symbols=["BTC", "ETH"])
    save_results(df)

    valid = df[df["window"].notna()]
    if valid.empty:
        print("\n  [GATE] Sin ventanas válidas en BTC ni ETH tras los fixes.")
        verdict_continue = False
    else:
        prof = (valid["return_oos_%"] > 0).sum()
        for base, g in valid.groupby("base"):
            p = (g["return_oos_%"] > 0).sum()
            print(f"  [GATE] {base}: {p}/{len(g)} ventanas OOS rentables | "
                  f"Sharpe OOS mediano {g['sharpe_oos'].median():.2f}")
        verdict_continue = prof > 0

    if not verdict_continue and not skip_gate:
        print("\n  [GATE] DETENIDO: el edge de referencia no sobrevivió los fixes "
              "del motor. Revisar antes de gastar cómputo en 20 símbolos "
              "(--skip-gate para forzar).")
        return False
    return True


def stage_backtest(symbols: list[str] | None, passes: list[str]) -> None:
    for p in passes:
        print(f"\n══ PASADA: {p} ══")
        save_results(run_pass(p, symbols=symbols))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=STAGES, default="all")
    ap.add_argument("--symbols", help="lista separada por comas (solo backtest/baseline)")
    ap.add_argument("--passes", default=",".join(PASSES),
                    help=f"pasadas a correr (default: {','.join(PASSES)})")
    ap.add_argument("--skip-gate", action="store_true",
                    help="continuar aunque el baseline BTC/ETH falle")
    args = ap.parse_args()

    exp = exp_cfg()
    random.seed(exp["seed"])
    np.random.seed(exp["seed"])
    ensure_dirs()
    write_run_meta({"stage": args.stage})

    symbols = args.symbols.split(",") if args.symbols else None
    passes  = [p for p in args.passes.split(",") if p in PASSES]

    if args.stage in ("universe", "all"):
        build_universe()
    if args.stage in ("data", "all"):
        download_and_prepare()
    if args.stage in ("lookahead", "all"):
        stage_lookahead()
    if args.stage in ("baseline", "all"):
        if not stage_baseline(args.skip_gate):
            sys.exit(2)
        if args.stage == "baseline":
            return
    if args.stage in ("backtest", "all"):
        stage_backtest(symbols, passes)
    if args.stage in ("analyze", "all"):
        analysis_mod.run_analysis()
    if args.stage in ("report", "all"):
        report_mod.generate_report()


if __name__ == "__main__":
    main()
