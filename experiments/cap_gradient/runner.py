"""
Corre el walk-forward del motor (src.backtest.engine.walk_forward) por
símbolo, con la MISMA grilla y ventanas para todos. Tres pasadas:

  main     — costos por tier de OI (pesimistas escalonados)
  stress_t3 — costos tier 3 (2.5x) uniformes para TODOS (test de robustez)
  oi_coins — señal de OI puro (oi_coins en lugar de oi_usdt = OI×precio),
             para medir cuánto del gradiente es ruido de volatilidad.
             No toca la estrategia: se swapea la columna a nivel de datos
             (la estrategia solo usa pct_change, invariante a escala).
"""

import traceback

import pandas as pd
from dateutil.relativedelta import relativedelta

from src.backtest.engine import walk_forward
from src.strategies.liquidity_sweep import LiquiditySweepParams, LiquiditySweepStrategy

from .config import RESULTS_PATH, exp_cfg, load_cfg
from .data import load_qa, prepared_path
from .universe import load_universe

PASSES = ("main", "stress_t3", "oi_coins")


def base_ls_params(cfg: dict) -> LiquiditySweepParams:
    """Params default de LS (sin overrides por símbolo: la grilla explora)."""
    base = {k: v for k, v in cfg["strategy"]["liquidity_sweep"].items()
            if not isinstance(v, dict)}
    return LiquiditySweepParams(**base)


def count_expected_windows(start, end, train_months: int, test_months: int) -> int:
    """Réplica del armado de ventanas de walk_forward(), para registrar
    cuántas ventanas permite la historia de cada símbolo."""
    n, cursor = 0, start.to_pydatetime()
    end = end.to_pydatetime()
    while True:
        train_end = cursor + relativedelta(months=train_months)
        test_end  = train_end + relativedelta(months=test_months)
        if test_end > end:
            return n
        n += 1
        cursor += relativedelta(months=test_months)


def run_pass(pass_name: str, symbols: list[str] | None = None) -> pd.DataFrame:
    assert pass_name in PASSES
    cfg, exp = load_cfg(), exp_cfg()
    g = cfg["global"]
    grid = dict(exp["grid"])

    uni = load_universe()
    uni = uni[uni["included"]].set_index("base")
    try:
        qa = load_qa().set_index("base")
        valid = qa[~qa["excluded"]].index.tolist()
    except FileNotFoundError:
        # QA aún no corrió (p.ej. baseline durante la descarga):
        # usar los símbolos cuyo frame preparado ya existe
        valid = [b for b in uni.index if prepared_path(b).exists()]
    if symbols:
        valid = [s for s in valid if s in symbols]
    valid = sorted(valid, key=lambda b: int(uni.loc[b, "rank"]))

    params = base_ls_params(cfg)
    records = []

    for base in valid:
        row  = uni.loc[base]
        rank = int(row["rank"])
        mult = 2.5 if pass_name == "stress_t3" else float(row["cost_mult"])
        fees     = exp["base_fees"] * mult
        slippage = exp["base_slippage"] * mult

        print(f"\n  ── WF [{pass_name}] {base} (rank {rank}, tier {row['tier']}, "
              f"fees {fees*100:.3f}%/lado + slip {slippage*100:.3f}%)")
        try:
            df = pd.read_parquet(prepared_path(base))
            if pass_name == "oi_coins":
                df = df.copy()
                df["oi_usdt"] = df["oi_coins"]

            expected = count_expected_windows(
                df.index[0], df.index[-1], exp["train_months"], exp["test_months"]
            )
            wf = walk_forward(
                df             = df,
                strategy_class = LiquiditySweepStrategy,
                base_params    = params,
                param_grid     = grid,
                train_months   = exp["train_months"],
                test_months    = exp["test_months"],
                min_trades     = exp["min_trades"],
                init_cash      = g["init_cash"],
                timeframe      = g["timeframe"],
                n_jobs         = -1,
                fees           = fees,
                slippage       = slippage,
            )
            if wf.empty:
                records.append({
                    "pass": pass_name, "base": base, "rank": rank,
                    "tier": int(row["tier"]), "cost_mult": mult,
                    "n_windows_expected": expected, "window": pd.NA,
                })
                continue

            for _, w in wf.iterrows():
                records.append({
                    "pass":        pass_name,
                    "base":        base,
                    "rank":        rank,
                    "tier":        int(row["tier"]),
                    "cost_mult":   mult,
                    "n_windows_expected": expected,
                    "window":      int(w["window"]),
                    "test_start":  str(w["test_start"]),
                    "test_end":    str(w["test_end"]),
                    "sharpe_is":   w["sharpe_is"],
                    "sharpe_oos":  w["sharpe"],
                    "return_oos_%": w["return_%"],
                    "max_dd_oos_%": w["max_dd_%"],
                    "prof_factor_oos": w["prof_factor"],
                    "win_rate_oos_%": w["win_rate_%"],
                    "trades_oos":  int(w["trades"]),
                    "low_sample":  bool(w["trades"] < exp["low_sample_trades"]),
                    "best_params": w["best_params"],
                })
        except Exception as e:
            print(f"  [ERROR] {base} falló en pasada {pass_name}: {e}")
            traceback.print_exc(limit=2)
            records.append({
                "pass": pass_name, "base": base, "rank": rank,
                "tier": int(row["tier"]), "cost_mult": mult,
                "n_windows_expected": pd.NA, "window": pd.NA,
                "best_params": f"RUN_ERROR: {e}",
            })

    return pd.DataFrame(records)


def save_results(df_new: pd.DataFrame) -> pd.DataFrame:
    """Append idempotente: reemplaza las pasadas re-corridas, conserva el resto."""
    if RESULTS_PATH.exists():
        old = pd.read_parquet(RESULTS_PATH)
        keep = old[~(
            old["pass"].isin(df_new["pass"].unique())
            & old["base"].isin(df_new["base"].unique())
        )]
        df_all = pd.concat([keep, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_parquet(RESULTS_PATH)
    print(f"  [RESULTS] {len(df_all)} registros (símbolo, ventana) → {RESULTS_PATH}")
    return df_all
