"""
Análisis del gradiente: agregados por símbolo, Spearman, tiers, stress.

Convención de signos: la hipótesis es "pares chicos rinden mejor".
El criterio prefijado (Spearman <= -0.4, p < 0.05) se evalúa sobre
Spearman(OI en USD, Sharpe OOS mediano): OI grande → Sharpe chico ⇒ rho
negativo. (Equivale a rho >= +0.4 contra el número de rank, porque rank
crece cuando el OI baja; Spearman es invariante a transformaciones
monótonas.)
"""

import ast
import json

import numpy as np
import pandas as pd
from scipy import stats

from .config import ANALYSIS_PATH, PLOTS_DIR, RESULTS_PATH, SUMMARY_PATH, exp_cfg
from .universe import load_universe


def load_results() -> pd.DataFrame:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"No existe {RESULTS_PATH}. Corré el backtest primero.")
    return pd.read_parquet(RESULTS_PATH)


def _param_stability(best_params: pd.Series) -> float:
    """Estabilidad del óptimo entre ventanas: share promedio del valor modal
    de cada parámetro (1.0 = el óptimo nunca salta; ~1/n_valores = aleatorio)."""
    parsed = []
    for s in best_params.dropna():
        try:
            parsed.append(ast.literal_eval(s))
        except (ValueError, SyntaxError):
            continue
    if len(parsed) < 2:
        return float("nan")
    dfp = pd.DataFrame(parsed)
    shares = [dfp[c].value_counts(normalize=True).iloc[0] for c in dfp.columns]
    return float(np.mean(shares))


def aggregate_symbols(results: pd.DataFrame) -> pd.DataFrame:
    """Agregado por (pass, símbolo). Las medianas de Sharpe usan solo
    ventanas con muestra suficiente (low_sample=False) cuando hay >= 2;
    si no, se calculan igual pero el símbolo queda flaggeado low_sample."""
    exp = exp_cfg()
    uni = load_universe()
    uni = uni[uni["included"]][["base", "rank", "tier", "oi_usd", "cost_mult"]]

    results = results.copy()
    # Ventanas sin trades dan Sharpe ±inf en vectorbt: no son información
    for col in ("sharpe_oos", "sharpe_is"):
        if col in results.columns:
            results[col] = results[col].replace([np.inf, -np.inf], np.nan)

    rows = []
    for (pass_name, base), g in results.groupby(["pass", "base"]):
        valid = g[g["window"].notna() & g["sharpe_oos"].notna()]
        rec = {
            "pass": pass_name, "base": base,
            "n_windows_expected": int(valid["n_windows_expected"].iloc[0])
                                  if len(valid) else int(g["n_windows_expected"].iloc[0] or 0),
            "n_windows_valid": len(valid),
        }
        if len(valid) == 0:
            rec.update({
                "pct_profitable": np.nan, "sharpe_oos_median": np.nan,
                "sharpe_oos_mean": np.nan, "degradation": np.nan,
                "trades_total": 0, "param_stability": np.nan,
                "symbol_low_sample": True,
            })
            rows.append(rec)
            continue

        good = valid[~valid["low_sample"].astype(bool)]
        sample = good if len(good) >= 2 else valid
        symbol_low_sample = len(good) < 2

        with np.errstate(divide="ignore", invalid="ignore"):
            degr = valid.loc[valid["sharpe_is"] > 0.1,
                             "sharpe_oos"] / valid.loc[valid["sharpe_is"] > 0.1, "sharpe_is"]

        rec.update({
            "pct_profitable":   round(float((valid["return_oos_%"] > 0).mean() * 100), 1),
            "sharpe_oos_median": round(float(sample["sharpe_oos"].median()), 3),
            "sharpe_oos_mean":   round(float(sample["sharpe_oos"].mean()), 3),
            "degradation":       round(float(degr.median()), 3) if len(degr) else np.nan,
            "trades_total":      int(valid["trades_oos"].sum()),
            "n_windows_low_sample": int(valid["low_sample"].astype(bool).sum()),
            "param_stability":   round(_param_stability(valid["best_params"]), 3),
            "symbol_low_sample": symbol_low_sample,
        })
        rows.append(rec)

    summary = pd.DataFrame(rows).merge(uni, on="base", how="left")
    return summary.sort_values(["pass", "rank"]).reset_index(drop=True)


def spearman_gradient(summary: pd.DataFrame, pass_name: str) -> dict:
    """Spearman entre OI en USD y métricas OOS, sobre símbolos con datos."""
    s = summary[(summary["pass"] == pass_name)
                & summary["sharpe_oos_median"].notna()]
    out = {"pass": pass_name, "n_symbols": len(s)}
    for metric in ("sharpe_oos_median", "pct_profitable"):
        if len(s) >= 4:
            rho, p = stats.spearmanr(s["oi_usd"], s[metric])
            out[metric] = {"rho": round(float(rho), 3), "p_value": round(float(p), 4)}
        else:
            out[metric] = {"rho": None, "p_value": None}
    return out


def tier_table(summary: pd.DataFrame, pass_name: str) -> pd.DataFrame:
    s = summary[(summary["pass"] == pass_name) & summary["sharpe_oos_median"].notna()]
    return (
        s.groupby("tier")
        .agg(
            n_symbols          = ("base", "count"),
            sharpe_oos_median  = ("sharpe_oos_median", "median"),
            pct_profitable_avg = ("pct_profitable", "mean"),
            trades_avg         = ("trades_total", "mean"),
            degradation_median = ("degradation", "median"),
        )
        .round(2)
        .reset_index()
    )


def evaluate_criteria(summary: pd.DataFrame, sp_main: dict) -> dict:
    """Criterio prefijado (no negociable después de ver resultados)."""
    exp = exp_cfg()
    rho = sp_main["sharpe_oos_median"]["rho"]
    p   = sp_main["sharpe_oos_median"]["p_value"]
    c1 = (rho is not None) and (rho <= -0.4) and (p < 0.05)

    s = summary[(summary["pass"] == "main") & (summary["rank"] > 5)]
    candidates = s[
        (s["pct_profitable"] >= 60.0)
        & (s["n_windows_valid"] >= 2)
        & (~s["symbol_low_sample"])
    ]
    c2 = len(candidates) >= 5

    if c1:
        verdict = "CONFIRMADA"
    elif c2:
        verdict = "PARCIAL (nicho explotable sin gradiente limpio)"
    else:
        verdict = "REFUTADA"

    return {
        "criterion_1_spearman": {"rho": rho, "p": p, "passed": bool(c1)},
        "criterion_2_small_caps": {
            "n_qualifying": int(len(candidates)),
            "required": 5,
            "passed": bool(c2),
            "candidates": candidates["base"].tolist(),
        },
        "verdict": verdict,
    }


def make_plots(summary: pd.DataFrame, results: pd.DataFrame) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    for pass_name, fname in (("main", "gradient_main.png"),
                             ("stress_t3", "gradient_stress_t3.png"),
                             ("oi_coins", "gradient_oi_coins.png")):
        s = summary[(summary["pass"] == pass_name) & summary["sharpe_oos_median"].notna()]
        if s.empty:
            continue
        fig, ax = plt.subplots(figsize=(11, 6.5))
        sizes = 30 + 270 * (s["trades_total"] / max(s["trades_total"].max(), 1))
        sc = ax.scatter(s["rank"], s["sharpe_oos_median"], s=sizes,
                        c=s["pct_profitable"], cmap="RdYlGn", vmin=0, vmax=100,
                        edgecolors="k", linewidths=0.6, zorder=3)
        for _, r in s.iterrows():
            weight = "bold" if r["base"] in ("BTC", "ETH") else "normal"
            ax.annotate(r["base"], (r["rank"], r["sharpe_oos_median"]),
                        textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=8, fontweight=weight)
        ax.axhline(0, color="gray", lw=0.8, ls="--", zorder=1)
        ax.set_xlabel("Rank por OI (1 = mayor open interest)")
        ax.set_ylabel("Sharpe OOS mediano")
        ax.set_title(f"Gradiente de capitalización — Liquidity Sweep OI [{pass_name}]\n"
                     "tamaño ∝ trades OOS · color = % ventanas rentables")
        fig.colorbar(sc, ax=ax, label="% ventanas OOS rentables")

        # Eje secundario: OI en USD (escala log) en las posiciones de los símbolos
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(s["rank"])
        ax2.set_xticklabels([f"{v/1e9:.1f}B" if v >= 1e9 else f"{v/1e6:.0f}M"
                             for v in s["oi_usd"]], fontsize=7, rotation=45)
        ax2.set_xlabel("Open Interest (USD, snapshot)")

        out = PLOTS_DIR / fname
        fig.tight_layout()
        fig.savefig(out, dpi=130)
        plt.close(fig)
        paths.append(str(out))
        print(f"  [PLOT] {out}")
    return paths


def run_analysis() -> dict:
    results = load_results()
    summary = aggregate_symbols(results)
    summary.to_parquet(SUMMARY_PATH)

    sp = {p: spearman_gradient(summary, p) for p in summary["pass"].unique()}
    tiers = {p: tier_table(summary, p).to_dict(orient="records")
             for p in summary["pass"].unique()}
    criteria = evaluate_criteria(summary, sp["main"])
    plots = make_plots(summary, results)

    analysis = {"spearman": sp, "tier_tables": tiers,
                "criteria": criteria, "plots": plots}
    ANALYSIS_PATH.write_text(json.dumps(analysis, indent=2, default=str))
    print(f"  [ANALYSIS] veredicto: {criteria['verdict']} → {ANALYSIS_PATH}")
    return analysis
