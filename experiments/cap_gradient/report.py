"""Genera results/experiment_cap_gradient/report.md."""

import json
from datetime import datetime, timezone

import pandas as pd

from .config import (ANALYSIS_PATH, EXP_DIR, QA_PATH, REPORT_PATH,
                     SUMMARY_PATH, exp_cfg)
from .universe import load_universe


def _md_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def generate_report() -> str:
    exp = exp_cfg()
    analysis = json.loads(ANALYSIS_PATH.read_text())
    summary  = pd.read_parquet(SUMMARY_PATH)
    uni      = load_universe()
    qa       = pd.read_parquet(QA_PATH)

    crit = analysis["criteria"]
    sp_main   = analysis["spearman"]["main"]
    sp_stress = analysis["spearman"].get("stress_t3", {})
    sp_coins  = analysis["spearman"].get("oi_coins", {})

    main = summary[summary["pass"] == "main"].copy()
    cols = ["rank", "base", "tier", "oi_usd", "n_windows_valid", "n_windows_expected",
            "pct_profitable", "sharpe_oos_median", "sharpe_oos_mean", "degradation",
            "trades_total", "n_windows_low_sample", "param_stability", "symbol_low_sample"]
    main_tbl = main[[c for c in cols if c in main.columns]].copy()
    main_tbl["oi_usd"] = (main_tbl["oi_usd"] / 1e6).round(0).astype("Int64").astype(str) + "M"

    excl_uni = uni[~uni["included"] & uni["exclusion_reason"].notna()
                   & (uni["exclusion_reason"] != "universo completo (top_n alcanzado)")]
    excl_qa  = qa[qa["excluded"]]

    snapshot = str(uni["snapshot_ts"].iloc[0])[:10]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    rho, p = sp_main["sharpe_oos_median"]["rho"], sp_main["sharpe_oos_median"]["p_value"]
    rho_pw, p_pw = sp_main["pct_profitable"]["rho"], sp_main["pct_profitable"]["p_value"]

    lines = [
        f"# Experimento: gradiente de capitalización — Liquidity Sweep OI",
        f"",
        f"Generado: {now} · Snapshot del universo: {snapshot} · "
        f"Config: `config_snapshot.toml` · Metadata: `run_meta.json`",
        f"",
        f"## Resumen ejecutivo",
        f"",
        f"**Hipótesis:** la rentabilidad out-of-sample de Liquidity Sweep OI correlaciona",
        f"inversamente con el tamaño del par (open interest en USD), porque el edge",
        f"proviene de cascadas de liquidación que pesan más donde no hay capital",
        f"institucional absorbiendo.",
        f"",
        f"**Veredicto (criterio prefijado): {crit['verdict']}**",
        f"",
        f"- Criterio 1 — Spearman(OI USD, Sharpe OOS mediano) ≤ −0.4 con p < 0.05: "
        f"rho = **{rho}**, p = **{p}** → {'CUMPLE' if crit['criterion_1_spearman']['passed'] else 'NO cumple'}",
        f"- Spearman(OI USD, % ventanas rentables): rho = {rho_pw}, p = {p_pw}",
        f"- Criterio 2 — ≥ 5 símbolos fuera del top 5 con ≥ 60% de ventanas OOS "
        f"rentables con costos pesimistas: **{crit['criterion_2_small_caps']['n_qualifying']}** "
        f"califican → {'CUMPLE' if crit['criterion_2_small_caps']['passed'] else 'NO cumple'}"
        f" ({', '.join(crit['criterion_2_small_caps']['candidates']) or 'ninguno'})",
        f"",
        f"Nota de signos: rho se calcula contra el OI en USD, de modo que un valor",
        f"negativo significa \"más OI → menos Sharpe\" (pares chicos mejores), que es",
        f"lo que la hipótesis predice.",
        f"",
        f"## Walk-forward",
        f"",
        f"Idéntico para todos los símbolos: train {exp['train_months']}m / test "
        f"{exp['test_months']}m rolling, grilla unificada de "
        f"{len(exp['grid']['oi_drop_pct']) * len(exp['grid']['funding_min']) * len(exp['grid']['take_profit']) * len(exp['grid']['atr_mult'])} "
        f"combinaciones, sin tuning por símbolo. Costos por tier de OI: "
        f"ranks 1-5 = 1.0x, 6-12 = 1.5x, 13-20 = 2.5x sobre base "
        f"{exp['base_fees']*100:.2f}% fee + {exp['base_slippage']*100:.3f}% slippage por lado.",
        f"",
        f"## Tabla por símbolo (pasada principal, costos por tier)",
        f"",
        _md_table(main_tbl),
        f"",
        f"Ventanas con < {exp['low_sample_trades']} trades OOS se marcan low_sample y",
        f"se excluyen de las medianas de Sharpe (columna `n_windows_low_sample`).",
        f"`symbol_low_sample=True` ⇒ el símbolo no tiene ≥2 ventanas con muestra",
        f"suficiente; sus medianas se calculan sobre todas las ventanas y deben",
        f"leerse con cautela.",
        f"",
        f"## Comparación por tier de costos",
        f"",
    ]

    for pass_name, label in (("main", "Pasada principal (costos por tier)"),
                             ("stress_t3", "Stress: costos tier 3 (2.5x) uniformes"),
                             ("oi_coins", "Sensibilidad: OI puro (oi_coins)")):
        if pass_name in analysis["tier_tables"]:
            tt = pd.DataFrame(analysis["tier_tables"][pass_name])
            sp_p = analysis["spearman"].get(pass_name, {})
            sh = sp_p.get("sharpe_oos_median", {})
            lines += [f"### {label}", "",
                      _md_table(tt), "",
                      f"Spearman(OI USD, Sharpe OOS mediano): rho = {sh.get('rho')}, "
                      f"p = {sh.get('p_value')}", ""]

    lines += [
        f"## Plots",
        f"",
        *[f"![{p.split('/')[-1]}](plots/{p.split('/')[-1]})" for p in analysis["plots"]],
        f"",
        f"## Limitaciones (sesgos conocidos)",
        f"",
        f"1. **Sesgo de supervivencia:** el universo es el snapshot actual de Bybit "
        f"({snapshot}), no point-in-time. Los perpetuos deslistados (que típicamente "
        f"murieron perdiendo) no están: esto **infla los resultados de los pares chicos**.",
        f"2. **Slippage estimado, no medido:** los multiplicadores por tier "
        f"({exp['tier_cost_mult']}) son supuestos. El stress test con tier 3 uniforme "
        f"acota el riesgo, pero ningún backtest reemplaza ejecución real.",
        f"3. **Ventanas efectivas desiguales:** símbolos con menos historia tienen menos "
        f"ventanas y sus períodos de test no coinciden en calendario con los de BTC "
        f"(confusión con régimen de mercado). Ver `n_windows_expected` / `n_windows_valid`.",
        f"4. **Una sola estrategia:** el gradiente (si existe) es de Liquidity Sweep OI; "
        f"no generaliza a otras estrategias sin testearlas.",
        f"5. **Exclusiones por arquitectura de datos:** símbolos top de Bybit sin spot "
        f"en Binance o sin 18m de historia quedaron afuera "
        f"({len(excl_uni)} candidatos excluidos del universo; ver más abajo). Eso recorta "
        f"el universo de pares chicos de forma no aleatoria.",
        f"6. **Señal OI×precio:** la 'caída de OI' del motor se computa sobre "
        f"oi_usdt = OI × precio, que mezcla la caída de OI con la del precio; los pares "
        f"más volátiles generan más señal espuria. La pasada `oi_coins` (OI puro) "
        f"mide la sensibilidad del gradiente a este efecto.",
        f"7. **Velas con huecos:** símbolos con > {exp['max_missing_pct']}% de velas "
        f"faltantes fueron excluidos ({len(excl_qa)} símbolos; ver QA).",
        f"",
        f"### Candidatos excluidos del universo",
        f"",
        _md_table(excl_uni[["symbol_bybit", "oi_usd", "exclusion_reason"]]
                  .assign(oi_usd=lambda d: (d["oi_usd"] / 1e6).round(0)))
        if len(excl_uni) else "(ninguno)",
        f"",
        f"### Símbolos excluidos por QA de datos",
        f"",
        _md_table(excl_qa[["base", "reason"]]) if len(excl_qa) else "(ninguno)",
        f"",
        f"## Recomendación",
        f"",
    ]

    if crit["criterion_1_spearman"]["passed"] or crit["criterion_2_small_caps"]["passed"]:
        cands = crit["criterion_2_small_caps"]["candidates"]
        lines += [
            f"Candidatos a paper trading (cumplen el criterio prefijado: fuera del top 5, "
            f"≥ 60% de ventanas OOS rentables con costos pesimistas, muestra suficiente):",
            f"",
            *([f"- **{c}**" for c in cands] if cands else ["- (ninguno)"]),
            f"",
            f"Siguiente paso sugerido: paper trading con los parámetros de la última "
            f"ventana de cada candidato, y re-validar el slippage real contra el supuesto.",
        ]
    else:
        lines += [
            f"**Hipótesis refutada bajo el criterio prefijado.** No se recomienda pasar "
            f"ningún símbolo a paper trading en base a este experimento. Los símbolos que "
            f"individualmente dieron bien NO se cherry-pickean: con "
            f"{int(summary[summary['pass'] == 'main']['base'].nunique())} símbolos testeados, "
            f"algunos darán bien por azar (data snooping sobre el universo).",
        ]

    REPORT_PATH.write_text("\n".join(lines))
    print(f"  [REPORT] → {REPORT_PATH}")
    return str(REPORT_PATH)
