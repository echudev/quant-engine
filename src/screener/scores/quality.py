"""Score de Calidad — Piotroski F-Score (0-9 escalado a 0-100).

Compara fundamentals del ultimo ano (suma de 4 trimestres) vs ano anterior.
Si los datos no alcanzan, retorna score=None.

Criterios (1 punto cada uno):
  Rentabilidad
    1. Net Income > 0 (ultimo ano)
    2. Operating Cash Flow > 0
    3. ROA mejorando vs ano anterior
    4. CFO > Net Income (calidad de earnings)
  Solvencia
    5. Long-term debt cayendo (ratio LT debt / total assets)
    6. Current ratio mejorando
    7. No emision neta de acciones
  Eficiencia
    8. Gross margin mejorando
    9. Asset turnover mejorando (revenue / total assets)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


@dataclass
class QualityResult:
    score: Optional[float]            # 0-100 o None si no hay datos suficientes
    f_score: Optional[int]            # 0-9
    components: dict = field(default_factory=dict)
    reason_skipped: Optional[str] = None


# Nombres alternativos que yfinance devuelve para cada concepto.
# Buscamos el primero que aparezca en el indice de cada DataFrame.
ROW_ALIASES = {
    "net_income":       ["Net Income", "Net Income Common Stockholders", "NetIncome"],
    "cfo":              ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities", "Total Cash From Operating Activities"],
    "revenue":          ["Total Revenue", "Operating Revenue", "Revenue"],
    "gross_profit":     ["Gross Profit"],
    "total_assets":     ["Total Assets"],
    "current_assets":   ["Current Assets", "Total Current Assets"],
    "current_liab":     ["Current Liabilities", "Total Current Liabilities"],
    "lt_debt":          ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"],
    "shares_out":       ["Common Stock Shares Outstanding", "Ordinary Shares Number", "Share Issued"],
}


def _find_row(df: pd.DataFrame, aliases: list[str]) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    for name in aliases:
        if name in df.index:
            return df.loc[name]
    return None


def _sum_n(series: pd.Series, n: int = 4) -> Optional[float]:
    """Suma los primeros n trimestres (mas recientes a la izquierda en yfinance)."""
    s = series.dropna()
    if len(s) < n:
        return None
    return float(s.iloc[:n].sum())


def _avg_n(series: pd.Series, n: int = 4) -> Optional[float]:
    s = series.dropna()
    if len(s) < n:
        return None
    return float(s.iloc[:n].mean())


def _last(series: pd.Series) -> Optional[float]:
    s = series.dropna()
    if len(s) < 1:
        return None
    return float(s.iloc[0])


def _prev_n(series: pd.Series, n: int = 4) -> Optional[float]:
    """El n-esimo trimestre hacia atras (para current ratio, shares_out anteriores)."""
    s = series.dropna()
    if len(s) <= n:
        return None
    return float(s.iloc[n])


def piotroski_score(
    qis: pd.DataFrame,
    qbs: pd.DataFrame,
    qcf: pd.DataFrame,
) -> QualityResult:
    """Calcula F-Score con datos trimestrales de yfinance.

    yfinance devuelve los DataFrames con trimestres mas recientes en las
    primeras columnas. .iloc[:, :4] son los ultimos 4 trimestres = ultimo ano.
    """
    # Validacion minima
    if qis is None or qis.empty or qbs is None or qbs.empty or qcf is None or qcf.empty:
        return QualityResult(score=None, f_score=None, reason_skipped="missing_financials")

    # Extraemos rows
    rows = {}
    for key, aliases in ROW_ALIASES.items():
        for src in (qis, qbs, qcf):
            row = _find_row(src, aliases)
            if row is not None:
                rows[key] = row
                break

    needed = ["net_income", "cfo", "revenue", "total_assets"]
    missing = [k for k in needed if k not in rows]
    if missing:
        return QualityResult(score=None, f_score=None, reason_skipped=f"missing_rows:{missing}")

    # Sumas TTM (ultimos 4 trimestres) y year-ago (trimestres 5-8)
    ni_ttm  = _sum_n(rows["net_income"], 4)
    ni_prev = _sum_n(rows["net_income"].iloc[4:], 4) if len(rows["net_income"].dropna()) >= 8 else None

    cfo_ttm  = _sum_n(rows["cfo"], 4)
    rev_ttm  = _sum_n(rows["revenue"], 4)
    rev_prev = _sum_n(rows["revenue"].iloc[4:], 4) if len(rows["revenue"].dropna()) >= 8 else None

    # Total assets: usamos el promedio o el snapshot mas reciente
    ta_now  = _last(rows["total_assets"])
    ta_prev = _prev_n(rows["total_assets"], 4)

    components = {}
    score = 0

    # 1. Net income positivo
    c1 = (ni_ttm is not None and ni_ttm > 0)
    components["net_income_positive"] = c1
    score += int(c1)

    # 2. CFO positivo
    c2 = (cfo_ttm is not None and cfo_ttm > 0)
    components["cfo_positive"] = c2
    score += int(c2)

    # 3. ROA mejorando
    roa_now  = (ni_ttm / ta_now) if (ni_ttm is not None and ta_now and ta_now != 0) else None
    roa_prev = (ni_prev / ta_prev) if (ni_prev is not None and ta_prev and ta_prev != 0) else None
    c3 = (roa_now is not None and roa_prev is not None and roa_now > roa_prev)
    components["roa_improving"] = c3
    score += int(c3)

    # 4. CFO > NI (calidad de earnings)
    c4 = (cfo_ttm is not None and ni_ttm is not None and cfo_ttm > ni_ttm)
    components["cfo_gt_ni"] = c4
    score += int(c4)

    # 5. LT debt ratio cayendo
    if "lt_debt" in rows:
        ltd_now  = _last(rows["lt_debt"])
        ltd_prev = _prev_n(rows["lt_debt"], 4)
        ratio_now  = (ltd_now / ta_now) if (ltd_now is not None and ta_now) else None
        ratio_prev = (ltd_prev / ta_prev) if (ltd_prev is not None and ta_prev) else None
        c5 = (ratio_now is not None and ratio_prev is not None and ratio_now < ratio_prev)
    else:
        c5 = False
    components["lt_debt_decreasing"] = c5
    score += int(c5)

    # 6. Current ratio mejorando
    if "current_assets" in rows and "current_liab" in rows:
        ca_now  = _last(rows["current_assets"])
        cl_now  = _last(rows["current_liab"])
        ca_prev = _prev_n(rows["current_assets"], 4)
        cl_prev = _prev_n(rows["current_liab"], 4)
        cr_now  = (ca_now / cl_now) if (ca_now is not None and cl_now) else None
        cr_prev = (ca_prev / cl_prev) if (ca_prev is not None and cl_prev) else None
        c6 = (cr_now is not None and cr_prev is not None and cr_now > cr_prev)
    else:
        c6 = False
    components["current_ratio_improving"] = c6
    score += int(c6)

    # 7. No dilucion: shares actuales <= shares ano anterior
    if "shares_out" in rows:
        so_now  = _last(rows["shares_out"])
        so_prev = _prev_n(rows["shares_out"], 4)
        c7 = (so_now is not None and so_prev is not None and so_now <= so_prev * 1.01)  # 1% tolerancia
    else:
        c7 = False
    components["no_dilution"] = c7
    score += int(c7)

    # 8. Gross margin mejorando
    if "gross_profit" in rows and rev_ttm and rev_prev:
        gp_ttm = _sum_n(rows["gross_profit"], 4)
        gp_prev = _sum_n(rows["gross_profit"].iloc[4:], 4) if len(rows["gross_profit"].dropna()) >= 8 else None
        gm_now  = (gp_ttm / rev_ttm) if (gp_ttm is not None and rev_ttm) else None
        gm_prev = (gp_prev / rev_prev) if (gp_prev is not None and rev_prev) else None
        c8 = (gm_now is not None and gm_prev is not None and gm_now > gm_prev)
    else:
        c8 = False
    components["gross_margin_improving"] = c8
    score += int(c8)

    # 9. Asset turnover mejorando: revenue / total_assets
    at_now  = (rev_ttm / ta_now) if (rev_ttm and ta_now) else None
    at_prev = (rev_prev / ta_prev) if (rev_prev and ta_prev) else None
    c9 = (at_now is not None and at_prev is not None and at_now > at_prev)
    components["asset_turnover_improving"] = c9
    score += int(c9)

    return QualityResult(
        score=score * (100.0 / 9.0),
        f_score=score,
        components=components,
    )
