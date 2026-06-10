"""
Test anti look-ahead del motor.

Reconstruye el frame de backtest con la data truncada N velas antes del
final y verifica que las señales históricas (entry, atr_sl, trend_daily)
del período común son EXACTAMENTE iguales a las del frame completo.
Si alguna parte del pipeline usara información futura (p.ej. la tendencia
diaria con el close del propio día, o un bfill), las señales cercanas al
corte cambiarían y el test falla.

Corre sobre 2 símbolos nuevos del experimento (fuera de BTC/ETH/SOL),
elegidos determinísticamente del universo. Requiere haber corrido las
etapas universe + data del experimento (usa la caché de data/).

Uso:  python -m tests.test_no_lookahead   (o vía pytest)
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.cap_gradient.config import DATA_DIR, exp_cfg, load_cfg
from experiments.cap_gradient.data import build_frame
from experiments.cap_gradient.runner import base_ls_params
from experiments.cap_gradient.universe import load_universe
from src.strategies.liquidity_sweep import LiquiditySweepStrategy

N_TRUNCATE = 500          # velas 1h recortadas del final
CHECK_COLS = ("entry", "trend_daily")


def _load_raw(row, cfg, exp):
    """Lee las cachés Parquet del fetcher sin tocar la red."""
    g = cfg["global"]
    tf = g["timeframe"]
    p_1h = DATA_DIR / f"{row['ccxt_symbol'].replace('/', '_')}_{tf}.parquet"
    p_1d = DATA_DIR / f"{row['ccxt_symbol'].replace('/', '_')}_{g['tf_trend']}.parquet"
    p_oi = DATA_DIR / f"oi_{row['symbol_bybit']}_{tf}.parquet"
    p_fr = DATA_DIR / f"funding_{row['binance_fut_symbol']}.parquet"
    for p in (p_1h, p_1d, p_oi, p_fr):
        if not p.exists():
            raise FileNotFoundError(f"Falta caché {p} — corré la etapa data primero")
    return (pd.read_parquet(p_1h), pd.read_parquet(p_1d),
            pd.read_parquet(p_oi), pd.read_parquet(p_fr))


def _check_at_cut(base, sig_full, df_raw, cut, exp, ma_1d, strat) -> int:
    """Trunca toda la data a `cut`, regenera señales y exige igualdad exacta
    en el período común. Devuelve la cantidad de barras comparadas."""
    df_1h, df_1d, df_oi, df_fr = df_raw
    t_1h = df_1h[df_1h.index <= cut]
    t_1d = df_1d[df_1d.index + pd.Timedelta(days=1) <= cut]  # solo días cerrados
    t_oi = df_oi[df_oi.index <= cut]
    t_fr = df_fr[df_fr.index <= cut]
    trunc = build_frame(t_1h, t_1d, t_oi, t_fr, exp["since_date"], ma_1d)
    sig_trunc = strat.prepare(trunc)

    common = sig_trunc.index
    assert len(common) > 1000, f"{base}: período común sospechosamente corto en {cut}"

    for col in CHECK_COLS:
        a = sig_full.loc[common, col].fillna(False)
        b = sig_trunc[col].fillna(False)
        diff = (a != b)
        assert not diff.any(), (
            f"{base}: LOOK-AHEAD detectado — {int(diff.sum())} barras de '{col}' "
            f"cambiaron al truncar en {cut} (primera: {diff.idxmax()})"
        )

    a = sig_full.loc[common, "atr_sl"]
    b = sig_trunc["atr_sl"]
    bad = ~(((a - b).abs() < 1e-12) | (a.isna() & b.isna()))
    assert not bad.any(), (
        f"{base}: LOOK-AHEAD en atr_sl — {int(bad.sum())} barras difieren "
        f"al truncar en {cut} (primera: {bad.idxmax()})"
    )
    return len(common)


def _adversarial_cuts(df_1d: pd.DataFrame, df_1h: pd.DataFrame,
                      ma_1d: int, n_flips: int = 6) -> list[pd.Timestamp]:
    """Puntos de corte donde el look-ahead se manifiesta si existe:
    mediodía de días en que la tendencia diaria FLIPEA (con la versión
    buggy, las barras intradía de ese día verían el flip antes de tiempo).
    Más un corte fijo de N_TRUNCATE velas como caso general."""
    ma    = df_1d["close"].rolling(ma_1d).mean()
    trend = (df_1d["close"] > ma)
    flips = trend.index[trend != trend.shift(1)]
    # Flips dentro del rango del backtest, espaciados a lo largo de la historia
    flips = [f for f in flips if f >= df_1h.index[0] + pd.Timedelta(days=30)
             and f <= df_1h.index[-1] - pd.Timedelta(days=2)]
    step = max(1, len(flips) // n_flips)
    cuts = [f.normalize() + pd.Timedelta(hours=12) for f in flips[::step][:n_flips]]
    cuts.append(df_1h.index[-(N_TRUNCATE + 1)])
    return cuts


def check_symbol(base: str) -> None:
    cfg, exp = load_cfg(), exp_cfg()
    uni = load_universe()
    row = uni[uni["base"] == base].iloc[0]
    ma_1d = cfg["strategy"]["liquidity_sweep"]["ma_trend_1d"]

    df_raw = _load_raw(row, cfg, exp)
    df_1h, df_1d, _, _ = df_raw

    full = build_frame(*df_raw, exp["since_date"], ma_1d)
    strat = LiquiditySweepStrategy(base_ls_params(cfg))
    sig_full = strat.prepare(full)

    cuts = _adversarial_cuts(df_1d, df_1h, ma_1d)
    n_common = 0
    for cut in cuts:
        n_common = _check_at_cut(base, sig_full, df_raw, cut, exp, ma_1d, strat)

    print(f"  [LOOKAHEAD] {base}: OK — {len(cuts)} cortes "
          f"(incl. {len(cuts)-1} días de flip de tendencia), señales idénticas")


def _pick_test_symbols() -> list[str]:
    """2 símbolos nuevos (no BTC/ETH/SOL), determinístico: ranks medios."""
    uni = load_universe()
    inc = uni[uni["included"] & ~uni["base"].isin(["BTC", "ETH", "SOL"])]
    inc = inc.sort_values("rank")
    picks = []
    for target in (8, 15):
        sub = inc[inc["rank"] >= target]
        pick = (sub.iloc[0] if len(sub) else inc.iloc[-1])["base"]
        if pick not in picks:
            picks.append(pick)
    return picks[:2]


def test_no_lookahead():
    for base in _pick_test_symbols():
        check_symbol(base)


def main() -> None:
    symbols = _pick_test_symbols()
    print(f"  [LOOKAHEAD] test sobre: {symbols} (truncando {N_TRUNCATE} velas)")
    for base in symbols:
        check_symbol(base)
    print("  [LOOKAHEAD] PASS")


if __name__ == "__main__":
    main()
