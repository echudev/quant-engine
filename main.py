"""
===================================================================
 BACKTEST ENGINE — v3
===================================================================
 Cambios vs v2:
   [1] Filtro de tendencia multi-timeframe (1H + 1D)
       → solo opera long cuando el diario confirma tendencia alcista
   [2] Stop Loss dinámico basado en ATR
       → SL = ATR_MULT × ATR(14), se adapta a la volatilidad real
   [3] Período de backtest desde 2023-01-01
       → evita el bear market 2022 para validar la lógica primero

 Estrategias:
   A) Liquidity Sweep    — caída brusca con volumen + confirmación
   B) RSI Mean Reversion — RSI oversold con tendencia alcista

 Timeframe : 1H  |  Tendencia : 1D
 Stack     : ccxt · pandas · vectorbt
===================================================================
"""

import ccxt
import pandas as pd
import numpy as np
import vectorbt as vbt
import itertools
from pathlib import Path
from datetime import datetime, timezone

# ──────────────────────────────────────────────────────────────────
#  CONFIGURACIÓN GLOBAL
# ──────────────────────────────────────────────────────────────────

SYMBOL      = "BTC/USDT"
TIMEFRAME   = "1h"
TF_TREND    = "1d"            # [CAMBIO 1] timeframe superior para tendencia
SINCE_DATE  = "2023-01-01"   # [CAMBIO 3] desde inicio del bull market 2023
EXCHANGE_ID = "binance"
INIT_CASH   = 10_000
CACHE_DIR   = Path("data")
CACHE_DIR.mkdir(exist_ok=True)

# ── Parámetros: Liquidity Sweep
LS_MA_PERIOD       = 50
LS_SWEEP_DROP_PCT  = 1.2
LS_SWEEP_VOL_MULT  = 1.4
LS_VOL_LOOKBACK    = 20
LS_CONFIRM_BODY    = 0.2
LS_TAKE_PROFIT     = 3.0
LS_ATR_MULT        = 2.0      # [CAMBIO 2] SL = 2 × ATR(14)
LS_ATR_PERIOD      = 14

# ── Parámetros: RSI Mean Reversion
RSI_PERIOD         = 14
RSI_OVERSOLD       = 35
RSI_EXIT           = 55
RSI_MA_TREND       = 200      # MA de tendencia en 1H (complementa el diario)
RSI_TAKE_PROFIT    = 4.0
RSI_ATR_MULT       = 2.0      # [CAMBIO 2] SL = 2 × ATR(14)
RSI_ATR_PERIOD     = 14


# ──────────────────────────────────────────────────────────────────
#  1. DATOS — caché Parquet + descarga incremental
# ──────────────────────────────────────────────────────────────────

def fetch_ohlcv(symbol: str, timeframe: str, since_date: str, exchange_id: str) -> pd.DataFrame:
    """Descarga OHLCV con caché local incremental en Parquet."""
    cache_file = CACHE_DIR / f"{symbol.replace('/', '_')}_{timeframe}.parquet"

    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True})

    if cache_file.exists():
        df_cached = pd.read_parquet(cache_file)
        last_ts   = df_cached.index[-1]
        since_ts  = int(last_ts.timestamp() * 1000) + 1
        print(f"[CACHE] {timeframe}: {len(df_cached)} velas en disco (hasta {last_ts.date()})")
    else:
        df_cached = None
        since_ts  = int(
            datetime.strptime(since_date, "%Y-%m-%d")
            .replace(tzinfo=timezone.utc).timestamp() * 1000
        )
        print(f"[DATA] {timeframe}: sin caché — descargando desde {since_date}...")

    all_ohlcv = []
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_ts, limit=1000)
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        since_ts = ohlcv[-1][0] + 1
        if len(ohlcv) < 1000:
            break

    if all_ohlcv:
        df_new = pd.DataFrame(all_ohlcv, columns=["timestamp","open","high","low","close","volume"])
        df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], unit="ms", utc=True)
        df_new.set_index("timestamp", inplace=True)
        print(f"[DATA] {timeframe}: {len(df_new)} velas nuevas.")
    else:
        df_new = None
        print(f"[CACHE] {timeframe}: todo al día.")

    frames = [f for f in [df_cached, df_new] if f is not None]
    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_parquet(cache_file)
    print(f"[CACHE] {timeframe}: {len(df)} velas totales → {cache_file}")
    return df


# ──────────────────────────────────────────────────────────────────
#  2. FILTRO DE TENDENCIA MULTI-TIMEFRAME  [CAMBIO 1]
# ──────────────────────────────────────────────────────────────────

def build_daily_trend(df_daily: pd.DataFrame, ma_period: int = 50) -> pd.Series:
    """
    Calcula tendencia alcista en timeframe diario.
    Retorna Serie booleana indexada por fecha (sin hora).
    La tendencia se confirma cuando close > MA(period) en 1D.
    """
    ma = df_daily["close"].rolling(ma_period).mean()
    trend = df_daily["close"] > ma
    trend.index = trend.index.normalize()   # quitar hora → solo fecha
    return trend


def apply_daily_trend(df_1h: pd.DataFrame, daily_trend: pd.Series) -> pd.DataFrame:
    """
    Agrega columna 'trend_daily' al DataFrame 1H.
    Hace forward-fill para que cada hora del día herede
    la tendencia calculada al cierre del día anterior.
    """
    df = df_1h.copy()
    # Reindexar la tendencia diaria al índice horario y propagar hacia adelante
    df["trend_daily"] = daily_trend.reindex(
        df.index.normalize(), method="ffill"
    ).values
    return df


# ──────────────────────────────────────────────────────────────────
#  3. ATR — Stop Loss dinámico  [CAMBIO 2]
# ──────────────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range.
    TR = max(high-low, |high-prev_close|, |low-prev_close|)
    """
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def atr_sl_series(df: pd.DataFrame, atr_mult: float, atr_period: int) -> pd.Series:
    """
    Retorna el SL como fracción del precio (para vectorbt sl_stop)
    basado en ATR: sl = atr_mult × ATR / close
    """
    atr = compute_atr(df, atr_period)
    return (atr_mult * atr / df["close"]).fillna(method="bfill")


# ──────────────────────────────────────────────────────────────────
#  4A. INDICADORES Y SEÑALES — Liquidity Sweep
# ──────────────────────────────────────────────────────────────────

def compute_indicators_ls(df: pd.DataFrame,
                           ma_period: int = LS_MA_PERIOD,
                           vol_lookback: int = LS_VOL_LOOKBACK,
                           atr_period: int = LS_ATR_PERIOD,
                           atr_mult: float = LS_ATR_MULT) -> pd.DataFrame:
    df = df.copy()
    df["ma50"]            = df["close"].rolling(ma_period).mean()
    df["trend_1h"]        = df["close"] > df["ma50"]
    df["candle_drop_pct"] = (df["open"] - df["close"]) / df["open"] * 100
    df["lower_wick_pct"]  = (df["close"] - df["low"]) / df["open"] * 100
    df["vol_ma"]          = df["volume"].rolling(vol_lookback).mean()
    df["vol_ratio"]       = df["volume"] / df["vol_ma"]
    df["body_pct"]        = abs(df["close"] - df["open"]) / df["open"] * 100
    df["atr_sl"]          = atr_sl_series(df, atr_mult, atr_period)  # [CAMBIO 2]
    return df


def generate_signals_ls(df: pd.DataFrame,
                         sweep_drop: float = LS_SWEEP_DROP_PCT,
                         vol_mult: float   = LS_SWEEP_VOL_MULT,
                         confirm_body: float = LS_CONFIRM_BODY) -> pd.DataFrame:
    df = df.copy()
    prev_drop       = df["candle_drop_pct"].shift(1) >= sweep_drop
    prev_vol_spike  = df["vol_ratio"].shift(1) >= vol_mult
    bullish_confirm = (df["close"] > df["open"]) & (df["body_pct"] >= confirm_body)
    trend_ok        = df["trend_1h"] & df["trend_daily"]   # [CAMBIO 1] ambos TF
    df["entry"]     = prev_drop & prev_vol_spike & bullish_confirm & trend_ok
    return df


# ──────────────────────────────────────────────────────────────────
#  4B. INDICADORES Y SEÑALES — RSI Mean Reversion
# ──────────────────────────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_indicators_rsi(df: pd.DataFrame,
                            rsi_period: int  = RSI_PERIOD,
                            ma_trend: int    = RSI_MA_TREND,
                            atr_period: int  = RSI_ATR_PERIOD,
                            atr_mult: float  = RSI_ATR_MULT) -> pd.DataFrame:
    df = df.copy()
    df["rsi"]     = compute_rsi(df["close"], rsi_period)
    df["ma200"]   = df["close"].rolling(ma_trend).mean()
    df["trend_1h"]= df["close"] > df["ma200"]
    df["atr_sl"]  = atr_sl_series(df, atr_mult, atr_period)  # [CAMBIO 2]
    return df


def generate_signals_rsi(df: pd.DataFrame,
                          oversold: float   = RSI_OVERSOLD,
                          exit_level: float = RSI_EXIT) -> pd.DataFrame:
    """
    Entrada : RSI cruza hacia arriba el nivel oversold con tendencia alcista (1H + 1D)
    Salida  : RSI sube sobre exit_level  o  SL/TP dinámico
    """
    df = df.copy()
    rsi_was_below  = df["rsi"].shift(1) < oversold
    rsi_now_above  = df["rsi"] >= oversold
    trend_ok       = df["trend_1h"] & df["trend_daily"]   # [CAMBIO 1]
    df["entry"]    = rsi_was_below & rsi_now_above & trend_ok
    df["exit_rsi"] = df["rsi"] > exit_level
    return df


# ──────────────────────────────────────────────────────────────────
#  5. BACKTEST GENÉRICO — SL dinámico por ATR
# ──────────────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame,
                 tp_pct: float,
                 exit_col: str = None) -> vbt.Portfolio:
    """
    SL dinámico: usa la columna 'atr_sl' calculada por vela.
    TP fijo como fracción del precio de entrada.
    """
    exits = df[exit_col] if exit_col and exit_col in df.columns \
            else pd.Series(False, index=df.index)

    return vbt.Portfolio.from_signals(
        close     = df["close"],
        entries   = df["entry"],
        exits     = exits,
        sl_stop   = df["atr_sl"],      # [CAMBIO 2] SL dinámico por ATR
        tp_stop   = tp_pct / 100,
        init_cash = INIT_CASH,
        fees      = 0.001,
        size      = 1.0,
        freq      = TIMEFRAME,
    )


# ──────────────────────────────────────────────────────────────────
#  6. REPORTE
# ──────────────────────────────────────────────────────────────────

def print_report(portfolio: vbt.Portfolio, name: str, tp: float) -> dict:
    stats         = portfolio.stats()
    total_return  = portfolio.total_return() * 100
    sharpe        = portfolio.sharpe_ratio()
    max_dd        = portfolio.max_drawdown() * 100
    win_rate      = stats.get("Win Rate [%]", float("nan"))
    total_trades  = stats.get("Total Trades", 0)
    profit_factor = stats.get("Profit Factor", float("nan"))

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  {SYMBOL} {TIMEFRAME}+{TF_TREND}  |  {SINCE_DATE} → hoy")
    print(f"  Capital ${INIT_CASH:,.0f}  |  TP {tp}%  |  SL ATR-dinámico")
    print(f"{'-'*60}")
    print(f"  Retorno      : {total_return:.2f}%")
    print(f"  Sharpe       : {sharpe:.2f}")
    print(f"  Max DD       : {max_dd:.2f}%")
    print(f"  Trades       : {total_trades}")
    print(f"  Win rate     : {win_rate:.1f}%")
    print(f"  Prof. factor : {profit_factor:.2f}")
    print(f"{'='*60}\n")

    return {
        "name": name, "return": total_return, "sharpe": sharpe,
        "max_dd": max_dd, "trades": total_trades,
        "win_rate": win_rate, "profit_factor": profit_factor,
    }


def print_comparison(results: list) -> None:
    print(f"\n{'━'*62}")
    print(f"  COMPARACIÓN")
    print(f"{'━'*62}")
    print(f"  {'Estrategia':<28} {'Retorno':>8} {'Sharpe':>7} {'MaxDD':>8} {'WR':>7} {'Trades':>7}")
    print(f"  {'-'*58}")
    for r in results:
        print(f"  {r['name']:<28} {r['return']:>7.1f}% {r['sharpe']:>7.2f} "
              f"{r['max_dd']:>7.1f}% {r['win_rate']:>6.1f}% {r['trades']:>7}")
    print(f"{'━'*62}\n")


# ──────────────────────────────────────────────────────────────────
#  7. GRID SEARCH
# ──────────────────────────────────────────────────────────────────

def optimize_ls(df: pd.DataFrame, min_trades: int = 30) -> pd.DataFrame:
    sweep_drops  = [0.8, 1.0, 1.2, 1.5, 2.0, 2.5]
    vol_mults    = [1.2, 1.4, 1.6, 1.8, 2.0]
    take_profits = [2.0, 3.0, 4.0, 5.0, 6.0]
    atr_mults    = [1.5, 2.0, 2.5, 3.0]

    combos  = list(itertools.product(sweep_drops, vol_mults, take_profits, atr_mults))
    results = []
    print(f"[OPT-LS] {len(combos)} combinaciones (mín {min_trades} trades)...")

    for drop, vol, tp, atr_m in combos:
        df_ind = compute_indicators_ls(df, atr_mult=atr_m)
        df_ind = apply_daily_trend(df_ind, df_ind["trend_daily"] if "trend_daily" in df_ind.columns
                                   else pd.Series(True, index=df_ind.index.normalize()))
        df_sig = generate_signals_ls(df_ind, sweep_drop=drop, vol_mult=vol)

        if df_sig["entry"].sum() < min_trades:
            continue
        try:
            pf     = run_backtest(df_sig, tp_pct=tp)
            stats  = pf.stats()
            trades = stats.get("Total Trades", 0)
            wr     = stats.get("Win Rate [%]", 0)
            dd     = pf.max_drawdown() * 100

            if trades < min_trades or dd < -30 or wr < 45:
                continue

            results.append({
                "sweep_drop": drop, "vol_mult": vol, "tp_%": tp, "atr_mult": atr_m,
                "trades": trades, "win_rate": round(wr, 1),
                "sharpe": round(pf.sharpe_ratio(), 3),
                "return_%": round(pf.total_return()*100, 2),
                "drawdown_%": round(dd, 2),
                "pf": round(stats.get("Profit Factor", 0), 2),
            })
        except Exception:
            continue

    if not results:
        print("[OPT-LS] Sin resultados que superen los filtros.")
        return pd.DataFrame()

    df_res = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    print(f"\n[OPT-LS] TOP 10 (de {len(results)} válidas):")
    print(df_res.head(10).to_string(index=False))
    return df_res


def optimize_rsi(df: pd.DataFrame, min_trades: int = 30) -> pd.DataFrame:
    oversold_levels = [25, 28, 30, 33, 35, 38, 40]
    exit_levels     = [50, 55, 60, 65]
    take_profits    = [3.0, 4.0, 5.0, 6.0, 8.0]
    atr_mults       = [1.5, 2.0, 2.5, 3.0]

    combos  = list(itertools.product(oversold_levels, exit_levels, take_profits, atr_mults))
    results = []
    print(f"\n[OPT-RSI] {len(combos)} combinaciones (mín {min_trades} trades)...")

    for oversold, exit_l, tp, atr_m in combos:
        if exit_l <= oversold:
            continue
        df_ind = compute_indicators_rsi(df, atr_mult=atr_m)
        df_sig = generate_signals_rsi(df_ind, oversold=oversold, exit_level=exit_l)

        if df_sig["entry"].sum() < min_trades:
            continue
        try:
            pf     = run_backtest(df_sig, tp_pct=tp, exit_col="exit_rsi")
            stats  = pf.stats()
            trades = stats.get("Total Trades", 0)
            wr     = stats.get("Win Rate [%]", 0)
            dd     = pf.max_drawdown() * 100

            if trades < min_trades or dd < -30 or wr < 45:
                continue

            results.append({
                "rsi_entry": oversold, "rsi_exit": exit_l, "tp_%": tp, "atr_mult": atr_m,
                "trades": trades, "win_rate": round(wr, 1),
                "sharpe": round(pf.sharpe_ratio(), 3),
                "return_%": round(pf.total_return()*100, 2),
                "drawdown_%": round(dd, 2),
                "pf": round(stats.get("Profit Factor", 0), 2),
            })
        except Exception:
            continue

    if not results:
        print("[OPT-RSI] Sin resultados que superen los filtros.")
        return pd.DataFrame()

    df_res = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    print(f"\n[OPT-RSI] TOP 10 (de {len(results)} válidas):")
    print(df_res.head(10).to_string(index=False))
    return df_res


# ──────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Datos 1H + 1D  [CAMBIO 1 + 3]
    # El diario se descarga desde más atrás para tener suficiente
    # historia para calcular la MA50 diaria desde el inicio del test
    df_1h    = fetch_ohlcv(SYMBOL, TIMEFRAME, SINCE_DATE,  EXCHANGE_ID)
    df_1d    = fetch_ohlcv(SYMBOL, TF_TREND,  "2021-01-01", EXCHANGE_ID)

    print()
    daily_trend = build_daily_trend(df_1d, ma_period=50)

    results_summary = []

    # ════════════════════════════════════════
    #  ESTRATEGIA A: Liquidity Sweep
    # ════════════════════════════════════════
    print("── Estrategia A: Liquidity Sweep ──")
    df_ls = compute_indicators_ls(df_1h)
    df_ls = apply_daily_trend(df_ls, daily_trend)
    df_ls = generate_signals_ls(df_ls)
    print(f"[LS] Señales: {df_ls['entry'].sum()}")

    pf_ls = run_backtest(df_ls, tp_pct=LS_TAKE_PROFIT)
    r_ls  = print_report(pf_ls, "Liquidity Sweep", LS_TAKE_PROFIT)
    results_summary.append(r_ls)

    # ════════════════════════════════════════
    #  ESTRATEGIA B: RSI Mean Reversion
    # ════════════════════════════════════════
    print("── Estrategia B: RSI Mean Reversion ──")
    df_rsi = compute_indicators_rsi(df_1h)
    df_rsi = apply_daily_trend(df_rsi, daily_trend)
    df_rsi = generate_signals_rsi(df_rsi)
    print(f"[RSI] Señales: {df_rsi['entry'].sum()}")

    pf_rsi = run_backtest(df_rsi, tp_pct=RSI_TAKE_PROFIT, exit_col="exit_rsi")
    r_rsi  = print_report(pf_rsi, "RSI Mean Reversion", RSI_TAKE_PROFIT)
    results_summary.append(r_rsi)

    # ════════════════════════════════════════
    #  TABLA COMPARATIVA
    # ════════════════════════════════════════
    print_comparison(results_summary)

    # ════════════════════════════════════════
    #  GRID SEARCH (descomentar para correr)
    # ════════════════════════════════════════
    # opt_ls  = optimize_ls(df_ls,  min_trades=30)
    # opt_rsi = optimize_rsi(df_rsi, min_trades=30)