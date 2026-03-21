"""
===================================================================
 LIQUIDITY SWEEP BACKTEST — Esqueleto base
===================================================================
 Estrategia:
   1. Tendencia alcista de mediano plazo  → precio sobre MA50 en 4H
   2. Liquidity sweep                     → caída brusca con volumen alto
   3. Entrada                             → vela de confirmación alcista
   4. Salida                              → Take Profit fijo + Stop Loss

 Stack: ccxt (datos) · pandas (procesamiento) · vectorbt (backtest)
===================================================================
"""

import ccxt
import pandas as pd
import numpy as np
import vectorbt as vbt
from datetime import datetime, timezone
from pathlib import Path

CACHE_DIR = Path("data")
CACHE_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────
#  PARÁMETROS — acá ajustás la estrategia
# ──────────────────────────────────────────────────────────────────

SYMBOL      = "BTC/USDT"
TIMEFRAME   = "4h"
SINCE_DATE  = "2022-01-01"   # desde cuándo bajar datos
EXCHANGE_ID = "binance"

# Tendencia
MA_PERIOD   = 50             # MA para filtro de tendencia

# Detección de sweep
SWEEP_DROP_PCT  = 1.2        # caída mínima en % para considerar sweep
SWEEP_VOL_MULT  = 1.4        # volumen mínimo vs. promedio (rolling 20)
VOL_LOOKBACK    = 20         # velas para calcular volumen promedio

# Confirmación de entrada
CONFIRM_BODY_PCT = 0.3       # cuerpo de vela confirmadora mínimo en %

# Salida
TAKE_PROFIT_PCT  = 3.0       # % de ganancia para cerrar
STOP_LOSS_PCT    = 1.5       # % de pérdida para cerrar (sin apalancamiento)
LEVERAGE         = 2.0       # apalancamiento aplicado al sizing

# Capital inicial
INIT_CASH        = 10_000    # USDT


# ──────────────────────────────────────────────────────────────────
#  1. DESCARGA DE DATOS
# ──────────────────────────────────────────────────────────────────

def fetch_ohlcv(symbol: str, timeframe: str, since_date: str, exchange_id: str) -> pd.DataFrame:
    """
    Descarga datos OHLCV con caché local en Parquet.
    Si ya existe caché, solo descarga velas nuevas desde el último timestamp.
    """
    cache_file = CACHE_DIR / f"{symbol.replace('/', '_')}_{timeframe}.parquet"

    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True})

    # ── Cargar caché si existe
    if cache_file.exists():
        df_cached = pd.read_parquet(cache_file)
        last_ts   = df_cached.index[-1]
        since_ts  = int(last_ts.timestamp() * 1000) + 1
        print(f"[CACHE] Cargado desde disco: {len(df_cached)} velas (hasta {last_ts})")
    else:
        df_cached = None
        since_ts  = int(
            datetime.strptime(since_date, "%Y-%m-%d")
            .replace(tzinfo=timezone.utc)
            .timestamp() * 1000
        )
        print(f"[DATA] Sin caché — descargando desde {since_date}...")

    # ── Descargar solo lo nuevo
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
        df_new = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], unit="ms", utc=True)
        df_new.set_index("timestamp", inplace=True)
        print(f"[DATA] {len(df_new)} velas nuevas descargadas.")
    else:
        df_new = None
        print("[CACHE] Todo al día, sin velas nuevas.")

    # ── Combinar y guardar
    frames = [f for f in [df_cached, df_new] if f is not None]
    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")]
    df.sort_index(inplace=True)
    df.to_parquet(cache_file)
    print(f"[CACHE] Guardado: {cache_file} ({len(df)} velas totales)")

    return df

# ──────────────────────────────────────────────────────────────────
#  2. INDICADORES
# ──────────────────────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula todos los indicadores necesarios para la estrategia.
    """
    df = df.copy()

    # Tendencia: MA50
    df["ma50"] = df["close"].rolling(MA_PERIOD).mean()
    df["trend_up"] = df["close"] > df["ma50"]

    # Caída porcentual de la vela (open → close)
    df["candle_drop_pct"] = (df["open"] - df["close"]) / df["open"] * 100

    # Sombra inferior (posible sweep de liquidez)
    df["lower_wick_pct"] = (df["close"] - df["low"]) / df["open"] * 100

    # Volumen relativo
    df["vol_ma"] = df["volume"].rolling(VOL_LOOKBACK).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma"]

    # Cuerpo de vela en % (para confirmación)
    df["body_pct"] = abs(df["close"] - df["open"]) / df["open"] * 100

    return df


# ──────────────────────────────────────────────────────────────────
#  3. SEÑALES
# ──────────────────────────────────────────────────────────────────

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera señales de entrada y salida.

    Lógica de entrada (todas deben cumplirse):
      - Tendencia alcista (close > MA50)
      - Vela anterior: caída fuerte con volumen alto  →  sweep
      - Vela actual: cierre alcista con cuerpo definido  →  confirmación

    Lógica de salida:
      - Take profit o stop loss (gestionado por vectorbt via sl_stop / tp_stop)
    """
    df = df.copy()

    # ── Sweep: vela anterior bajista con volumen alto
    prev_drop      = df["candle_drop_pct"].shift(1) >= SWEEP_DROP_PCT
    prev_vol_spike = df["vol_ratio"].shift(1) >= SWEEP_VOL_MULT

    sweep_detected = prev_drop & prev_vol_spike

    # ── Confirmación: vela actual alcista con cuerpo definido
    bullish_confirm = (df["close"] > df["open"]) & (df["body_pct"] >= CONFIRM_BODY_PCT)

    # ── Filtro de tendencia
    in_uptrend = df["trend_up"]

    # ── Señal final
    df["entry"] = sweep_detected & bullish_confirm & in_uptrend

    print(f"[SIGNALS] Entradas detectadas: {df['entry'].sum()}")
    return df


# ──────────────────────────────────────────────────────────────────
#  4. BACKTEST
# ──────────────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame) -> vbt.Portfolio:
    """
    Ejecuta el backtest con vectorbt.
    Usa sl_stop y tp_stop para gestión automática de salidas.
    """
    entries = df["entry"]
    price   = df["close"]

    # Stop Loss y Take Profit como fracción del precio de entrada
    sl = STOP_LOSS_PCT / 100
    tp = TAKE_PROFIT_PCT / 100

    portfolio = vbt.Portfolio.from_signals(
        close        = price,
        entries      = entries,
        exits        = pd.Series(False, index=df.index),   # salida solo por SL/TP
        sl_stop      = sl,
        tp_stop      = tp,
        init_cash    = INIT_CASH,
        fees         = 0.001,        # 0.1% por operación (taker fee Binance)
        size         = 1.0,     # leverage como multiplicador de size
        size_type    = "percent",
        freq         = TIMEFRAME,
    )

    return portfolio


# ──────────────────────────────────────────────────────────────────
#  5. MÉTRICAS Y REPORTE
# ──────────────────────────────────────────────────────────────────

def print_report(portfolio: vbt.Portfolio) -> dict:
    """
    Imprime métricas clave y retorna dict con resultados.
    """
    stats = portfolio.stats()

    total_return    = portfolio.total_return() * 100
    sharpe          = portfolio.sharpe_ratio()
    max_dd          = portfolio.max_drawdown() * 100
    win_rate        = stats.get("Win Rate [%]", float("nan"))
    total_trades    = stats.get("Total Trades", 0)
    profit_factor   = stats.get("Profit Factor", float("nan"))
    avg_trade       = stats.get("Avg Winning Trade [%]", float("nan"))

    print("\n" + "=" * 55)
    print("  RESULTADOS DEL BACKTEST")
    print("=" * 55)
    print(f"  Período         : {SINCE_DATE} → hoy")
    print(f"  Símbolo         : {SYMBOL}  {TIMEFRAME}")
    print(f"  Capital inicial : ${INIT_CASH:,.0f} USDT")
    print(f"  Apalancamiento  : {LEVERAGE}x")
    print("-" * 55)
    print(f"  Retorno total   : {total_return:.2f}%")
    print(f"  Sharpe ratio    : {sharpe:.2f}")
    print(f"  Max drawdown    : {max_dd:.2f}%")
    print(f"  Total trades    : {total_trades}")
    print(f"  Win rate        : {win_rate:.1f}%")
    print(f"  Profit factor   : {profit_factor:.2f}")
    print("=" * 55 + "\n")

    return {
        "total_return_pct": total_return,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
    }


# ──────────────────────────────────────────────────────────────────
#  6. OPTIMIZACIÓN DE PARÁMETROS (grid search básico)
# ──────────────────────────────────────────────────────────────────

def optimize(df: pd.DataFrame):
    """
    Búsqueda de parámetros óptimos.
    ATENCIÓN: riesgo de overfitting — usar walk-forward validation.
    """
    import itertools

    sweep_drops  = [1.5, 2.0, 2.5, 3.0]
    vol_mults    = [1.5, 1.8, 2.0, 2.5]
    take_profits = [2.0, 3.0, 4.0, 5.0]
    stop_losses  = [1.0, 1.5, 2.0]

    best_sharpe  = -np.inf
    best_params  = {}
    results      = []

    combos = list(itertools.product(sweep_drops, vol_mults, take_profits, stop_losses))
    print(f"[OPT] Testeando {len(combos)} combinaciones...")

    for drop, vol, tp, sl in combos:
        # Override globals temporalmente
        global SWEEP_DROP_PCT, SWEEP_VOL_MULT, TAKE_PROFIT_PCT, STOP_LOSS_PCT
        SWEEP_DROP_PCT  = drop
        SWEEP_VOL_MULT  = vol
        TAKE_PROFIT_PCT = tp
        STOP_LOSS_PCT   = sl

        df_ind = compute_indicators(df)
        df_sig = generate_signals(df_ind)

        if df_sig["entry"].sum() < 5:   # descartar si hay muy pocas operaciones
            continue

        pf    = run_backtest(df_sig)
        sh    = pf.sharpe_ratio()
        ret   = pf.total_return() * 100
        dd    = pf.max_drawdown() * 100
        trades = pf.stats().get("Total Trades", 0)

        results.append({
            "sweep_drop": drop, "vol_mult": vol,
            "tp": tp, "sl": sl,
            "sharpe": sh, "return": ret, "drawdown": dd, "trades": trades
        })

        if sh > best_sharpe:
            best_sharpe = sh
            best_params = {"sweep_drop": drop, "vol_mult": vol, "tp": tp, "sl": sl}

    print(f"\n[OPT] Mejor Sharpe: {best_sharpe:.2f}")
    print(f"[OPT] Mejores parámetros: {best_params}")

    return pd.DataFrame(results).sort_values("sharpe", ascending=False)


# ──────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Datos
    df_raw = fetch_ohlcv(SYMBOL, TIMEFRAME, SINCE_DATE, EXCHANGE_ID)

    # 2. Indicadores
    df_ind = compute_indicators(df_raw)

    # 3. Señales
    df_sig = generate_signals(df_ind)

    # 4. Backtest
    portfolio = run_backtest(df_sig)

    # 5. Reporte
    metrics = print_report(portfolio)

    # 6. Plot (descomentar para ver gráfico)
    portfolio.plot().show()

    # 7. Optimización (descomentar para correr grid search)
    # opt_results = optimize(df_raw)
    # print(opt_results.head(10))