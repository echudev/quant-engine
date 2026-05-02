# Stock Screener — Arquitectura del módulo

## Estructura de directorios

```
src/screener/
├── __init__.py
├── universe.py              # carga de S&P 500 desde Wikipedia + cache
├── fetcher.py               # wrapper yfinance + cache (parquet/json)
├── indicators.py            # OBV, A/D Line, CMF, SMA, momentum, etc.
├── scores/
│   ├── __init__.py
│   ├── quality.py           # Piotroski F-Score
│   ├── valuation.py         # drawdown + percentiles de múltiplos
│   ├── accumulation.py      # Wyckoff: OBV slope, divergencia, CMF, vol ratio
│   ├── momentum.py          # 12-1, distancia SMA200, slope SMA50
│   └── insider.py           # opcional: cluster + $ neto + C-suite
├── composite.py             # combina los 5 scores con pesos
├── filters.py               # filtros duros (market cap, ADV, historia mínima)
├── runner.py                # orquesta toda la corrida end-to-end
└── report.py                # genera la tabla CSV/markdown del output

config/
└── screener.toml            # parámetros del screener (separado de settings.toml)

data/
├── universe_sp500.csv
├── ohlcv/<TICKER>.parquet
└── fundamentals/<TICKER>_*.{json,parquet}

results/
└── screener_<timestamp>.csv

main_screener.py             # entrypoint top-level (paralelo a main.py existente)
```

**Decisión de diseño**: módulo `src/screener/` independiente del motor de cripto
(`src/strategies/`, `src/backtest/`, `src/data/fetcher.py`). Comparten poco.

## Flujo de datos end-to-end

```
1. universe.load_sp500()
       ↓
   Lista de ~500 tickers + sectores

2. fetcher.fetch_all(tickers)
       ↓ (yfinance bulk + per-ticker)
   data/ohlcv/<T>.parquet
   data/fundamentals/<T>_*.{json,parquet}

3. filters.apply_hard_filters(tickers, market_cap_min, adv_min, ...)
       ↓
   Tickers que pasan filtros duros (~200-300)

4. Para cada ticker en paralelo:
     a. indicators.compute(ohlcv) → OBV, A/D, CMF, SMA, momentum
     b. scores.quality.piotroski_score(quarterly_data)
     c. scores.valuation.score(info, history)
     d. scores.accumulation.score(ohlcv, indicators)
     e. scores.momentum.score(ohlcv, indicators)
     f. scores.insider.score(insider_txns)  [opcional]
       ↓
   Dict por ticker con todos los scores + métricas raw

5. composite.weighted_score(scores_per_ticker, weights)
       ↓
   Score compuesto 0-100 por ticker

6. report.generate(results, top_n=30)
       ↓
   results/screener_<ts>.csv + tabla en consola
```

## Componentes por archivo

### `src/screener/universe.py`

```python
def load_sp500(cache_path="data/universe_sp500.csv", refresh=False) -> pd.DataFrame:
    """
    Devuelve DataFrame con columns: [Symbol, Security, GICS_Sector, GICS_Sub_Industry].
    Refresh manual (no semanal) — el universo cambia ~20 veces al año.
    """
```

### `src/screener/fetcher.py`

```python
def fetch_ohlcv(ticker: str, period: str = "5y") -> pd.DataFrame: ...
def fetch_ohlcv_bulk(tickers: list[str], period: str = "5y") -> dict[str, pd.DataFrame]: ...
def fetch_fundamentals(ticker: str) -> dict: ...
def fetch_insider(ticker: str) -> pd.DataFrame | None: ...

def refresh_all(tickers: list[str], force: bool = False) -> None:
    """
    Orquesta refresh respetando políticas:
    - OHLCV: incremental siempre
    - .info y trimestrales: 1x/semana max
    - insiders: 1x/semana max
    """
```

Cache en disco con timestamps en `data/meta/last_refresh.json`.

### `src/screener/indicators.py`

Funciones puras sobre DataFrames OHLCV:

```python
def obv(df: pd.DataFrame) -> pd.Series: ...
def ad_line(df: pd.DataFrame) -> pd.Series: ...
def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series: ...
def sma(series: pd.Series, period: int) -> pd.Series: ...
def linear_slope(series: pd.Series, n: int) -> float: ...
def momentum_12_1(close: pd.Series) -> float: ...
def drawdown_from_ath(close: pd.Series) -> float: ...
def drawdown_from_52w_high(close: pd.Series) -> float: ...
```

### `src/screener/scores/quality.py`

```python
@dataclass
class QualityResult:
    f_score: int           # 0-9
    score_normalized: float  # 0-100
    components: dict       # detalle de qué criterios cumplió

def piotroski_score(qis: pd.DataFrame, qbs: pd.DataFrame, qcf: pd.DataFrame) -> QualityResult:
    """
    Compara último año (4 trimestres) vs año anterior.
    Retorna F-Score y desglose de los 9 criterios.
    """
```

### `src/screener/scores/valuation.py`

```python
@dataclass
class ValuationResult:
    score: float           # 0-100
    drawdown_ath_pct: float
    pe_percentile: float | None
    pb_percentile: float | None
    ev_ebitda_percentile: float | None
    pfcf_percentile: float | None

def valuation_score(info: dict, ohlcv: pd.DataFrame, ...) -> ValuationResult: ...
```

Para el MVP usamos drawdown como driver principal (peso 50% si los percentiles
históricos no son confiables) y los múltiplos crudos sin percentil histórico.
Iteramos después si hace falta más sofisticación.

### `src/screener/scores/accumulation.py`

```python
@dataclass
class AccumulationResult:
    score: float           # 0-100
    obv_slope_60d: float
    price_slope_60d: float
    divergence_kind: str   # "bullish_div", "bullish_confirm", "distribution", "none"
    cmf_avg_30d: float
    volume_ratio_20_252: float
    ad_slope_60d: float

def accumulation_score(ohlcv: pd.DataFrame) -> AccumulationResult: ...
```

### `src/screener/scores/momentum.py`

```python
@dataclass
class MomentumResult:
    score: float           # 0-100
    momentum_12_1: float
    distance_to_sma200: float
    sma50_slope: float

def momentum_score(ohlcv: pd.DataFrame) -> MomentumResult: ...
```

### `src/screener/scores/insider.py`

```python
@dataclass
class InsiderResult:
    score: float | None    # 0-100, None si no hay datos
    n_buyers_90d: int
    net_purchase_usd_180d: float
    csuite_buys_180d: int

def insider_score(insider_df: pd.DataFrame | None, market_cap: float) -> InsiderResult: ...
```

### `src/screener/composite.py`

```python
DEFAULT_WEIGHTS = {
    "quality": 0.25,
    "valuation": 0.25,
    "accumulation": 0.25,
    "momentum": 0.15,
    "insider": 0.10,
}

def composite_score(scores: dict, weights: dict = DEFAULT_WEIGHTS) -> float:
    """
    Si insider score es None, renormaliza los pesos restantes a 1.0.
    """
```

### `src/screener/filters.py`

```python
def passes_hard_filters(
    info: dict,
    ohlcv: pd.DataFrame,
    market_cap_min: float = 10e9,
    adv_min_usd: float = 50e6,
    history_min_days: int = 504,
) -> tuple[bool, str | None]:
    """
    Devuelve (pasa, razón_si_no_pasa).
    """
```

### `src/screener/runner.py`

Orquesta todo. Equivalente al `main.py` del motor de cripto, pero para screener.

```python
def run_screener(
    universe: str = "sp500",
    top_n: int = 30,
    refresh_data: bool = False,
    weights: dict | None = None,
) -> pd.DataFrame: ...
```

### `src/screener/report.py`

```python
def to_csv(df: pd.DataFrame, path: str) -> None: ...
def to_console(df: pd.DataFrame, top_n: int = 30) -> None:
    """Tabla bonita con tabulate o similar."""
def to_markdown(df: pd.DataFrame, path: str) -> None: ...
```

## Configuración: `config/screener.toml`

Separado de `config/settings.toml` para no mezclar con la lógica de cripto.

```toml
[universe]
source = "sp500"        # "sp500" | "russell1000" | "custom"
custom_tickers_path = "" # solo si source = "custom"

[filters]
market_cap_min = 10_000_000_000   # 10B USD
adv_min_usd    = 50_000_000        # 50M USD
history_min_days = 504             # 2 años

[weights]
quality      = 0.25
valuation    = 0.25
accumulation = 0.25
momentum     = 0.15
insider      = 0.10

[output]
top_n        = 30
csv_path     = "results/screener_{ts}.csv"
markdown_path = "results/screener_{ts}.md"
print_console = true

[refresh]
ohlcv_max_age_days        = 1
fundamentals_max_age_days = 7
insider_max_age_days      = 7
```

## Entrypoint: `main_screener.py`

Top-level script paralelo a `main.py` existente:

```python
from src.screener.runner import run_screener
from src.screener.report import to_csv, to_console, to_markdown

def main():
    df = run_screener()
    to_console(df, top_n=30)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    to_csv(df, f"results/screener_{ts}.csv")
    to_markdown(df, f"results/screener_{ts}.md")

if __name__ == "__main__":
    main()
```

Comando: `python main_screener.py`

## Paralelización

- **Bulk download de OHLCV**: usar `yf.download(tickers_list, threads=True)` para
  bajar 500 tickers de OHLCV en 1-2 minutos.
- **Fundamentals per-ticker**: usar `joblib.Parallel(n_jobs=4)` para 4 workers
  paralelos. yfinance es rate-limited en `.info` así que 4 workers es seguro.
- **Cálculo de scores**: paralelo con `joblib.Parallel(n_jobs=-1)` ya que es
  CPU-bound puro sobre datos en memoria.

## Testing manual del MVP

Sin test suite formal por ahora (el repo no la tiene). Validación manual:

1. Correr screener completo en S&P 500
2. Verificar que aparecen nombres "razonables" en el top 30 (mezcla de value
   stocks bien conocidos en oferta + algún beaten-down quality)
3. Verificar contra el ejemplo INTC (correr con `--as-of 2024-10-01` cuando
   se implemente backtesting histórico)
4. Manualmente revisar 3-5 candidatos del top: ¿realmente cumplen lo que el
   score dice?

## Roadmap incremental

**MVP (fase 1)**:
- Universo S&P 500
- Score de calidad (Piotroski) + valoración (drawdown + múltiplos snapshot)
- Score de acumulación + momentum
- Sin insider score (lo agregamos después si los datos están bien)
- Output CSV + consola

**Fase 2** (si todo funciona):
- Insider score con datos de yfinance (probar disponibilidad)
- Percentiles históricos de múltiplos (no solo snapshot)
- Detección de "spring" Wyckoff explícita
- Output markdown con gráfico embebido

**Fase 3** (escalar):
- Russell 1000 / Russell 3000
- Backtest histórico (¿el screener habría detectado INTC en Oct 2024?)
- Integración FMP free tier para validación cruzada de fundamentals
- Alertas: ejecutar semanal y notificar cambios en el top 30

## Dependencias adicionales (ya tenemos algunas)

Probablemente necesarias y a verificar en `pyproject.toml`:

- `yfinance` — fetcher principal (probablemente no instalado)
- `pandas`, `numpy` — ya instalado
- `pyarrow` — ya instalado (parquet)
- `joblib` — ya instalado
- `tabulate` — para output tabular en consola (probablemente no instalado)
- `lxml`, `beautifulsoup4` — para scrape de Wikipedia (`pd.read_html` los necesita)

## Integración con el resto del repo

El screener **no toca** el motor de cripto existente. Convive en paralelo:

- `main.py` → backtests de cripto (existente)
- `main_screener.py` → screener de acciones (nuevo)
- `src/strategies/` → estrategias de cripto (existente, sin cambios)
- `src/screener/` → módulo nuevo (no comparte código con cripto)
- `config/settings.toml` → cripto (existente)
- `config/screener.toml` → screener (nuevo)
- `data/` → comparten directorio raíz, subdirs separados
- `results/` → comparten directorio, prefijos distintos en filenames
