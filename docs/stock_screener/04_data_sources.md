# Stock Screener — Fuentes de Datos

## Resumen

| Dato | Fuente | Costo | Notas |
|---|---|---|---|
| Universo (S&P 500) | Wikipedia | Gratis | Scrape de la página oficial; refresh trimestral |
| OHLCV diario (5 años) | yfinance | Gratis | Bulk batch, suficiente |
| Fundamentals trimestrales | yfinance | Gratis | "Best effort" — calidad mediana pero alcanza |
| Insider transactions | yfinance | Gratis | Datos limitados; opcional |
| Holders institucionales | yfinance | Gratis | Snapshot actual |

Todo el stack es **gratis**. Si en el futuro necesitamos más calidad o frecuencia,
escalamos a FMP/Tiingo/Polygon.

---

## 1. Universo de tickers — Wikipedia S&P 500

URL: `https://en.wikipedia.org/wiki/List_of_S%26P_500_companies`

Scrape de la primera tabla de la página. Devuelve ~500 tickers + sector + sub-industria.

```python
import pandas as pd
sp500_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
tickers = sp500_table['Symbol'].tolist()
```

**Cache**: guardamos `data/universe_sp500.csv` y refrescamos manualmente cada
trimestre (la composición del S&P 500 cambia ~20 veces al año, no es crítico).

**Universos alternativos** (para futuro):
- Russell 1000: requiere fuente paga o scrape de iShares IWB holdings
- NASDAQ 100: scrape de Wikipedia similar

---

## 2. yfinance — Endpoints utilizados

Importación: `import yfinance as yf`

Uso típico: `t = yf.Ticker("AAPL")` luego acceder a sus atributos.

### 2.1 `.history(period="5y", interval="1d", auto_adjust=True)`

OHLCV diario, ajustado por splits y dividendos.

**Columnas retornadas**: `Open, High, Low, Close, Volume, Dividends, Stock Splits`

**Usado para**:
- Cálculo de OBV, A/D Line, CMF (Score de Acumulación)
- Cálculo de SMA50, SMA200 (Score de Momentum)
- Drawdown desde 52w high y desde ATH (Score de Valoración)
- Momentum 12-1 (Score de Momentum)
- Volumen promedio 20d, 252d (Score de Acumulación)

**Optimización**: yfinance permite descargar múltiples tickers en un solo call:
```python
df = yf.download(tickers_list, period="5y", interval="1d", group_by='ticker', threads=True)
```
Esto es **mucho** más rápido que ticker por ticker.

### 2.2 `.fast_info`

Metadatos rápidos, retorna en milisegundos. Más confiable que `.info` pero con
menos campos.

**Campos consumidos**:
- `market_cap` → filtro de market cap
- `last_price`, `year_high`, `year_low` → drawdown 52w
- `ten_day_average_volume`, `three_month_average_volume` → liquidez
- `currency`, `exchange` → filtros de calidad (USD, NYSE/NASDAQ)
- `shares` → para confirmar shares outstanding

### 2.3 `.info`

Snapshot de fundamentals. **Limitación**: la calidad de yfinance.info varía
mucho entre tickers; algunos campos pueden estar `None` o `nan`. Hay que
manejar fallbacks.

**Campos consumidos**:

*Sector/clasificación*:
- `sector`, `industry`, `country`

*Valoración (Score de Valoración)*:
- `trailingPE`, `forwardPE`
- `priceToBook`
- `priceToSalesTrailing12Months`
- `enterpriseToEbitda`, `enterpriseToRevenue`
- `enterpriseValue`

*Rentabilidad (Score de Calidad)*:
- `profitMargins`, `operatingMargins`, `grossMargins`
- `returnOnAssets`, `returnOnEquity`
- `ebitdaMargins`

*Crecimiento*:
- `earningsGrowth`, `revenueGrowth`
- `earningsQuarterlyGrowth`

*Solvencia (Score de Calidad)*:
- `debtToEquity`
- `currentRatio`, `quickRatio`
- `totalCash`, `totalDebt`

*Tamaño y flotante*:
- `floatShares`, `sharesOutstanding`
- `heldPercentInsiders`, `heldPercentInstitutions`

*Dividendo*:
- `dividendYield`, `payoutRatio`
- `fiveYearAvgDividendYield`

*Analistas (info contextual, no entra en score)*:
- `recommendationKey` ("buy", "hold", "sell", etc.)
- `targetMeanPrice`, `numberOfAnalystOpinions`

### 2.4 `.quarterly_income_stmt`

Income statement trimestral, últimos ~4-8 trimestres.

**Filas que usamos**:
- `Total Revenue`
- `Gross Profit` → para `gross_margin = Gross Profit / Total Revenue`
- `Operating Income` (EBIT)
- `Net Income`
- `Diluted EPS`

**Para Piotroski**: comparamos último año (suma de 4 trimestres) vs año anterior.

### 2.5 `.quarterly_balance_sheet`

Balance trimestral, últimos ~4-8 trimestres.

**Filas que usamos**:
- `Total Assets`
- `Total Liabilities Net Minority Interest` (o similar)
- `Stockholders Equity`
- `Long Term Debt`
- `Total Debt`
- `Cash And Cash Equivalents`
- `Common Stock Shares Outstanding` (o `Ordinary Shares Number`)
- `Current Assets`, `Current Liabilities` → para current ratio

### 2.6 `.quarterly_cashflow`

Cashflow trimestral.

**Filas que usamos**:
- `Operating Cash Flow` (CFO)
- `Capital Expenditure` (capex; viene negativo, lo normalizamos)
- `Free Cash Flow = CFO - |Capex|` (lo calculamos nosotros)

### 2.7 `.insider_transactions` y `.insider_purchases` (opcional)

DataFrames con transacciones de insiders reportadas en Form 4 SEC.

**Limitación**: yfinance no siempre tiene esto, y cuando lo tiene puede estar
desactualizado. Si está vacío o `None`, omitimos el Score de Insider y
renormalizamos pesos.

**Columnas típicas**:
- `Insider`, `Position`, `Date Reported`
- `Transaction` ("Purchase", "Sale", "Option Exercise", etc.)
- `Shares`, `Value`

### 2.8 `.institutional_holders` y `.major_holders`

Snapshot de los mayores tenedores. Lo usamos solo como **info contextual** en el
output, no entra en el score.

---

## 3. Datos calculados localmente

A partir de OHLCV crudo de yfinance, calculamos **localmente** (sin más calls):

### Indicadores técnicos

- **OBV (On-Balance Volume)**: cumulative volume con signo del cambio de precio
- **A/D Line (Accumulation/Distribution)**: cumulative del CLV × volumen
- **CMF (Chaikin Money Flow, 20-period)**: rolling MFM × volumen / volumen
- **SMA50, SMA200**: medias móviles simples
- **ATR(14)**: Average True Range para contexto

### Métricas de precio

- **Drawdown desde 52w high**: `(high_52w - close) / high_52w`
- **Drawdown desde ATH**: `(ath_5y - close) / ath_5y`
- **Momentum 12-1**: `close[t-21] / close[t-252] - 1`
- **Distancia a SMA200**: `(close - sma200) / sma200`

### Percentiles históricos

Para valoración, calculamos el percentil del múltiplo actual vs su propia
historia 5y. Como yfinance no da serie histórica de P/E directamente,
**reconstruimos** los múltiplos a posteriori:

```
P/E histórico ≈ price_histórico / EPS_TTM_de_ese_momento
```

Esto requiere combinar `.history()` con datos trimestrales de earnings. Para
el MVP, podemos usar una aproximación simplificada (P/E actual vs distribución
de los últimos N puntos del 5y) o calcular los múltiplos snapshot y comparar
contra una distribución estática.

**Alternativa simple para el MVP**: calcular el percentil del **drawdown desde
ATH** como proxy de "qué tan barato está vs su propia historia", sin necesitar
serie histórica de múltiplos. Esto pierde precisión pero es robusto y fácil.

---

## 4. Estrategia de cache

Crítico para no martillar yfinance y para iteraciones rápidas:

```
data/
├── universe_sp500.csv              # tickers + sector (refresh trimestral)
├── ohlcv/
│   ├── AAPL.parquet               # OHLCV 5y por ticker
│   ├── MSFT.parquet
│   └── ...
├── fundamentals/
│   ├── AAPL_info.json             # .info dict
│   ├── AAPL_qis.parquet           # quarterly income stmt
│   ├── AAPL_qbs.parquet           # quarterly balance sheet
│   ├── AAPL_qcf.parquet           # quarterly cashflow
│   ├── AAPL_insiders.parquet      # insider transactions (si hay)
│   └── ...
└── meta/
    └── last_refresh.json          # timestamps por ticker
```

### Política de refresh

- **OHLCV**: incremental — al refrescar, solo bajamos las velas nuevas desde la
  última fecha cacheada (similar al fetcher de cripto en `src/data/fetcher.py`).
- **Fundamentals trimestrales**: refrescar máximo 1 vez por semana. No cambian
  hasta que reportan earnings.
- **`.info`**: refrescar 1 vez por semana. Algunos campos como market_cap
  cambian a diario pero los ratios fundamentales no.
- **Insider data**: refrescar 1 vez por semana.

---

## 5. Manejo de errores y fallbacks

**Casos comunes**:

| Problema | Cómo se maneja |
|---|---|
| Ticker delisted o ya no en yfinance | Skip + log warning, no romper la corrida |
| `.info` vacío o con campos `None` | Usar fallbacks; si faltan demasiados campos, excluir del score |
| `.quarterly_*` vacío | Excluir del Score de Calidad (necesita historia trimestral) |
| Histórico < 2 años | Excluir del filtro duro |
| `.insider_transactions` vacío | Omitir Score de Insider, renormalizar pesos |
| Yahoo rate limit / API change | Reintentos con backoff, log y continuar |

**Reglas generales**:
- Nunca crashear toda la corrida por un ticker problemático
- Loggear cada exclusión con razón clara
- En el output final, mostrar cuántos tickers se procesaron vs cuántos se
  excluyeron y por qué

---

## 6. Costos y rate limits

yfinance no tiene rate limit oficial, pero en la práctica:
- ~2 requests/segundo es seguro
- Bulk download (`yf.download()`) puede bajar 500 tickers OHLCV en 1-2 min
- Fundamentals (`.info`, `.quarterly_*`) son por-ticker; ~1-2 seg cada uno
- 500 tickers fundamentales = ~10-20 min sin paralelización

Para acelerar:
- Paralelizar con `joblib` o `concurrent.futures` (4-8 workers)
- Cachear agresivamente
- Refresh incremental (no re-bajar todo cada vez)
