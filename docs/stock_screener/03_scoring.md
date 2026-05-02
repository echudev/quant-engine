# Stock Screener — Metodología de Scoring

Cada pilar produce un score normalizado **0-100** (donde 100 es el mejor). El score
compuesto final es una suma ponderada de los pilares.

---

## 1. Score de Calidad — Piotroski F-Score (0-9 → 0-100)

El F-Score original es 0-9 puntos. Lo escalamos a 0-100 multiplicando por
`100/9` ≈ 11.11.

### Los 9 criterios de Piotroski

Cada uno suma 1 punto si se cumple, 0 si no:

**Rentabilidad (4 puntos)**
1. **ROA positivo**: Net Income (último año) > 0
2. **CFO positivo**: Operating Cash Flow > 0
3. **ROA mejorando**: ROA actual > ROA año anterior
4. **Calidad de earnings**: CFO > Net Income (los earnings se respaldan con cash real)

**Solvencia (3 puntos)**
5. **Long-term debt cayendo**: deuda LP actual < deuda LP año anterior
   (o ratio deuda/activos cayendo)
6. **Current ratio mejorando**: current ratio actual > año anterior
7. **No emisión de acciones**: shares outstanding actual ≤ año anterior
   (no hay dilución)

**Eficiencia operativa (2 puntos)**
8. **Gross margin mejorando**: gross margin actual > año anterior
9. **Asset turnover mejorando**: revenue/assets actual > año anterior

### Interpretación

- F-Score ≥ 8: empresa de muy alta calidad, mejorando en casi todo
- F-Score 6-7: calidad decente
- F-Score ≤ 4: deterioro significativo, probablemente *value trap*

### Pseudocódigo

```python
def piotroski_score(financials_current, financials_prior):
    score = 0
    # Rentabilidad
    score += int(financials_current.net_income > 0)
    score += int(financials_current.cfo > 0)
    score += int(financials_current.roa > financials_prior.roa)
    score += int(financials_current.cfo > financials_current.net_income)
    # Solvencia
    score += int(financials_current.lt_debt_ratio < financials_prior.lt_debt_ratio)
    score += int(financials_current.current_ratio > financials_prior.current_ratio)
    score += int(financials_current.shares_out <= financials_prior.shares_out)
    # Eficiencia
    score += int(financials_current.gross_margin > financials_prior.gross_margin)
    score += int(financials_current.asset_turnover > financials_prior.asset_turnover)

    return score * (100 / 9)  # → 0-100
```

---

## 2. Score de Valoración (0-100)

Mide cuán "en oferta" está la acción respecto a su **propia historia** y a su
máximo histórico (no respecto al sector — eso introduce sesgos).

### Componentes

| Componente | Peso | Cómo se mide |
|---|---|---|
| Drawdown desde ATH | 30% | `(ATH - precio_actual) / ATH × 100` truncado a [0, 70] |
| Percentil P/E vs propia historia 5y | 20% | `100 - rank_pct(P/E_actual, P/E_historia_5y)` |
| Percentil P/B vs propia historia 5y | 20% | `100 - rank_pct(P/B_actual, P/B_historia_5y)` |
| Percentil EV/EBITDA vs propia historia 5y | 20% | `100 - rank_pct(EV/EBITDA_actual, EV/EBITDA_historia_5y)` |
| Percentil P/FCF vs propia historia 5y | 10% | Igual, si el dato está disponible |

**Por qué `100 - rank_pct`**: queremos que un múltiplo bajo = score alto. Si
P/E está en el percentil 10 de su historia (muy bajo), score = 90.

**Drawdown truncado a 70**: si una empresa cayó 90% desde el ATH, no le damos
necesariamente score perfecto en valoración — puede estar quebrada. El cap a
70 reconoce que más drawdown no necesariamente es mejor.

### Pseudocódigo

```python
def valuation_score(ticker_data, history_5y):
    drawdown_pct = min((ATH - price_now) / ATH * 100, 70)
    drawdown_score = (drawdown_pct / 70) * 100

    pe_score = 100 - percentile_rank(pe_now, history_5y.pe)
    pb_score = 100 - percentile_rank(pb_now, history_5y.pb)
    ev_ebitda_score = 100 - percentile_rank(ev_ebitda_now, history_5y.ev_ebitda)
    pfcf_score = 100 - percentile_rank(pfcf_now, history_5y.pfcf) if pfcf_available else None

    return weighted_average([
        (drawdown_score, 0.30),
        (pe_score, 0.20),
        (pb_score, 0.20),
        (ev_ebitda_score, 0.20),
        (pfcf_score, 0.10) if pfcf_score else None,
    ])
```

**Manejo de casos especiales**:
- Si P/E es negativo (empresa con pérdidas): excluimos ese componente y
  redistribuimos pesos
- Si la historia 5y no está disponible: usamos lo que haya, mínimo 2 años
- Si todos los múltiplos son no-disponibles: score de valoración = solo drawdown

---

## 3. Score de Acumulación (Wyckoff) (0-100)

El score más "técnico". Detecta flujo de capital entrante mientras el precio no
ha despegado todavía.

### Componentes

| Componente | Peso | Cómo se mide |
|---|---|---|
| OBV slope positivo (60d) | 25% | Pendiente lineal del OBV últimos 60 días, normalizada |
| Divergencia OBV vs precio | 25% | OBV slope > 0 mientras price slope ≤ 0 (o muy pequeña) |
| CMF(20) promedio últimos 30d > 0 | 20% | Promedio CMF, escalado a [0, 100] |
| Volume ratio 20d/252d | 15% | `vol_avg_20d / vol_avg_252d`, > 1.0 da score positivo |
| A/D Line slope positivo | 15% | Pendiente lineal A/D Line últimos 60 días |

### Detalle por componente

**OBV slope (25%)**
```
OBV[t] = OBV[t-1] + volume[t]   if close[t] > close[t-1]
       = OBV[t-1] - volume[t]   if close[t] < close[t-1]
       = OBV[t-1]               if close[t] == close[t-1]
```
Calculamos regresión lineal del OBV sobre los últimos 60 días. Pendiente positiva
y significativa → score alto.

**Divergencia OBV vs precio (25%)** — el componente más característico de Wyckoff
```
price_slope_60d  = pendiente lineal del close últimos 60d
obv_slope_60d    = pendiente lineal del OBV últimos 60d

if obv_slope > 0 AND price_slope ≤ 0:
    divergencia bullish clara → score 100
elif obv_slope > 0 AND price_slope > 0 (ambos suben):
    confirmación → score 60
elif obv_slope ≤ 0 AND price_slope ≤ 0 (ambos bajan):
    distribución/agotamiento → score 20
else:
    sin señal → score 0
```

**CMF (Chaikin Money Flow) (20%)**
```
CMF(20) = sum_20d( ((C - L) - (H - C)) / (H - L) × volume ) / sum_20d(volume)
```
Promediamos CMF de los últimos 30 días. CMF promedio > 0.1 sostenido → score 100.
CMF < -0.1 → score 0.

**Volume ratio (15%)**
```
vol_ratio = avg_volume_20d / avg_volume_252d

ratio < 0.8     → score 0    (interés cayendo, mala señal)
ratio 0.8-1.0   → score 30
ratio 1.0-1.2   → score 60
ratio 1.2-1.5   → score 85
ratio > 1.5     → score 100  (interés disparado)
```

**A/D Line slope (15%)**
```
A/D = sum( ((C - L) - (H - C)) / (H - L) × volume )
```
Misma lógica que OBV — pendiente lineal de los últimos 60 días.

### Pseudocódigo

```python
def accumulation_score(ohlcv_252d):
    obv = compute_obv(ohlcv_252d)
    ad = compute_ad_line(ohlcv_252d)
    cmf = compute_cmf(ohlcv_252d, period=20)

    # OBV slope normalizado
    obv_slope = linear_slope(obv[-60:])
    obv_slope_score = sigmoid_normalize(obv_slope) * 100

    # Divergencia
    price_slope = linear_slope(ohlcv_252d.close[-60:])
    if obv_slope > 0 and price_slope <= 0:
        divergence_score = 100
    elif obv_slope > 0 and price_slope > 0:
        divergence_score = 60
    elif obv_slope <= 0 and price_slope <= 0:
        divergence_score = 20
    else:
        divergence_score = 0

    # CMF
    cmf_avg_30d = cmf[-30:].mean()
    cmf_score = clip((cmf_avg_30d + 0.1) / 0.2 * 100, 0, 100)

    # Volume ratio
    vol_ratio = ohlcv_252d.volume[-20:].mean() / ohlcv_252d.volume[-252:].mean()
    vol_score = bucket_score(vol_ratio, [0.8, 1.0, 1.2, 1.5], [0, 30, 60, 85, 100])

    # A/D slope
    ad_slope = linear_slope(ad[-60:])
    ad_slope_score = sigmoid_normalize(ad_slope) * 100

    return weighted_average([
        (obv_slope_score, 0.25),
        (divergence_score, 0.25),
        (cmf_score, 0.20),
        (vol_score, 0.15),
        (ad_slope_score, 0.15),
    ])
```

---

## 4. Score de Momentum (0-100)

Confirmador. Da puntos cuando el mercado empieza a validar la tesis.

### Componentes

| Componente | Peso | Cómo se mide |
|---|---|---|
| Momentum 12-1 (Jegadeesh-Titman) | 50% | Return 12 meses excluyendo último mes |
| Distancia a SMA200 | 30% | (price - SMA200) / SMA200, idealmente entre -10% y +20% |
| SMA50 slope | 20% | Pendiente de SMA50 últimos 30 días |

### Detalle

**Momentum 12-1 (50%)**
```
mom_12_1 = (close[t-21] - close[t-252]) / close[t-252]
```
Excluimos el último mes (~21 días) para evitar el efecto de reversión a corto plazo.

```
mom_12_1 < -0.20  → score 0    (todavía en caída fuerte)
mom_12_1 -0.20 a 0 → score 40   (cerca de tocar fondo)
mom_12_1 0 a 0.20  → score 80   (recuperando)
mom_12_1 > 0.20    → score 100  (tendencia clara)
```

**Distancia a SMA200 (30%)** — queremos cerca o recién arriba, no lejos
```
distance = (price - SMA200) / SMA200

distance < -0.30   → score 20   (muy abajo, todavía bear)
distance -0.30 a -0.10 → score 50   (zona de oportunidad pero riesgo)
distance -0.10 a 0     → score 80   (justo bajo, prometedor)
distance 0 a 0.20      → score 100  (recién arriba — ideal)
distance > 0.20        → score 60   (muy estirado arriba)
```

**SMA50 slope (20%)**
```
slope = (SMA50[t] - SMA50[t-30]) / SMA50[t-30]

slope < -0.05      → score 0
slope -0.05 a 0    → score 40
slope 0 a 0.05     → score 80
slope > 0.05       → score 100
```

---

## 5. Score de Insider Buying (bonus, 0-100)

### Componentes

| Componente | Peso | Cómo se mide |
|---|---|---|
| # de insiders comprando últimos 90d | 40% | Cluster detection: ≥3 distintos en 90d → bonus |
| $ neto de compras vs ventas (180d) | 40% | (USD comprado - USD vendido) / market_cap |
| Compras de C-suite (CEO/CFO/COO) últimos 180d | 20% | Pondera más alto que directores externos |

**Threshold básico**: si hay 0 compras en 180 días, score = 0. Si hay actividad
significativa de cluster + ejecutivos clave, score puede llegar a 100.

**Importante**: yfinance da datos de insider_transactions limitados. Si los datos
no están disponibles, este pilar se omite y los pesos restantes se renormalizan.

---

## Score Compuesto Final

```
score_total = (
    0.25 × score_calidad +
    0.25 × score_valoracion +
    0.25 × score_acumulacion +
    0.15 × score_momentum +
    0.10 × score_insider
)
```

Si insider data no está disponible, los pesos restantes se renormalizan a 1.0:

```
score_total = (
    0.278 × score_calidad +
    0.278 × score_valoracion +
    0.278 × score_acumulacion +
    0.166 × score_momentum
)
```

### Filtros duros (antes de scorear)

Antes de calcular scores, descartamos tickers que no cumplan:
- Market cap > $10B (configurable)
- ADV (volumen promedio diario) > $50M en USD
- Histórico mínimo de 2 años de datos disponibles
- País listado (filtro opcional para evitar OTC/ADRs raros)

### Ranking final

Ordenamos por `score_total` descendente y mostramos top N (default 30).

Para cada candidato del top, mostramos:
- Ticker, sector, market cap
- Scores individuales (4-5 columnas)
- Score compuesto
- Métricas raw clave para revisión: P/E, P/B, ROE, F-Score, drawdown, dividend yield
