# Stock Screener — Estrategia y Justificación

## Comparativa de estrategias para largo plazo

Estrategias con evidencia académica/empírica de outperformance vs índice, ordenadas
por solidez de la evidencia:

| Estrategia | Premium histórico vs S&P | Fuente principal |
|---|---|---|
| **Quality factor** (alta ROIC, ROE, baja deuda, márgenes estables) | +1.5 a 3%/año | Fama-French; Asness/AQR "Quality Minus Junk" (2013) |
| **Piotroski F-Score** (9 puntos fundamentals sobre value stocks) | +7.5%/año en value | Piotroski (2000), Journal of Accounting Research |
| **Magic Formula** (EBIT/EV + ROC ranking) | +5-8%/año (1988-2004) | Greenblatt, "The Little Book That Beats the Market" (2005) |
| **Momentum 12-1** (return 12 meses excluyendo último mes) | +8-10%/año | Jegadeesh & Titman (1993), Journal of Finance |
| **Insider buying clusters** (≥3 insiders comprando en 90 días) | +6%/año | Cohen, Malloy, Pomorski (2012), Journal of Finance |
| **Value puro** (low P/B, P/E) | +3-5%/año (flojo última década) | Fama-French (1992) |
| **Wyckoff / Volume Spread Analysis** | Sin estudios académicos rigurosos | Tradición práctica (Wyckoff, Tom Williams) |

### ¿Por qué Wyckoff no aparece como winner académico?

Wyckoff es **subjetivo y discrecional** en su forma original. Detectar una "spring",
una "test", o una "sign of strength" requiere juicio humano. Los estudios académicos
que intentaron sistematizarlo en reglas duras encontraron resultados mixtos.

**Pero**: Wyckoff sigue siendo la mejor herramienta práctica para **timing de
entrada** en activos ya seleccionados por fundamentals. Lo usamos para esa función,
no como sistema completo.

## Por qué Wyckoff solo no alcanza

### El problema de la *value trap*

Un activo puede mostrar acumulación clara y aún así caer durante años. Casos
históricos:

- **GE 2017-2020**: volumen alto en zona de "soporte" (~$25), terminó cayendo
  hasta $6 antes de recuperarse. El problema: deterioro estructural en GE Capital
  y power generation que el volumen no detectaba.
- **Macy's 2018-2020**: múltiples patrones Wyckoff, terminó cayendo a $5 durante COVID.
- **Boeing 2019-2024**: acumulación visible, pero los problemas estructurales
  (737 MAX, calidad, gobernanza) hicieron que la "acumulación" fuera prematura.

**La lección**: el volumen muestra que alguien está comprando, pero no garantiza
que esos compradores tengan razón. Necesitás un filtro fundamental que valide que
la empresa merece sobrevivir y prosperar.

## El sistema de 4 pilares

Combinamos **4 factores con bajísima correlación entre sí**, lo que reduce drásticamente
falsos positivos:

### Pilar 1 — Calidad (Quality / Piotroski F-Score)

**Para qué sirve**: filtrar empresas en deterioro fundamental. Rechaza casos donde
los earnings caen, el cashflow se deteriora, la deuda crece, los márgenes se
contraen, o la empresa diluye accionistas.

**Por qué F-Score específicamente**: combina rentabilidad, solvencia y eficiencia
operativa en un solo score 0-9. Está validado por décadas de estudios.

**Threshold típico**: F-Score ≥ 6 (de 9) para considerar.

### Pilar 2 — Valoración ("en oferta")

**Para qué sirve**: comprar caro mata returns. Aún empresas excelentes producen
returns mediocres si entrás en máximos.

**Cómo lo medimos**:
- Drawdown desde el ATH (más drawdown = más oferta)
- Múltiplos (P/E, P/B, EV/EBITDA, P/FCF) en percentil bajo respecto a su propia
  historia (la empresa está "barata para sí misma")
- No usamos múltiplos absolutos vs sector porque eso castiga industrias con
  múltiplos estructuralmente altos (tech)

### Pilar 3 — Acumulación (Wyckoff / Flow)

**Para qué sirve**: timing de entrada. Detectar que las manos fuertes están
acumulando antes que el precio se mueva.

**Cómo lo medimos**:
- **OBV** (On-Balance Volume) con pendiente positiva mientras el precio está
  plano o cae → divergencia bullish
- **A/D Line** (Accumulation/Distribution) con pendiente positiva
- **CMF (Chaikin Money Flow)** sostenido > 0 últimas 4-6 semanas
- **Volume ratio**: volumen promedio últimos 20 días / volumen promedio últimos
  252 días > 1.0 (interés creciente)
- Detección opcional de **"spring" Wyckoff**: ruptura falsa de mínimo + recuperación

### Pilar 4 — Momentum confirmador

**Para qué sirve**: evitar *catching falling knives*. Esperar a que el mercado
empiece a confirmar la tesis antes de entrar.

**Cómo lo medimos**:
- Return 12 meses excluyendo el último mes (momentum de Jegadeesh & Titman)
- Distancia del precio a SMA200 (no demasiado abajo, ideal recién recuperándola)
- Slope de SMA50 girando a positiva

**Importante**: el momentum es *confirmador*, no excluyente. Damos puntos si está
presente, pero no descartamos automáticamente activos sin momentum si los otros
3 pilares son fuertes (porque a veces se entra antes de la confirmación).

## Pilar opcional 5 — Insider buying (bonus)

Si está disponible y es relevante, sumamos puntos extra cuando insiders compran
agresivamente. Cohen-Malloy-Pomorski (2012) muestra que insiders no rutinarios
(no algorítmicos, no recompras) que compran en ventana de 90 días generan +6%/año
de outperformance. Es un **booster**, no un requisito.

## Pesos sugeridos del score compuesto

```
score_total = (
    0.25 × score_calidad +
    0.25 × score_valoracion +
    0.25 × score_acumulacion +
    0.15 × score_momentum +
    0.10 × score_insider
)
```

**Justificación**:
- Calidad, valoración y acumulación son los 3 pilares principales (75% del peso)
- Momentum es confirmador (15%)
- Insider es bonus (10%) y a veces no hay datos disponibles

Estos pesos son configurables. Casos donde tendría sentido ajustar:
- Inversor más value: subir valoración a 0.35, bajar momentum a 0.05
- Inversor más quality (estilo Buffett): subir calidad a 0.35
- Si los datos de insider son demasiado ruidosos: bajar a 0.05 y redistribuir

## Ejemplo trabajado: INTC en Octubre 2024

| Pilar | Componentes | Score estimado |
|---|---|---|
| Calidad | F-Score 6/9 (CFO positivo, ROA mejorando, no dilución) | ~67 |
| Valoración | -65% ATH, P/B en percentil 5 de su historia 10y | ~90 |
| Acumulación | OBV divergente positivo, CMF > 0 sostenido, volumen 20d/252d > 1.2 | ~80 |
| Momentum | 12-1 negativo todavía, pero precio recuperando SMA50 | ~40 |
| Insider | Compras institucionales aumentando | ~50 |
| **Total ponderado** | | **~70** |

Un score compuesto de 70+ típicamente entraría en el top 30 de un universo
S&P 500 cuando el mercado general está caro.

## Referencias

- Piotroski, J. (2000). "Value Investing: The Use of Historical Financial
  Statement Information to Separate Winners from Losers Among Value Stocks".
  *Journal of Accounting Research*.
- Asness, C., Frazzini, A., Pedersen, L. (2013). "Quality Minus Junk". AQR.
- Jegadeesh, N., Titman, S. (1993). "Returns to Buying Winners and Selling
  Losers: Implications for Stock Market Efficiency". *Journal of Finance*.
- Cohen, L., Malloy, C., Pomorski, L. (2012). "Decoding Inside Information".
  *Journal of Finance*.
- Greenblatt, J. (2005). "The Little Book That Beats the Market".
- Wyckoff, R. (1931). "The Richard D. Wyckoff Method of Trading and
  Investing in Stocks".
