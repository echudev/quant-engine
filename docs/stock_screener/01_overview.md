# Stock Screener — Overview

## Qué es

Un screener para identificar **acciones atractivas para inversión a largo plazo**
combinando factores de calidad, valoración, acumulación (Wyckoff) y momentum.

## Objetivo

Dado un universo (S&P 500 por defecto), devolver un ranking de los **top N candidatos**
según un score compuesto de 4 dimensiones independientes, cada una con base académica
o práctica documentada.

El output esperado es una tabla rankeada con:
- Ticker, sector, market cap
- Score por dimensión (0-100 normalizados)
- Score compuesto final
- Métricas clave para revisión manual (P/E, ROE, drawdown, F-Score, etc.)

## Caso de uso típico

> "Quiero encontrar acciones grandes, baratas, con calidad fundamental sólida y
> mostrando signos tempranos de acumulación. INTC en octubre 2024 sería el ejemplo
> arquetípico: large cap, drawdown profundo, fundamentals decentes, OBV divergente
> alcista."

El screener corre semanal/mensualmente, no es trading de alta frecuencia. La salida
alimenta una watchlist para análisis manual (leer 10-Ks, escuchar earnings calls)
antes de tomar posición.

## Lo que NO es

- **No es un sistema de trading automático**. El screener identifica candidatos;
  la decisión de comprar requiere análisis humano del contexto (¿la empresa
  enfrenta una disrupción estructural? ¿hay litigios? ¿guidance comprometido?).
- **No es market timing del índice**. No te dice cuándo entrar al mercado en
  general, solo qué activos son interesantes en cada momento.
- **No es para small caps especulativos**. El filtro de market cap (>$10B por
  defecto) excluye micro/small caps donde la dinámica es muy distinta.

## Tesis central

Cada factor por sí solo tiene debilidades documentadas:

| Factor | Debilidad si va solo |
|---|---|
| Solo valoración (low P/E) | *Value trap*: empresas baratas que merecen estarlo |
| Solo calidad (alta ROIC) | Pagás demasiado por la calidad — entrada en máximos |
| Solo Wyckoff/acumulación | Sin filtro fundamental, comprás acumulación en empresas en deterioro estructural |
| Solo momentum | Comprás extension; entrás tarde en tendencias maduras |

**La combinación cubre las debilidades cruzadas**:

> Calidad alta + valoración baja + acumulación detectada + momentum confirmador
> = empresas sólidas, en oferta, con flujo de dinero entrando, y precio
> empezando a confirmar la tesis.

## Ejemplo guía: INTC en Octubre 2024

| Dimensión | Estado en Oct 2024 | Cumple? |
|---|---|---|
| **Market cap** | ~$80B | ✅ Large cap |
| **Valoración** | -65% del ATH histórico, P/B en mínimos de 10 años | ✅ "En oferta" |
| **Calidad (Piotroski)** | F-Score ~5-6 (CFO+, márgenes mejorando, debt estable) | ✅ Mediano-alto |
| **Acumulación (Wyckoff)** | Volumen alto + precio lateral 4-6 meses, OBV divergente | ✅ Acumulación clara |
| **Momentum confirmador** | Precio recuperando SMA50, 6m return saliendo de mínimos | ✅ Confirmando |

Un screener que combine los 4 factores **habría incluido INTC** en su top 30 en
ese momento. Un screener basado solo en momentum la habría excluido (estaba en
mínimos). Uno basado solo en valoración la habría incluido junto con muchos
*value traps* sin criterio para distinguir.

## Próximos pasos

1. Leer `02_strategy.md` para el detalle de por qué cada factor entra
2. Leer `03_scoring.md` para la metodología exacta de cómo se calcula cada score
3. Leer `04_data_sources.md` para qué endpoints de yfinance se usan
4. Leer `05_architecture.md` para el diseño del módulo
