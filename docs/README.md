# Documentación del proyecto quant-engine

Este directorio contiene la documentación de diseño y referencia para los módulos del motor.

## Estructura

```
docs/
├── README.md                      ← este archivo (índice)
└── stock_screener/                ← Screener de acciones para largo plazo
    ├── 01_overview.md             ← Qué es y por qué existe
    ├── 02_strategy.md             ← Estrategia: por qué combinamos 4 factores
    ├── 03_scoring.md              ← Metodología detallada de cada score
    ├── 04_data_sources.md         ← Fuentes de datos y endpoints yfinance
    └── 05_architecture.md         ← Arquitectura del módulo y flujo de datos
```

## Convenciones

- Todo en español (idioma del proyecto)
- Cada documento es autocontenido pero referencia otros cuando aplica
- Los números/fórmulas se mantienen verificables: cuando citamos un paper o estudio,
  incluimos referencia para que pueda chequearse
- Los snippets de código son ilustrativos. La fuente de verdad es el código en `src/`

## Otros módulos

El motor original de cripto (Liquidity Sweep, Funding Flush, RSI Reversion) está
documentado en `CLAUDE.md` (raíz del proyecto). Este `docs/` agrupa los módulos
nuevos de mayor alcance que merecen documentación dedicada.
