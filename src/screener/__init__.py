"""Stock screener para inversion a largo plazo.

Sistema compuesto de 4-5 pilares:
- Calidad (Piotroski F-Score)
- Valoracion (drawdown + multiples vs propia historia)
- Acumulacion (Wyckoff: OBV, A/D Line, CMF, volume ratio)
- Momentum confirmador (12-1, distancia SMA200)
- Insider buying (opcional)

Ver docs/stock_screener/ para detalles de diseno.
"""
