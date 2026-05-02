"""Score compuesto ponderado.

Combina los scores individuales (calidad, valoracion, acumulacion, momentum,
insider) con pesos configurables. Si alguno es None, renormaliza los pesos
restantes a 1.0.
"""
from __future__ import annotations

DEFAULT_WEIGHTS = {
    "quality":      0.25,
    "valuation":    0.25,
    "accumulation": 0.25,
    "momentum":     0.15,
    "insider":      0.10,
}


def composite_score(
    scores: dict[str, float | None],
    weights: dict[str, float] | None = None,
) -> float:
    """Score compuesto 0-100.

    `scores` es un dict como {"quality": 67.0, "valuation": 80.0,
    "accumulation": 75.0, "momentum": 50.0, "insider": None}.
    Pilares con score=None se omiten y los pesos restantes se renormalizan.
    """
    weights = weights or DEFAULT_WEIGHTS
    used: dict[str, tuple[float, float]] = {}  # name -> (score, weight)

    for name, w in weights.items():
        s = scores.get(name)
        if s is None:
            continue
        used[name] = (float(s), float(w))

    if not used:
        return 0.0

    total_w = sum(w for _, w in used.values())
    if total_w == 0:
        return 0.0

    weighted = sum(s * w for s, w in used.values())
    return float(weighted / total_w)
