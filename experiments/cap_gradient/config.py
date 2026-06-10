"""
Config y paths del experimento cap_gradient.
Toda la parametrización vive en config/settings.toml [experiment.cap_gradient].
"""

import json
import shutil
import subprocess
import tomllib
from datetime import datetime, timezone
from pathlib import Path

ROOT          = Path(__file__).resolve().parents[2]
SETTINGS_PATH = ROOT / "config" / "settings.toml"
DATA_DIR      = ROOT / "data"
PREPARED_DIR  = DATA_DIR / "experiment_cap_gradient"
EXP_DIR       = ROOT / "results" / "experiment_cap_gradient"
PLOTS_DIR     = EXP_DIR / "plots"

UNIVERSE_PATH = EXP_DIR / "universe.parquet"
QA_PATH       = EXP_DIR / "data_qa.parquet"
RESULTS_PATH  = EXP_DIR / "results.parquet"
SUMMARY_PATH  = EXP_DIR / "symbol_summary.parquet"
ANALYSIS_PATH = EXP_DIR / "analysis.json"
REPORT_PATH   = EXP_DIR / "report.md"


def ensure_dirs() -> None:
    for d in (DATA_DIR, PREPARED_DIR, EXP_DIR, PLOTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_cfg() -> dict:
    with open(SETTINGS_PATH, "rb") as f:
        return tomllib.load(f)


def exp_cfg(cfg: dict | None = None) -> dict:
    cfg = cfg or load_cfg()
    return cfg["experiment"]["cap_gradient"]


def tier_for_rank(rank: int, exp: dict) -> tuple[int, float]:
    """(tier, cost multiplier) para un rank de OI 1-based."""
    for i, max_rank in enumerate(exp["tier_max_rank"]):
        if rank <= max_rank:
            return i + 1, float(exp["tier_cost_mult"][i])
    return len(exp["tier_max_rank"]), float(exp["tier_cost_mult"][-1])


def write_run_meta(extra: dict | None = None) -> None:
    """Snapshot de config + versiones + commit para reproducibilidad."""
    ensure_dirs()
    try:
        git_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        git_dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip())
    except Exception:
        git_hash, git_dirty = "unknown", None

    import ccxt, numpy, pandas, scipy, vectorbt
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit":    git_hash,
        "git_dirty":     git_dirty,
        "versions": {
            "pandas":    pandas.__version__,
            "numpy":     numpy.__version__,
            "vectorbt":  vectorbt.__version__,
            "ccxt":      ccxt.__version__,
            "scipy":     scipy.__version__,
        },
        **(extra or {}),
    }
    shutil.copy(SETTINGS_PATH, EXP_DIR / "config_snapshot.toml")
    meta_path = EXP_DIR / "run_meta.json"
    existing = json.loads(meta_path.read_text()) if meta_path.exists() else {"runs": []}
    existing["runs"].append(meta)
    meta_path.write_text(json.dumps(existing, indent=2))
