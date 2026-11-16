from __future__ import annotations
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")


def out(task: str) -> Path:
    d = ROOT / task / "outputs"
    (d / "tables").mkdir(parents=True, exist_ok=True)
    (d / "figures").mkdir(parents=True, exist_ok=True)
    return d


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def workbook() -> Path:
    import os
    p = Path(os.environ.get("CORTEXHR_DATA", ROOT / "data" / "raw" / "02_CortexHR_AI_ML_Dataset.xlsx"))
    if not p.exists():
        raise SystemExit(f"Dataset not found at {p}. Put 02_CortexHR_AI_ML_Dataset.xlsx in data/raw/ (see data/raw/README.md).")
    return p
