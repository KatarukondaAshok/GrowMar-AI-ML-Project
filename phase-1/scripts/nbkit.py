"""Tiny helpers to build and execute notebooks from Python (used by the build_task*_notebook.py scripts)."""
import subprocess
from pathlib import Path

import nbformat as nbf

SETUP = '''import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
# Find the project root (the folder that contains `src/`) no matter where the notebook is opened from
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src").exists())
sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import display

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
DATA = ROOT / "data" / "raw" / "02_CortexHR_AI_ML_Dataset.xlsx"
TASK = ROOT / "{task_dir}"
OUT = TASK / "outputs"; FIG = OUT / "figures"; TAB = OUT / "tables"
for d in (FIG, TAB): d.mkdir(parents=True, exist_ok=True)
print("Project root:", ROOT)'''


def setup(task_dir):
    return SETUP.replace("{task_dir}", task_dir)


class NB:
    def __init__(self):
        self.cells = []

    def md(self, text):
        self.cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))

    def code(self, text):
        self.cells.append(nbf.v4.new_code_cell(text.strip("\n")))

    def save(self, path, execute=True):
        nb = nbf.v4.new_notebook()
        nb["cells"] = self.cells
        nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
        path = Path(path)
        nbf.write(nb, path)
        if execute:
            r = subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                                "--ExecutePreprocessor.timeout=600", str(path)], capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stderr[-3000:])
                raise SystemExit("notebook failed")
        print("built", path)
