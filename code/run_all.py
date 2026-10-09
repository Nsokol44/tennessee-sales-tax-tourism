#!/usr/bin/env python3
"""Run scripts 01 to 13 in order.

Usage: python code/run_all.py (after pip install -r requirements.txt).
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

for script in ["01_build_panel.py", "02_core_models.py", "03_lisa.py",
               "04_validation.py", "05_figures.py", "06_robustness_real_terms.py",
               "07_review_checks.py", "08_figures_v2.py", "09_affinity_check.py",
               "10_population_controls.py", "11_extended_checks.py",
               "12_residual_map.py", "13_figures_standard.py"]:
    print(f"\n===== {script} =====", flush=True)
    subprocess.run([sys.executable, str(HERE / script)], check=True)
print("\nPipeline complete.")
