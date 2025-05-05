#!/usr/bin/env python
"""
run_experiments.py
Batch‑run train_lp.py for different (n,m) combos, collect benchmarks, plot.
"""

import subprocess, json, pathlib, time, csv, sys, shutil
import pandas as pd, matplotlib.pyplot as plt
from itertools import product

# ─── grid & settings ───
N_vals = [30, 50, 70]      # variables
M_vals = [15, 25, 35]      # constraints
samples = 20000
epochs  = 60
lam_feas, lam_gap = 3.0, 0.02
hidden_map = {30:"128,64", 50:"256,128,64", 70:"512,256,128"}

exp_root = pathlib.Path("experiments"); exp_root.mkdir(exist_ok=True)
rows = []

for n, m in product(N_vals, M_vals):
    tag = f"n{n}_m{m}"
    run_dir = exp_root / tag; run_dir.mkdir(exist_ok=True)
    print(f"\n=== {tag} ===")
    cmd = [
        sys.executable, "train_lp.py",
        "--n", str(n), "--m", str(m),
        "--hidden", hidden_map[n],
        "--samples", str(samples),
        "--epochs",  str(epochs),
        "--lam_feas", str(lam_feas),
        "--lam_gap",  str(lam_gap)
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (run_dir/"run.log").write_text(proc.stdout + proc.stderr)

    # copy benchmark
    bjson = pathlib.Path("results_lp")/"benchmark.json"
    if not bjson.exists():
        print("! benchmark missing for", tag)
        continue
    bench = json.load(open(bjson))
    bench.update({"tag": tag, "n": n, "m": m, "time_s": time.time()-t0})
    rows.append(bench)
    shutil.copy(bjson, run_dir/"benchmark.json")
    print(f"✓ done {tag}")

# ─── save summary CSV ───
sum_csv = exp_root/"summary.csv"
with open(sum_csv, "w", newline="") as f:
    csv.DictWriter(f, rows[0].keys()).writeheader()
    csv.DictWriter(f, rows[0].keys()).writerows(rows)
print("Summary saved to", sum_csv)

# ─── plot ───
df = pd.DataFrame(rows).set_index("tag")
df[["feas_violation","opt_gap"]].plot(kind="bar", figsize=(9,4))
plt.ylabel("value"); plt.title("Accuracy across configs"); plt.tight_layout(); plt.show()

