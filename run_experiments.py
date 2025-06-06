#!/usr/bin/env python3
"""
Run the Loop ABM for several num_test_algo values in one shot.
Figures for each batch appear in ./output/<run_id>/.
"""

# ── Imports ─────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")              # no GUI

import matplotlib.pyplot as plt
import plot_utils
import time
import logging
from datetime import datetime
from pathlib import Path

from abm_utils import free_abm, loop_abm
from rankers   import RandomRanker, CTRanker, MFRanker, BPRanker

# ── Base config ─────────────────────────────────────────────────────────
N, T               = 500000, 100
initial_steps      = 10
num_test_random    = 0                         # fixed
fraction_SM_obs    = 0.5
fraction_SS_obs    = 1.0
quarantine_HH      = False
test_HH            = False
adoption_fraction  = 1.0
fp_rate            = 0.0
fn_rate            = 0.0
seed               = 1                         # kept constant
n_seed_infection   = 3                         # unused here but preserved

# ── Rankers ─────────────────────────────────────────────────────────────
rankers = {
    "None": None,
    "RG"  : RandomRanker(),
    "CT"  : CTRanker(),
    #"MF"  : MFRanker(),
    #"BP"  : BPRanker(),
}

# ── Logging (single file) ───────────────────────────────────────────────
log_dir = Path("results")
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=log_dir / "ranker_log.txt",
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    force=True,
)
root_logger = logging.getLogger("ranker")
root_logger.propagate = False

# ── Batch values (the only thing that changes) ──────────────────────────
test_algo_values = [5000]

# ── Main loop ───────────────────────────────────────────────────────────
for num_test_algo in test_algo_values:
    run_id   = f"algo{num_test_algo}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir  = Path("output") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== num_test_algo = {num_test_algo}  →  {run_id} ===")

    # Pre-build static grid layout
    plots = plot_utils.plot_style(N, T)
    fig, _ = plot_utils.plotgrid(rankers, plots, initial_steps, save_path=None)

    results = {}
    for name, rk in rankers.items():
        print(f"   • {name}")
        if rk is None:
            data = free_abm({}, logger=root_logger)
        else:
            data = loop_abm(
                params={},
                N=N, T=T,
                inference_algorithm=rk,
                seed=seed,
                logger=logging.getLogger(f"loop_abm.{name}"),
                data=None,
                callback=lambda _: None,            # disable live plotting
                initial_steps=initial_steps,
                num_test_random=num_test_random,
                num_test_algo=num_test_algo,        # ← key change
                fraction_SM_obs=fraction_SM_obs,
                fraction_SS_obs=fraction_SS_obs,
                quarantine_HH=quarantine_HH,
                test_HH=test_HH,
                adoption_fraction=adoption_fraction,
                fp_rate=fp_rate,
                fn_rate=fn_rate,
                name_file_res=f"{name}_{run_id}",
            )
        results[name] = data

    # Populate grid and save
    for pt in plots:
        for name in rankers:
            if pt in results[name]:
                plots[pt][name][0].set_data(range(len(results[name][pt])),
                                            results[name][pt])
    fig.legend(labels=rankers.keys(), loc='lower right')
    fig.savefig(out_dir / f"grid_{run_id}.png", dpi=300)
    plt.close(fig)

    # Quick semilog infected plot
    plt.figure()
    to_plot = "I"
    palette = ["tab:red", "tab:grey", "tab:purple", "tab:green", "tab:orange"]
    for name, color in zip(rankers, palette):
        plt.plot(results[name][to_plot], label=name, color=color)
    plt.semilogy()
    plt.ylabel("Infected"); plt.xlabel("Days"); plt.legend(); plt.xlim(0, T)
    plt.tight_layout()
    plt.savefig(out_dir / f"infections_{run_id}.png", dpi=300)
    plt.close()

    # Light breather so the OS can flush I/O
    del results
    time.sleep(0.5)

print("\nAll batches finished. Check the ./output/ folders for figures.")
