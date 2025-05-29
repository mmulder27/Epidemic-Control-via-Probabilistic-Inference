import matplotlib
#matplotlib.use("Agg")  # Avoid GUI issues if running headless

import matplotlib.pyplot as plt
import plot_utils
import time
import logging
from abm_utils import free_abm, loop_abm
from rankers import RandomRanker, CTRanker
from pathlib import Path

# --- Configuration ---
N = 500000
T = 100
initial_steps = 10
num_test_random = 0
num_test_algo = 200
fraction_SM_obs = 0.5
fraction_SS_obs = 1.0
quarantine_HH = True
test_HH = True
adoption_fraction = 1.0
fp_rate = 0.0
fn_rate = 0.0
seed = 1
n_seed_infection = 50

prob_seed = 1/N
prob_sus = 0.55
pseed = prob_seed / (2 - prob_seed)
psus = prob_sus * (1 - pseed)
pautoinf = 1/N

# Ranker hyperparameters
tau = 5,
delta = 10,
mu = 1/30,
lamb = 0.014


# For loopy BP Ranker                           
maxit0 = 20,
maxit1 = 20,
tol = 1e-3,
memory_decay = 1e-5,
window_length = 21,
tau=7

# --- Rankers to test ---
rankers = {
    "None": None,
    "RG": RandomRanker(),
    "CT": CTRanker()
}

# --- Plotting setup (static) ---
plots = plot_utils.plot_style(N, T)
save_path_fig = f"./output/plot_run_N_{N}_T_{T}_seed_{seed}.png"
fig, callback = plot_utils.plotgrid(rankers, plots, initial_steps, save_path=None)
time.sleep(0.5)


# --- Logging setup  ---
log_path = Path("results") / "ranker_log.txt"
log_path.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(log_path),
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)

# Then pass the logger into free_abm
logger = logging.getLogger("ranker")

# --- Run simulations ---
ress = {}
for s in rankers:
    print(f"Running simulation for: {s}")
    if rankers[s] is None:
        data = free_abm({}, logger=logger)
    else: 
        data = loop_abm(
            params={},
            inference_algorithm=rankers[s],
            seed=seed,
            logger=logging.getLogger(f"loop_abm.{s}"),
            data = None,
            callback=lambda _: None,  # disable live plotting
            initial_steps=initial_steps,
            num_test_random=num_test_random,
            num_test_algo=num_test_algo,
            fraction_SM_obs=fraction_SM_obs,
            fraction_SS_obs=fraction_SS_obs,
            quarantine_HH=quarantine_HH,
            test_HH=test_HH,
            adoption_fraction=adoption_fraction,
            fp_rate=fp_rate,
            fn_rate=fn_rate,
            name_file_res=s + f"_N_{N}_T_{T}_seed_{seed}"
    )
    ress[s] = data

# --- Final plots: populate lines and save ---
for pt in plots:
    for i, r in enumerate(rankers):
        if pt in ress[r]:
            plots[pt][r][0].set_data(
                range(len(ress[r][pt])),  # x-axis
                ress[r][pt]               # y-axis
            )

fig.legend(labels=rankers.keys(), loc='lower right')
Path(save_path_fig).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(save_path_fig)
print(f"Saved static plot to: {save_path_fig}")

# --- Final plot: Number infected over time ---
plt.figure()
to_plot = "I"
for s in ress.keys():
    plt.plot(ress[s][to_plot], label=s)

plt.semilogy()
plt.ylabel("Infected")
plt.xlabel("Days")
plt.legend()
plt.xlim(0, 100)

# Show the plot (GUI) or save to file
plt.tight_layout()
plt.savefig("./output/mitigation_by_ranking_method.png")
plt.show()
plt.close('all')
