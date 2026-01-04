# smf_vs_enkf_vsN.py
from typing import Tuple, Dict, List
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

# your project imports
from config.cli import get_parameters
from utils import get_dataloader, partial_obs_operator
from train_test_utils_v2 import test_ClassicFilter_v2


# ------------------------------------------------------------
# Small helpers for safe/resumable caching (unchanged)
# ------------------------------------------------------------
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _safe_save(obj, path: str):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)  # atomic on POSIX

def _load_cache(path: str):
    if not os.path.exists(path):
        return {}
    try:
        data = torch.load(path)
        if not isinstance(data, dict):
            print(f"[warn] Cache at {path} is not a dict. Ignoring.")
            return {}
        return data
    except Exception as e:
        print(f"[warn] Failed to load cache {path}: {e}. Ignoring.")
        return {}


# ------------------------------------------------------------
# Optional: your helper that reads DAPPER benchmarks (unchanged)
# ------------------------------------------------------------
def get_benchmarks(args):
    """
    Reads save/benchmark/benchmarks_{args.dataset}.csv and returns
    best row (loc_radius, infl, rmse, rrmse_mean) for sigma_y == 1.
    """
    file_path = f'save/benchmark/benchmarks_{args.dataset}.csv'
    if not os.path.exists(file_path):
        return None, None  # signal no benchmark found

    df = pd.read_csv(file_path, usecols=['method', 'N', 'sigma_y', 'best_loc_rad','best_infl','rmse', 'rrmse_mean'])

    method = args.v
    if method == 'EnKF':
        method = 'EnKF_PertObs'
    elif method == 'iEnKS-PertObs':
        method = 'iEnKF_PertObs'
    elif method == 'ESRF':
        method = 'LETKF'
    elif method == 'LETKF':
        method = 'LETKF'

    method_data = df[(df['method'] == method) & (df['N'] == args.N)]
    sigma_y_1 = method_data[method_data['sigma_y'] == 1][['best_loc_rad','best_infl','rmse', 'rrmse_mean']]
    if len(sigma_y_1) == 0:
        return None, None

    arr = sigma_y_1.to_numpy()
    loc_radius, infl, rmse_dapper, rrmse_dapper = arr[0,0], arr[0,1], arr[0,2], arr[0,3]
    # If loc radius is nan in file, default to 4 like your helper
    if isinstance(loc_radius, float) and np.isnan(loc_radius):
        loc_radius = 4
    return float(loc_radius), float(infl)


# ------------------------------------------------------------
# Single run helper (now sources dataset/sigma_y/seed from get_parameters)
# ------------------------------------------------------------
@torch.no_grad()
def run_once(
    base_args,
    N: int,
    method: str,
    *,
    p_rbf: int = 0,
    rho: float = 0.0,
    dataset: str = "lorenz63",
    seed: int = 42,
    test_traj_num: int = 64*8,
    test_batch_size: int = 64,
    use_dapper_for_enkf: bool = True,
) -> tuple[float, float]:
    # fresh parse every run
    args = BASE_ARGS

    # --- pull key values from base_args (get_parameters) with hard-coded fallbacks ---
    args.dataset = getattr(BASE_ARGS, "dataset", dataset)
    args.sigma_y = getattr(BASE_ARGS, "sigma_y", 2)       # your previous hard-coded 2 preserved
    args.seed     = getattr(BASE_ARGS, "seed", seed)

    # common config
    args.test_only = True
    args.v = method
    args.N = N
    args.cp_load_path = 'no'
    # args.random_noise = False
    args.access_to_noise = True
    args.test_traj_num = getattr(BASE_ARGS, "test_traj_num", test_traj_num)
    args.test_batch_size = getattr(BASE_ARGS, "test_batch_size", test_batch_size)

    # SMF knobs (match your stochastic_map_filter_analysis signature)
    if method == 'SMF':
        args.smf_rho = float(rho)
        args.smf_rbf_p = int(p_rbf)

    if args.seed is not None and args.seed != "None":
        torch.manual_seed(int(args.seed))

    # dataloader + H
    test_loader = get_dataloader(args, test_only=True)
    H_info = partial_obs_operator(args.ori_dim, args.obs_inds, args.device)

    # EnKF explicit knobs (simple defaults; you can still swap to CSV helper later)
    infl = 1.05
    loc_radius = None

    # run and summarize
    print(f"Running {method} dataset={args.dataset} N={N}, p_rbf={p_rbf}, rho={rho}, "
          f"loc_radius={loc_radius}, infl={infl}")
    try:
        rrmse_tensor = test_ClassicFilter_v2(
            test_loader,
            args,
            plot=False,
            H_info=H_info,
            plot_figures=False,
            save_pdf=False,
            infl=infl,
            loc_radius=loc_radius
        )
        mean_rmse = rrmse_tensor['mean_rmse']
        std_rmse = rrmse_tensor['std_rmse']
    except Exception as e:
        print(f"Error running {method} with N={N}, p_rbf={p_rbf}, rho={rho}: {e}")
        mean_rmse = float('nan')
        std_rmse = float('nan')
    return mean_rmse, std_rmse


# ------------------------------------------------------------
# Grid over N and rho (for SMF) — unchanged except base_args use
# ------------------------------------------------------------
def sweep_methods_vs_N(
    dataset: str = "lorenz63",
    Ns: List[int] = [10, 20, 40, 60, 100, 200, 400],
    rho_grid: List[float] = [round(0.05*i, 2) for i in range(0, 11)],  # 0.00..0.50
    p_rbfs: List[int] = [0, 1, 2],
    seed: int = 42
) -> Dict[str, Dict[int, Dict]]:
    """
    Returns nested dict:
    results[method][N] = {
        'best_mean_rrmse': float,
        'std_rrmse': float,
        'best_rho': float or None
    }
    """
    # base_args = get_parameters()
    # base_args.dataset = dataset

    _ensure_dir("smf_results")
    cache_path = f"smf_results/smf_vsN_cache_{dataset}.pt"
    results = _load_cache(cache_path)
    if results:
        print(f"[resume] Loaded existing cache with methods: {list(results.keys())}")

    # --- EnKF curve ---
    enkf_key = "EnKF"
    results.setdefault(enkf_key, {})
    for N in Ns:
        if N in results[enkf_key]:
            print(f"[skip] EnKF N={N} already in cache.")
            continue
        mean_rrmse, std_rrmse = run_once(
            BASE_ARGS, N, "EnKF",
            dataset=dataset, seed=seed
        )
        results[enkf_key][N] = {
            'best_mean_rrmse': float(mean_rrmse),
            'std_rrmse': float(std_rrmse),
            'best_rho': None
        }
        _safe_save(results, cache_path)
        print(f"[EnKF] N={N}: mean RRMSE={mean_rrmse:.3f} (std={std_rrmse:.3f})")

    # --- SMF variants (p_rbf = 0,1,2) ---
    for p in p_rbfs:
        method_key = f"SMF_p{p}"
        results.setdefault(method_key, {})
        for N in Ns:
            if N in results[method_key]:
                print(f"[skip] {method_key} N={N} already in cache.")
                continue
            best = (np.inf, 0.0)  # (best_mean, std)
            best_rho = None
            for rho in rho_grid:
                mean_rrmse, std_rrmse = run_once(
                    BASE_ARGS, N, "SMF",
                    p_rbf=p, rho=rho,
                    dataset=dataset, seed=seed
                )
                if mean_rrmse < best[0]:
                    best = (mean_rrmse, std_rrmse)
                    best_rho = rho
                print(f"[SMF p={p}] N={N}, rho={rho:.2f} -> mean RRMSE={mean_rrmse:.3f} (std={std_rrmse:.3f})")
            results[method_key][N] = {
                'best_mean_rrmse': float(best[0]),
                'std_rrmse': float(best[1]),
                'best_rho': float(best_rho)
            }
            _safe_save(results, cache_path)
            print(f"[SMF p={p}] N={N}: best rho={best_rho:.2f}, mean RRMSE={best[0]:.3f} (std={best[1]:.3f})")

    _safe_save(results, cache_path)
    return results


# ------------------------------------------------------------
# Plotting (unchanged)
# ------------------------------------------------------------
def plot_rrmse_vs_N(results: Dict, dataset: str):
    out_dir = f"smf_results/plots/{dataset}"
    _ensure_dir(out_dir)

    methods = ["EnKF", "SMF_p0", "SMF_p1", "SMF_p2"]
    labels = {
        "EnKF": "EnKF (bench infl/loc)",
        "SMF_p0": "SMF linear (p=0, best ρ)",
        "SMF_p1": "SMF linear + 1 RBF (best ρ)",
        "SMF_p2": "SMF linear + 2 RBF (best ρ)",
    }
    colors = {
        "EnKF": "tab:red",
        "SMF_p0": "tab:blue",
        "SMF_p1": "tab:green",
        "SMF_p2": "tab:purple",
    }

    plt.figure(figsize=(8, 5))
    for m in methods:
        if m not in results or not results[m]:
            continue
        Ns = sorted(results[m].keys())
        means = [results[m][N]['best_mean_rrmse'] for N in Ns]
        stds  = [results[m][N]['std_rrmse'] for N in Ns]
        plt.errorbar(Ns, means, yerr=stds, marker='o', linestyle='-', label=labels[m], color=colors[m])

    plt.xscale('log', base=10)
    plt.xticks([10, 20, 40, 60, 100, 200, 400], [10, 20, 40, 60, 100, 200, 400])
    plt.xlabel("Ensemble size N (log scale)")
    plt.ylabel("Mean RRMSE")
    plt.title(f"EnKF vs SMF (linear/RBF), dataset={dataset}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, "rrmse_vs_N.png")
    plt.savefig(out_path, dpi=160)
    print(f"Saved plot: {out_path}")
    plt.close()

    # (Optional) print the best rho table for SMF methods
    for m in ["SMF_p0", "SMF_p1", "SMF_p2"]:
        if m in results and results[m]:
            print(f"\nBest ρ per N for {m}:")
            Ns = sorted(results[m].keys())
            for N in Ns:
                print(f"  N={N:3d}: rho={results[m][N]['best_rho']:.2f}, meanRRMSE={results[m][N]['best_mean_rrmse']:.3f}")


# ------------------------------------------------------------
# Main — get_parameters() decides dataset/sigma_y/seed; Ns/rho hard-coded
# ------------------------------------------------------------
if __name__ == "__main__":
    BASE_ARGS = get_parameters()
    # dataset from get_parameters(); defaults to 'lorenz63' if missing
    DATASET = getattr(BASE_ARGS, "dataset", "lorenz63")
    # BASE_ARGS = get_parameters()              # picks up --dataset, --sigma_y, --seed, etc. from CLI
    BASE_ARGS.test_only = True
    # BASE_ARGS.random_noise = False
    # BASE_ARGS.access_to_noise = True
    BASE_ARGS.test_traj_num = 64 * 16
    BASE_ARGS.test_batch_size = 64
    BASE_ARGS.cp_load_path = 'no'
    BASE_ARGS.sigma_y = 2
    BASE_ARGS.dataset = 'lorenz63'
    BASE_ARGS.access_to_H = True
    BASE_ARGS.access_to_noise = True
    BASE_ARGS.random_noise = False
    # keep dataset and sigma_y from CLI; hard-code a seed unless overridden by CLI
    if getattr(BASE_ARGS, 'seed', None) in (None, "None"):
        BASE_ARGS.seed = 42

    # hard-coded sweep ranges (per your request)
    Ns = [10, 20, 40, 60, 100, 200, 400]
    rho_grid = [round(0.1*i, 2) for i in range(0, 6)]  # 0.00..0.50
    results = sweep_methods_vs_N(
        dataset=DATASET,
        Ns=Ns,
        rho_grid=rho_grid,
        p_rbfs=[0, 1, 2],
        seed=getattr(BASE_ARGS, "seed", 42)
    )
    
    plot_rrmse_vs_N(results, dataset=DATASET)
