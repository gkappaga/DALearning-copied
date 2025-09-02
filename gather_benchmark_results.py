import subprocess
import torch
import os
import pandas as pd
from utils import redirect_output
# Parameters to sweep
methods = ['EnKF', 'ESRF', 'LETKF']
N_values = [5, 10, 15, 20, 40, 60, 100]

# Constants (edit as needed)
dataset = 'lorenz96'
output_csv = 'benchmarks_lorenz96_varyingnoise.csv'
base_save_dir = 'save/benchmark/'

# Initialize results list
results = []

for method in methods:
    access_to_noise_values = [None] if method != 'EnKF' else [True, False]

    for access_to_noise in access_to_noise_values:
        for N in N_values:
            suffix = ""
            extra_args = []

            if method == 'EnKF':
                suffix = f"_access_{access_to_noise}"
                if access_to_noise:
                    extra_args.append("--access_to_noise")

            folder_name = os.path.join("save", f"benchmark_{dataset}_varyingnoise_{method}{suffix}")
            os.makedirs(folder_name, exist_ok=True)

            with redirect_output(folder_name, filename="test_output.txt", enable_redirect=True):
                print('-------------------------------NEW BENCHMARK RUN-------------------------------------')
                print(f"Running: method={method}, N={N}, access_to_noise={access_to_noise}")

                # Run benchmark
                subprocess.run([
                    'python', 'evaluate_benchmark.py',
                    '--dataset', dataset,
                    '--v', method,
                    '--N', str(N),
                    '--cp_load_path', 'no',
                    '--random_noise',
                    *extra_args  # append EnKF-specific args
                ])
                # Load output
                model_folder = os.path.join('save/benchmark_models/', f"benchmark_{dataset}_varyingnoise_{method}{suffix}_{N}")
                record_path = os.path.join(model_folder, f"output_records_{N}.pt")

                if not os.path.exists(record_path):
                    print(f"❌ Missing file: {record_path}")
                    continue

                tensor_dict = torch.load(record_path)
                nn_stats = tensor_dict['nn']

                row = {
                    'method': method,
                    'N': N,
                    'access_to_noise': access_to_noise if method == 'EnKF' else None,
                    'mean_rmse': nn_stats['mean_rmse'],
                    'std_rmse': nn_stats['std_rmse'],
                    'mean_rrmse': nn_stats['mean_rrmse'],
                    'std_rrmse': nn_stats['std_rrmse'],
                    'mean_rmv': nn_stats['mean_rmv'],
                    'std_rmv': nn_stats['std_rmv'],
                    'valid_percent': nn_stats['valid_percent'],
                    'nan_exist': nn_stats['valid_percent'] < 1.0
                }

                results.append(row)

# Save CSV
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)