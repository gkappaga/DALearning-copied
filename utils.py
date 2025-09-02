import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import sys
import datetime
from contextlib import contextmanager

from torch.utils.data import Dataset, DataLoader
import torch
from torch.optim import AdamW, SGD
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from dapper.mods.KS import Model as DapperKS

@contextmanager
# def redirect_output(save_folder, filename="output.txt", enable_redirect=True):
#     """
#     Context manager to redirect all stdout and stderr output to a specified file.
    
#     Args:
#         save_folder (str): The folder where the output file should be saved.
#         filename (str): The name of the output file (default: "output.txt").
#     """
#     if not enable_redirect:
#         # If redirection is not enabled, just yield and do nothing else.
#         try:
#             yield
#         # finally:
#         #     # No cleanup needed if redirection was not active.
#         #     pass
#         return # Exit the function early

#     # Ensure the directory exists if redirection is enabled
#     os.makedirs(save_folder, exist_ok=True)
#     output_file = os.path.join(save_folder, filename)

#     # Backup original stdout and stderr
#     original_stdout = sys.stdout
#     original_stderr = sys.stderr

#     file_exists = os.path.exists(output_file)

#     try:
#         with open(output_file, "a") as f:  # Open in append mode
#             if file_exists:
#                 # Add a timestamp for new entries if the file already exists
#                 f.write(f"\n--- New Session: [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ---\n")
#             else:
#                 # Optional: Add a header if the file is new
#                 f.write(f"--- Log started: [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ---\n")

#             sys.stdout = f
#             sys.stderr = f
#             yield  # Execute the code block within the context
#     finally:
#         # Restore stdout and stderr only if they were redirected
#         if enable_redirect:
#             sys.stdout = original_stdout
#             sys.stderr = original_stderr
#             # Notify in the console that output was appended
#             # This message will go to the original stdout (e.g., the console)
#             print(f"Output appended to {output_file}")
    
#     with open(output_file, "a") as f:  # Open in append mode
#         if file_exists:
#             f.write(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n")
        
#         sys.stdout = f
#         sys.stderr = f
#         try:
#             yield  # Execute the code block within the context
#         finally:
#             # Restore stdout and stderr after execution
#             sys.stdout = original_stdout
#             sys.stderr = original_stderr
#             print(f"Output appended to {output_file}")  # Notify in the console

def redirect_output(save_folder, filename="output.txt", enable_redirect=True):
    if not enable_redirect:
        yield
        return

    os.makedirs(save_folder, exist_ok=True)
    output_file = os.path.join(save_folder, filename)
    file_exists = os.path.exists(output_file)

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        with open(output_file, "a") as f:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if file_exists:
                f.write(f"\n--- New Session: [{timestamp}] ---\n")
            else:
                f.write(f"--- Log started: [{timestamp}] ---\n")

            sys.stdout = f
            sys.stderr = f

            yield  # ✅ only one yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        print(f"Output appended to {output_file}")

def check_nan_in_model(model):
    for param in model.parameters():
        if torch.isnan(param).any():
            # print("NaN detected in model parameters")
            return True
    # print("No NaN in model parameters")
    return False

def get_mean_std(data_tensor):
    return torch.mean(data_tensor).item(), torch.std(data_tensor).item()

def plot_particle_trajectories_with_histograms(particles: torch.Tensor, 
                                                true_traj: torch.Tensor, 
                                                dim_indices: list,
                                                num_time_steps: int = 100,
                                                mode: str = 'width',
                                                save_fig: bool = False,
                                                save_name: str = 'example_fig',
                                                hist_step: int = 1):
    if particles.is_cuda:
        particles = particles.detach().cpu()
    if true_traj.is_cuda:
        true_traj = true_traj.detach().cpu()
    J, N, d = particles.shape
    max_time_step = np.minimum(J, num_time_steps)
    time_steps = np.arange(max_time_step)
    step_width = 1.0

    for dim in dim_indices:
        if dim >= true_traj.shape[-1]:
            continue
        plt.figure(figsize=(12, 6))
        
        plt.plot(time_steps, true_traj[:max_time_step, dim], label='True Trajectory', color='blue', linewidth=1)
        particle_mean = particles[:max_time_step, :, dim].mean(dim=1)
        plt.plot(time_steps, particle_mean, label='Particle Mean', color='red', linestyle='--', linewidth=1)

        if mode == 'color':
            all_bin_masses = []
            for t in range(0, max_time_step, hist_step):
                data = particles[t, :, dim].numpy()
                hist, bins = np.histogram(data, bins=15, density=True)
                for h, b_start, b_end in zip(hist, bins[:-1], bins[1:]):
                    bin_mass = h * (b_end - b_start)
                    all_bin_masses.append(bin_mass)
            global_max_mass = max(all_bin_masses)

        for t in range(0, max_time_step, hist_step):
            data = particles[t, :, dim].numpy()
            hist, bins = np.histogram(data, bins=15, density=True)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])

            if mode == 'width':
                hist_norm = hist / hist.max() * 0.8 if hist.max() > 0 else hist
                plt.fill_betweenx(bin_centers, t - hist_norm, t + hist_norm,
                                  facecolor='orange', edgecolor='none', alpha=0.5)
            elif mode == 'color':
                cmap = cm.get_cmap('Oranges')
                for h, b_start, b_end in zip(hist, bins[:-1], bins[1:]):
                    bin_mass = h * (b_end - b_start)
                    norm_val = bin_mass / global_max_mass
                    color = cmap(0.2 + 0.5 * norm_val)
                    plt.fill_between([t - step_width / 2, t + step_width / 2],
                                    b_start, b_end,
                                    color=color, linewidth=0)

        plt.xlabel('Time Step')
        plt.ylabel(f'States')
        plt.title(f'Dimension Index {dim}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if save_fig:
            plt.savefig(f"{save_name}_dim_{dim}_{mode}.png", dpi=150)
            plt.close()
        else:
            plt.show()

def plot_results_3d(preds, gts, start_ind, end_ind):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(preds[0, start_ind:end_ind], preds[1, start_ind:end_ind], preds[2, start_ind:end_ind], c='r', label='Predictions')
    ax.plot(gts[0, start_ind:end_ind], gts[1, start_ind:end_ind], gts[2, start_ind:end_ind], c='b', label='Ground-Truth')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.legend()

    plt.show()

def plot_results_2d(preds, gts, start_ind, end_ind, plot_inds=None, save_path='vis_results.png'):
    if plot_inds is None:
        plot_inds = [0]
        
    num_plots = len(plot_inds)
    fig, axes = plt.subplots(num_plots, 1, figsize=(14, 4 * num_plots))  

    x_inds = torch.arange(start_ind, end_ind)

    for i, plot_ind in enumerate(plot_inds):
        ax = axes[i] if num_plots > 1 else axes  
        ax.plot(x_inds, preds[plot_ind, start_ind:end_ind], c='r', label='Predictions')
        ax.plot(x_inds, gts[plot_ind, start_ind:end_ind], c='b', label='Ground-Truth')

        ax.set_title(f'Dimension Index: {plot_ind}')

        if i < num_plots - 1:
            ax.tick_params(labelbottom=False)

        ax.legend()

    plt.tight_layout() 
    plt.savefig(save_path)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

def rk4(func, u, t, dt):
    # Compute intermediary values for k
    k1 = func(dt, u)
    k2 = func(t + dt/2, u + dt/2*k1)
    k3 = func(t + dt/2, u + dt/2*k2)
    k4 = func(t + dt, u + dt*k3)
    # Compute updated values for u and t
    u_n = u + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    return u_n

class VL20(nn.Module):
    """Modeled after dapper implementation"""

    def __init__(self, nX=36, F=10, G=10, alpha=1, gamma=1):
        super(VL20, self).__init__()
        self.fe = 0
        self.nX = nX
        self.F = F
        self.G = G
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, t, x):
        self.fe += 1
        out = torch.zeros_like(x)
        # print( torch.split(x, self.nX, -1))
        X, theta = x[:, 0, :], x[:, 1, :]

        # Velocities
        out[:, 0, :] = (torch.roll(X, 1, -1) - torch.roll(X, -2, -1)) * torch.roll(X, -1, -1)
        out[:, 0, :] -= self.gamma * X
        out[:, 0, :] += self.F - self.alpha * theta
        # Temperatures
        out[:, 1, :] = torch.roll(X, 1, -1) * torch.roll(theta, 2, -1) - \
                       torch.roll(X, -1, -1) * torch.roll(theta, -2, -1)
        out[:, 1, :] -= self.gamma * theta
        out[:, 1, :] += self.alpha * X + self.G
        return out


class L63(nn.Module):
    def __init__(self):
        super(L63, self).__init__()
        self.fe = 0

    @staticmethod
    def forward(t, x, sig=10.0, rho=28.0, beta=8.0 / 3):
        rvals = torch.zeros_like(x)
        rvals[:, 0] = sig * (x[:, 1] - x[:, 0])
        rvals[:, 1] = x[:, 0] * (rho - x[:, 2]) - x[:, 1]
        rvals[:, 2] = x[:, 0] * x[:, 1] - beta * x[:, 2]
        return rvals
    
class L96(nn.Module):
    def __init__(self):
        super(L96, self).__init__()

    def forward(t, x, F=8.):
        x_m2 = torch.roll(x, -2, -1)
        x_m1 = torch.roll(x, -1, -1)
        x_p1 = torch.roll(x, 1, -1)
        return (x_p1 - x_m2) * x_m1 - x + F * torch.ones_like(x)
    


##### old generate data, not sure if it is correct
def gen_data_old(dataset, t, steps_test, steps_valid, step=None, check_disk=True, steps_burn=1000, true_v0=None):
    """ Generates training and test data for given model. Defaults used in experiments
    are hardcoded making this somewhat longer than it needs to be.

    args
    -----
    dataset : string
        Name of the dynamics to use for data gen
    steps_test : int
        Number of steps to use for test set. Steps_test + steps_valid Must be smaller than len(t)
    steps_valid : int
        Number of steps to use for valid set. Steps_test + steps_valid Must be smaller than len(t)
    step : int
        Step size taken by integrator. If no step size provided, uses t[1] - t[0]
    check_disk : bool
        Indicates whether to check if saved first.

    returns
    -------
    train : torch.Tensor
        Training sequence
    valid : torch.Tensor
        Validation sequence
    test : torch.Tensor
        Test sequence
    """
    makedirs('data/%s' % dataset)
    if step is None:
        step = t[1] - t[0]
    rstep = t[1] - t[0]
    if dataset == 'lorenz96':
        with torch.no_grad():
            if check_disk and os.path.exists('data/%s/true_v_%.3fstep.npy' % (dataset, rstep)):
                true_v = torch.Tensor(np.load('data/%s/true_v_%.3fstep.npy' % (dataset, rstep)))
                true_v0_test = true_v[-1]
                true_v = true_v[:-1]
                true_v_test = odeint(L96(), true_v0_test, t[:steps_burn + steps_valid + steps_test],
                                     method='rk4', options={'step_size': step})
            else:
                if true_v0 is None or true_v0.shape != (1, 40):
                    true_v0 = torch.randn(1, 40) + 5
                true_v = odeint(L96(), true_v0, t, method='rk4', options={'step_size': step})
                if check_disk:
                    np.save('data/%s/true_v_%.3fstep.npy' % (dataset, rstep), true_v)
                true_v0_test = true_v[-1]
                true_v = true_v[:-1]
                true_v_test = odeint(L96(), true_v0_test, t[:steps_burn + steps_test + steps_valid],
                                     method='rk4', options={'step_size': step})
    elif dataset == 'vl20':
        with torch.no_grad():
            if check_disk and os.path.exists('data/%s/true_v_%.3fstep.npy' % (dataset, rstep)):
                true_v = torch.Tensor(np.load('data/%s/true_v_%.3fstep.npy' % (dataset, rstep)))
                true_v0_test = true_v[-1]
                true_v = true_v[:-1]
                true_v_test = odeint(VL20(), true_v0_test, t[:steps_burn + steps_valid + steps_test],
                                     method='rk4', options={'step_size': step})
            else:
                if true_v0 is None or true_v0.shape != (1, 2, 36):
                    true_v0 = torch.randn(1, 2, 36) + 5
                true_v = odeint(VL20(), true_v0, t, method='rk4', options={'step_size': step})
                if check_disk:
                    np.save('data/%s/true_v_%.3fstep.npy' % (dataset, rstep), true_v)
                true_v0_test = true_v[-1]
                true_v = true_v[:-1]
                true_v_test = odeint(VL20(), true_v0_test, t[:steps_burn + steps_test + steps_valid],
                                     method='rk4', options={'step_size': step})
    # Two level Lorenz - just used the DAPPER implementation here since we didn't need the ability to
    # differentiably forward integrate
    elif dataset == 'lorenzuv':
        with torch.no_grad():
            if check_disk and os.path.exists('data/%s/true_v_%.3fstep.npy' % (dataset, rstep)):
                true_v = torch.Tensor(np.load('data/%s/true_v_%.3fstep.npy' % (dataset, rstep)))

                true_v_test = true_v[-(steps_test + steps_valid):].unsqueeze(1)
                true_v = true_v[:-(steps_test + steps_valid)].unsqueeze(1)
            else:
                raise NotImplementedError
    # Not used in paper due to dimensionality making full rank EnKF too simple to compute
    # Also, there might be an error in this somewhere since it was never debugged
    elif dataset == 'lorenz63':
        with torch.no_grad():
            if check_disk and os.path.exists('data/%s/true_v_%.3fstep.npy' % (dataset, rstep)):
                true_v = torch.Tensor(np.load('data/%s/true_v_%.3fstep.npy' % (dataset, rstep)))
                true_v0_test = true_v[-1]
                true_v = true_v[:-1]
                true_v_test = odeint(L63(), true_v0_test, t[:steps_burn + steps_valid + steps_test],
                                     method='rk4', options={'step_size': step})
            else:
                if true_v0 is None or true_v0.shape != (1, 3):
                    true_v0 = torch.Tensor([[1.509, -1.531, 25.46]])
                true_v = odeint(L63(), true_v0, t, method='rk4', options={'step_size': step})
                if check_disk:
                    np.save('data/%s/true_v_%.3fstep.npy' % (dataset, rstep), true_v)
                true_v0_test = true_v[-1]
                true_v = true_v[:-1]
                true_v_test = odeint(L63(), true_v0_test, t[:steps_burn + steps_test + steps_valid],
                                     method='rk4', options={'step_size': step})
    elif dataset == 'ks':
        with torch.no_grad():
            if check_disk and os.path.exists('data/%s/true_v_%.3fstep.npy' % (dataset, rstep)):
                true_v = torch.Tensor(np.load('data/%s/true_v_%.3fstep.npy' % (dataset, rstep)))
                true_v0_test = true_v[-1].unsqueeze(0)
                # print('EEEEE', true_v0_test.shape)
                true_v = true_v[:-1].unsqueeze(1)
                true_v_test = custom_int(true_v0_test, etd_rk4_wrapper(), steps_test + steps_valid).unsqueeze(1)
            else:
                # Generate initial point from Dapper
                grid = 32 * np.pi * torch.linspace(0, 1, 128 + 1)[1:]
                x0_Kassam = torch.cos(grid / 16) * (1 + torch.sin(grid / 16))
                x0 = x0_Kassam.clone().unsqueeze(0)
                for _ in range(150):
                    x0 = custom_int(x0, etd_rk4_wrapper(), 150)[-1:]
                true_v0 = custom_int(x0, etd_rk4_wrapper(), 10 ** 3)[-1:]
                true_v = custom_int(true_v0, etd_rk4_wrapper(), t.shape[0])
                if check_disk:
                    np.save('data/%s/true_v_%.3fstep.npy' % (dataset, rstep), true_v)
                true_v0_test = true_v[-1].unsqueeze(0)
                true_v = true_v[:-1].unsqueeze(1)
                true_v_test = custom_int(true_v0_test, etd_rk4_wrapper(), steps_test + steps_valid).unsqueeze(1)
    else:
        raise ValueError('Dataset not implemented')
    return true_v, true_v_test[steps_burn:steps_valid + steps_burn], true_v_test[steps_burn + steps_valid:]


def gen_data(dataset, t, steps_test, steps_valid, v0=None, sigma_v=0,
             check_disk=True, steps_burn=1000, dt_iter=2, prefix="", test_only=False):
    """
    Generate training, validation, and test data for a given model.

    Parameters
    ----------
    dataset : str
        The name of the dataset, determines which model to use.
    t : array-like
        Time steps for integration.
    steps_test : int
        Number of steps in the test dataset.
    steps_valid : int
        Number of steps in the validation dataset.
    v0 : torch.Tensor, optional
        Initial state vector.
    sigma_v : float, default=0
        Standard deviation of Gaussian noise to add at each step.
    check_disk : bool, default=True
        If True, check if saved data exists on disk before generating.
    steps_burn : int, default=1000
        Number of initial steps to discard (burn-in).
    dt_iter : int, default=2
        Number of sub-iterations within each time step.
    prefix : str, default=""
        Prefix for saved file names.
    test_only : bool, default=False
        If True, generate only test data.

    Returns
    -------
    torch.Tensor
        Training sequence.
    torch.Tensor
        Validation sequence.
    torch.Tensor
        Test sequence.
    """

    dt = t[1] - t[0]  # Time step size for integration
    directory = f'data/{dataset}/'
    os.makedirs(directory, exist_ok=True)

    # Internal function for saving or loading data
    def __save_or_load_data(file_path, data=None):
        """Save data if provided, otherwise load data from the file path."""
        if data is not None:
            np.save(file_path, data.astype(np.float32))  # Save as float32
        else:
            return torch.tensor(np.load(file_path), dtype=torch.float32)  # Load as float32

    # Internal function to generate trajectory
    def __generate_trajectory(vf, steps, model, dt, dt_iter, sigma_v, burn_in=0):
        """Generate a trajectory for a given initial state, steps, model, and noise level."""
        trajectory = []
        for step in range(burn_in + steps):
            for _ in range(dt_iter):
                vf = rk4(model.forward, vf, step * dt, dt / dt_iter)
            vf = vf + sigma_v * torch.randn_like(vf, dtype=torch.float32, device=vf.device)  # Ensure float32
            trajectory.append(vf.unsqueeze(0))
        return torch.cat(trajectory).to(dtype=torch.float32)  # Ensure the output is float32

    # Define model and dimensions based on dataset type
    if dataset == "lorenz63":
        model, dim, default_v0 = L63, 3, torch.randn(1, 3, dtype=torch.float32)  # Ensure float32
    elif dataset == "lorenz96":
        model, dim, default_v0 = L96, 40, torch.randn(1, 40, dtype=torch.float32) + 5  # Ensure float32
    elif dataset == "ks":
        # Kuramoto-Sivashinsky model specifics
        model = etd_rk4_wrapper(device=None, dt=dt / dt_iter)
        # grid = 32 * np.pi * torch.linspace(0, 1, 128 + 1, dtype=torch.float32)[1:]  # Ensure float32
        # x0_Kassam = torch.cos(grid / 16) * (1 + torch.sin(grid / 16))
        # x0 = x0_Kassam.clone().unsqueeze(0)
        # # Single 150-step integration to stabilize x0
        # x0 = custom_int(x0, model, 150, dt, dt_iter)[-1:].to(dtype=torch.float32)  # Ensure float32
        # default_v0 = custom_int(x0, model, 10 ** 3, dt, dt_iter)[-1:].to(dtype=torch.float32)  # Ensure float32
        dapper_x0 = DapperKS(dt=dt.numpy() / dt_iter).x0
        default_v0 = torch.randn(1, 128, dtype=torch.float32) + dapper_x0
    else:
        raise ValueError('Dataset not implemented')

    v0 = v0 if v0 is not None else default_v0
    prefix_path = f'{directory}/{prefix}true_v_withnoise_{dt:.3f}step.npy'
    test_file_path = f'{directory}/test_{steps_valid}_{steps_test}_v_{dt:.3f}step.npy'
    
    # Training data generation
    with torch.no_grad():
        if not test_only:
            if check_disk and os.path.exists(prefix_path):
                v_traj = __save_or_load_data(prefix_path)
            else:
                if dataset == "ks":
                    # Custom integration for KS model with modified dt_iter
                    v_traj = custom_int(v0, model, len(t), dt, dt_iter).unsqueeze(1).to(dtype=torch.float32)  # Ensure float32
                else:
                    # General case for Lorenz models
                    v_traj = __generate_trajectory(v0, len(t), model, dt, dt_iter, sigma_v)
                if check_disk:
                    __save_or_load_data(prefix_path, v_traj.numpy())

        # Determine final state for testing
        vf = v_traj[-1].view(1, -1) if not test_only else v0

        # Testing data generation
        if check_disk and os.path.exists(test_file_path):
            v_traj_test = __save_or_load_data(test_file_path)
        else:
            if dataset == "ks":
                # Custom integration for KS model's testing data with modified dt_iter
                v_traj_test = custom_int(vf, model, steps_burn + steps_test + steps_valid, dt, dt_iter).unsqueeze(1).to(dtype=torch.float32)  # Ensure float32
            else:
                v_traj_test = __generate_trajectory(vf, steps_burn + steps_test + steps_valid, model, dt, dt_iter, sigma_v)
            if check_disk:
                __save_or_load_data(test_file_path, v_traj_test.numpy())

    if test_only:
        return v_traj_test[steps_burn:steps_valid + steps_burn], v_traj_test[steps_burn + steps_valid:]
    
    return v_traj, v_traj_test[steps_burn:steps_valid + steps_burn], v_traj_test[steps_burn + steps_valid:]





def etd_rk4_wrapper(device=None, dt=0.5, DL=32, Nx=128):
    """ Returns an ETD-RK4 integrator for the KS equation. Currently very specific, need
    to adjust this to fit into the same framework as the ODE integrators

    Directly ported from https://github.com/nansencenter/DAPPER/blob/master/dapper/mods/KS/core.py
    which is adapted from kursiv.m of Kassam and Trefethen, 2002, doi.org/10.1137/S1064827502410633.
    """
    if device is None:
        device = torch.device('cpu')
    kk = np.append(np.arange(0, Nx / 2), 0) * 2 / DL  # wave nums for rfft
    h = dt

    # Operators
    L = kk ** 2 - kk ** 4  # Linear operator for K-S eqn: F[ - u_xx - u_xxxx]

    # Precompute ETDRK4 scalar quantities
    E = torch.Tensor(np.exp(h * L)).unsqueeze(0).to(device)  # Integrating factor, eval at dt
    E2 = torch.Tensor(np.exp(h * L / 2)).unsqueeze(0).to(device)  # Integrating factor, eval at dt/2

    # Roots of unity are used to discretize a circular countour...
    nRoots = 16
    roots = np.exp(1j * np.pi * (0.5 + np.arange(nRoots)) / nRoots)
    # ... the associated integral then reduces to the mean,
    # g(CL).mean(axis=-1) ~= g(L), whose computation is more stable.
    CL = h * L[:, None] + roots  # Contour for (each element of) L
    # E * exact_integral of integrating factor:
    Q = torch.Tensor(h * ((np.exp(CL / 2) - 1) / CL).mean(axis=-1).real).unsqueeze(0).to(device)
    # RK4 coefficients (modified by Cox-Matthews):
    f1 = torch.Tensor(h * ((-4 - CL + np.exp(CL) * (4 - 3 * CL + CL ** 2)) / CL ** 3).mean(axis=-1).real).unsqueeze(
        0).to(device)
    f2 = torch.Tensor(h * ((2 + CL + np.exp(CL) * (-2 + CL)) / CL ** 3).mean(axis=-1).real).unsqueeze(0).to(device)
    f3 = torch.Tensor(h * ((-4 - 3 * CL - CL ** 2 + np.exp(CL) * (4 - CL)) / CL ** 3).mean(axis=-1).real).unsqueeze(
        0).to(device)

    D = 1j * torch.Tensor(kk).to(device)  # Differentiation to compute:  F[ u_x ]

    def NL(v, verb=False):
        return -.5 * D * torch.fft.rfft(torch.fft.irfft(v, dim=-1) ** 2, dim=-1)

    def inner(v, t, dt, verb=False):
        v = torch.fft.rfft(v, dim=-1)
        N1 = NL(v, verb)
        v1 = E2 * v + Q * N1

        N2a = NL(v1)
        v2a = E2 * v + Q * N2a

        N2b = NL(v2a)
        v2b = E2 * v1 + Q * (2 * N2b - N1)

        N3 = NL(v2b)
        v = E * v + N1 * f1 + 2 * (N2a + N2b) * f2 + N3 * f3
        return torch.fft.irfft(v, dim=-1)

    return inner


# This basically is just a hack for KS training
def custom_int(x0, int_function, steps, dt=0.25, dt_iter=1):
    out = [x0]
    x = x0
    for _ in range(steps):
        for _ in range(dt_iter):  # Execute the integration function `dt_iter` times per step
            x = int_function(x, None, dt / dt_iter)
        out.append(x)
    return torch.cat(out, 0)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


class TimeStack:
    def __call__(self, batch):
        return torch.cat(batch, dim=1)


class ChunkedTimeseries(Dataset):
    """Chunked timeseries dataset."""

    def __init__(self, seq, chunk_size=40, overlap=.25, transform=None):
        """
        Args:
            seq (torch.Tensor): Tensor containing time series
            chunk_size (int): size of chunks to produce
            overlap (float):
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.seq = seq
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.n = seq.shape[0]
        self.starts = np.array([i * chunk_size for i in range(self.n // chunk_size)])
        self.transform = transform

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.overlap <= 0:
            add_on = 0
        else:
            add_on = np.random.randint(int(self.overlap * self.chunk_size))
        start = min(self.starts[idx] + add_on,
                    self.n - self.chunk_size)
        sample = self.seq[start:start + self.chunk_size]
        if self.transform:
            sample = self.transform(sample)
        return sample


def mystery_operator(H_size, device, seed=None):
    """ Creates a random projection matrix for
    random lossy feature generation. """
    if seed is not None:
        torch.manual_seed(seed)
    proj = torch.randn(*H_size).to(device)
    def inner(x):
        return x @ proj
    return inner, proj

def partial_obs_operator(ori_dim, obs_inds, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    proj = torch.zeros(ori_dim, len(obs_inds), device=device)
    for i, obs_ind in enumerate(obs_inds):
        proj[obs_ind, i] = 1
    proj = proj.unsqueeze(0)
    def inner(x):
        return x @ proj
    return inner, proj

def get_dataloader(args, x0=None, test_only=False):

    t = torch.arange(0, args.train_steps * args.train_traj_num * args.dt, args.dt)
    gen_trajs = gen_data(args.dataset, t, v0=x0, sigma_v=args.sigma_v,
                                                 steps_test=args.test_steps * args.test_traj_num,
                                                 steps_valid=args.valid_steps,
                                                 check_disk=args.new_data,
                                                 steps_burn=args.burn_steps,
                                                 dt_iter=args.dt_iter,
                                                 prefix=f"{args.sigma_v}_{args.train_steps}_{args.train_traj_num}_{args.trail}",
                                                 test_only=test_only
                                                 )
    if test_only:
        true_v_valid, true_v_test = gen_trajs
    else:
        true_v, true_v_valid, true_v_test = gen_trajs

    # training data
    if not test_only:
        train_data = ChunkedTimeseries(true_v, args.train_steps, args.overlap_rate)
        train_loader = DataLoader(train_data, batch_size=args.batch_size,
                                shuffle=True, num_workers=args.num_loader_workers, collate_fn=TimeStack())
        print("Train loader length:", len(train_loader))


    # test data
    test_data = ChunkedTimeseries(true_v_test, args.test_steps, 0)
    test_loader = DataLoader(test_data, batch_size=args.test_batch_size,
                             shuffle=False, num_workers=args.num_loader_workers, collate_fn=TimeStack())

    print("Dataloader generated.")

    if test_only:
        return test_loader
    
    return train_loader, test_loader

def create_optimizer(model, args, apply_multiplier=False):
    # Check if model is a list
    if isinstance(model, list):
        # Aggregate all parameters from the list of models
        parameters = []
        for m in model:
            parameters += list(m.parameters())
    else:
        # If model is not a list, use its parameters directly
        parameters = model.parameters()

    # Create the optimizer based on whether it's SGD or AdamW
    if args.SGD:
        return SGD(parameters, lr=args.learning_rate,
                momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        if apply_multiplier:
            return AdamW(parameters, lr=args.learning_rate * 4,
                        weight_decay=args.weight_decay)
        return AdamW(parameters, lr=args.learning_rate,
                    weight_decay=args.weight_decay)


# def combined_lr_scheduler(args):
#     def lr_lambda(epoch):
#         if epoch < args.warm_up_epochs:
#             # Warm-up phase
#             return args.warm_up_rate ** epoch
#         else:
#             # After warm-up, apply step decay
#             decay_epochs = [int(e) for e in args.lr_decay_epochs.split(',')]
#             decay_factor = sum([epoch >= e for e in decay_epochs])
#             return args.warm_up_rate ** args.warm_up_epochs * (args.lr_decay_rate ** decay_factor)

#     return lr_lambda

def combined_lr_scheduler(args):
    def lr_lambda(epoch):
        if epoch < args.warm_up_epochs:
            # Exponential warm-up normalized to reach 1.0 at the end
            base = args.warm_up_rate
            normalized = base ** (epoch + 1) / base ** args.warm_up_epochs
            return normalized
        else:
            # After warm-up: step decay
            decay_epochs = [int(e) for e in args.lr_decay_epochs.split(',')]
            decay_factor = sum([epoch >= e for e in decay_epochs])
            return args.lr_decay_rate ** decay_factor
    return lr_lambda


def setup_optimizer_and_scheduler(model, args, apply_multiplier=False):
    optimizer = create_optimizer(model, args, apply_multiplier=apply_multiplier)
    lr_lambda = combined_lr_scheduler(args)
    scheduler = LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def projection_fun(V, H):
    if V.dim() == 2:
        return V @ H.T
    elif V.dim() == 3:
        B = V.shape[0]
        H = H.unsqueeze(0).expand(B, -1, -1)
        return torch.bmm(V, H.transpose(1, 2))
    else:
        raise ValueError("Input tensor dimension should be 2 or 3")


def save_checkpoint(model, optimizer, scheduler, filename="checkpoint.pth"):
    """
    Save the model, optimizer, and scheduler state dictionaries to a checkpoint file.
    Handles single models or lists of models, with support for DataParallel.
    """
    # Check if model is a list
    if isinstance(model, list):
        # Save state dicts for each model in the list
        model_state_dicts = [m.module.state_dict() if isinstance(m, torch.nn.DataParallel) else m.state_dict() for m in model]
    else:
        # Save state dict for a single model, handle DataParallel
        model_state_dicts = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()

    # Create the checkpoint dictionary
    checkpoint = {
        "model_state_dict": model_state_dicts,
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None
    }

    # Save the checkpoint
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(model, optimizer=None, scheduler=None, filename="checkpoint.pth", use_data_parallel=False):
    """
    Load the model, optimizer, and scheduler state dictionaries from a checkpoint file.
    Handles single models or lists of models, with support for DataParallel.
    """
    if not os.path.exists(filename):
        print(f"Checkpoint file {filename} does not exist.")
        return model, optimizer, scheduler

    # Load the checkpoint
    checkpoint = torch.load(filename, weights_only=True)

    # Check if model is a list and load the state dicts accordingly
    if isinstance(model, list):
        # Ensure the model list length matches the state dicts list length
        if len(model) != len(checkpoint["model_state_dict"]):
            raise ValueError("The number of models in the checkpoint does not match the number of provided models.")

        for i, m in enumerate(model):
            state_dict = checkpoint["model_state_dict"][i]
            # Handle DataParallel
            if use_data_parallel:
                state_dict = {"module." + k if not k.startswith("module.") else k: v for k, v in state_dict.items()}
            else:
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            m.load_state_dict(state_dict)
    else:
        # Handle single model
        state_dict = checkpoint["model_state_dict"]
        if use_data_parallel:
            state_dict = {"module." + k if not k.startswith("module.") else k: v for k, v in state_dict.items()}
        else:
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    # Load optimizer state dict if provided
    if optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state dict if provided
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print(f"Checkpoint loaded from {filename}")

    return model, optimizer, scheduler

def batch_covariance(x, y=None):
    B, N, D = x.shape
    mean_x = x.mean(dim=1, keepdim=True)
    x_centered = x - mean_x

    if y is None:
        cov_matrix = torch.bmm(x_centered.transpose(1, 2), x_centered) / N
    else:
        B_y, N_y, d = y.shape
        assert B == B_y and N == N_y
        mean_y = y.mean(dim=1, keepdim=True)
        y_centered = y - mean_y
        cov_matrix = torch.bmm(x_centered.transpose(1, 2), y_centered) / N
    
    return cov_matrix

def post_process(E, infl):
    """Inflate, Rotate.

    To avoid recomputing/recombining anomalies,
    this should have been inside `EnKF_analysis`

    But it is kept as a separate function

    - for readability;
    - to avoid inflating/rotationg smoothed states (for the `EnKS`).
    """

    if check_infl(infl):
        A, mu = center(E)
        # B, N, _ = E.shape
        # T = torch.eye(N).unsqueeze(0).expand(B, -1, -1).to(E.device)
        # T = infl * T
        # E = mu + torch.bmm(T, A)
        E = mu + infl * A
    return E

def mean0(E, axis=1, rescale=True):
    return center(E, axis=axis, rescale=rescale)[0]
