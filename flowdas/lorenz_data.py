import torch
import numpy as np
from config import Config

def rk4_step(state, dt, sigma_w):
    """
    Implements Eq S.34 - S.38 (Appendix D.2.1)
    Deterministic RK4 update followed by stochastic noise addition.
    """
    a, b, c = state[..., 0], state[..., 1], state[..., 2]
    
    # k1
    k1_a = dt * Config.MU * (b - a)
    k1_b = dt * (a * (Config.RHO - c) - b)
    k1_c = dt * (a * b - Config.TAU * c)
    
    # k2
    a2 = a + 0.5 * k1_a
    b2 = b + 0.5 * k1_b
    c2 = c + 0.5 * k1_c
    k2_a = dt * Config.MU * (b2 - a2)
    k2_b = dt * (a2 * (Config.RHO - c2) - b2)
    k2_c = dt * (a2 * b2 - Config.TAU * c2)
    
    # k3
    a3 = a + 0.5 * k2_a
    b3 = b + 0.5 * k2_b
    c3 = c + 0.5 * k2_c
    k3_a = dt * Config.MU * (b3 - a3)
    k3_b = dt * (a3 * (Config.RHO - c3) - b3)
    k3_c = dt * (a3 * b3 - Config.TAU * c3)
    
    # k4
    a4 = a + k3_a
    b4 = b + k3_b
    c4 = c + k3_c
    k4_a = dt * Config.MU * (b4 - a4)
    k4_b = dt * (a4 * (Config.RHO - c4) - b4)
    k4_c = dt * (a4 * b4 - Config.TAU * c4)
    
    # Update deterministic
    da = (k1_a + 2*k2_a + 2*k3_a + k4_a) / 6.0
    db = (k1_b + 2*k2_b + 2*k3_b + k4_b) / 6.0
    dc = (k1_c + 2*k2_c + 2*k3_c + k4_c) / 6.0
    
    next_state_det = torch.stack([a + da, b + db, c + dc], dim=-1)
    
    # Add noise (Eq 11)
    # The noise in Eq 11 is dW. For discrete step dt, noise is N(0, dt) * sigma?
    # Paper says: "xi is process noise with std deviation sigma=0.25".
    # Usually in SDE discretization: x_{k+1} = f(x_k) + sigma * sqrt(dt) * N(0,1)
    # However, Eq 1 just says "+ xi_k". If xi_k is the noise *per step*, we use sigma directly.
    # Given Eq 1 format x_{k+1} = Psi(x_k) + xi_k, xi_k is drawn from N(0, sigma^2).
    noise = torch.randn_like(next_state_det) * sigma_w
    
    return next_state_det + noise

def observation_map(state):
    """
    Eq 12: y = arctan(a) + eta
    """
    # Only the first component 'a' is observed via arctan
    a = state[..., 0:1]
    return torch.atan(a)

def generate_dataset(num_traj, len_traj, burn_in=1000):
    """
    Generates dataset as per 'Dataset and experiments' section.
    Includes burn-in to reach stationary regime.
    """
    # Initial random states
    states = torch.randn(num_traj, 3)
    
    # Burn-in
    for _ in range(burn_in):
        states = rk4_step(states, Config.DT_SIM, Config.SIGMA_PROCESS)
        
    trajectory = []
    # Generate actual trajectory
    # We need to pairs (x_k, x_{k+1})
    # We generate sequence of states.
    curr_state = states
    trajectory.append(curr_state)
    
    for _ in range(len_traj - 1):
        curr_state = rk4_step(curr_state, Config.DT_SIM, Config.SIGMA_PROCESS)
        trajectory.append(curr_state)
        
    # Shape: [Time, Batch, Dim] -> [Batch, Time, Dim]
    data = torch.stack(trajectory).permute(1, 0, 2)
    return data

class LorenzDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        # Flatten data to pairs (x_k, x_{k+1}) as per Appendix B.2
        # Input data shape: [Batch, Time, Dim]
        # X0: all states except last
        # X1: all states except first
        self.x0 = data[:, :-1, :].reshape(-1, 3)
        self.x1 = data[:, 1:, :].reshape(-1, 3)
        
    def __len__(self):
        return self.x0.shape[0]
    
    def __getitem__(self, idx):
        return self.x0[idx], self.x1[idx]