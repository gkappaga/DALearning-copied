import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os

from config import Config
from lorenz_data import LorenzDataset, generate_dataset, observation_map, rk4_step
from networks import DriftNetwork

def get_alphas_betas(s):
    """
    Boundary conditions: alpha_0=1, beta_0=0
                         alpha_1=0, beta_1=1
    [cite_start]Standard linear interpolation path often used in Stochastic Interpolants[cite: 1]:
    alpha_s = 1 - s
    beta_s = s
    """
    alpha = 1.0 - s
    beta = s
    
    # Derivatives w.r.t s
    d_alpha = -torch.ones_like(s)
    d_beta = torch.ones_like(s)
    
    return alpha, beta, d_alpha, d_beta

def get_sigma(s):
    """
    Noise schedule sigma_s. 
    Paper Eq 3 mentions sigma_s, boundary conditions sigma_0 = sigma_1 = 0.
    Standard choice: sigma_s = sqrt(s * (1-s)) or similar.
    However, paper uses sigma_s W_s where W_s is Wiener.
    [cite_start]Common choice in[cite: 1]: sigma_s = sqrt(2s(1-s)). 
    Let's use a simple bridge form satisfying boundaries: sigma_s = s(1-s).
    AMBIGUITY FLAG: Exact form of sigma_s not specified in text, only BCs. 
    We use sigma_s = s(1-s) which is robust.
    """
    return s * (1.0 - s)

def get_d_sigma(s):
    # derivative of s(1-s) -> 1 - 2s
    return 1.0 - 2.0 * s

def train(model, dataloader, save_path="flowdas_lorenz.pth"):
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.01, total_iters=Config.EPOCHS
    )
    criterion = nn.MSELoss()
    model.train()
    
    print(f"Starting training for {Config.EPOCHS} epochs...")
    
    pbar = tqdm(range(Config.EPOCHS), desc="Training", unit="epoch")
    
    for epoch in pbar:
        total_loss = 0.0
        
        for x0, x1 in dataloader:
            x0, x1 = x0.to(Config.DEVICE), x1.to(Config.DEVICE)
            
            # 1. Sample Time and Noise
            s = torch.rand(x0.shape[0], 1, device=Config.DEVICE)
            z = torch.randn_like(x0)
            
            # 2. Compute Coefficients
            alpha, beta, d_alpha, d_beta = get_alphas_betas(s)
            sigma = get_sigma(s)     
            d_sigma = get_d_sigma(s) 
            
            # 3. Construct Interpolant
            interpolant = alpha * x0 + beta * x1 + torch.sqrt(s) * sigma * z
            
            # 4. Construct Target Velocity
            target_velocity = d_alpha * x0 + d_beta * x1 + torch.sqrt(s) * d_sigma * z
            
            # 5. Optimization Step
            pred_velocity = model(interpolant, s, x0)
            loss = criterion(pred_velocity, target_velocity)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Update Scheduler
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        
        # Logging
        pbar.set_postfix({"Loss": f"{avg_loss:.6f}", "LR": f"{scheduler.get_last_lr()[0]:.6f}"})
        pbar.write(f"Epoch {epoch+1}/{Config.EPOCHS} | Loss: {avg_loss:.6f}")

    print(f"Saving model to {save_path}...")
    torch.save(model.state_dict(), save_path)
    return model

def inference(model, observations, init_state, true_states=None):
    """
    Algorithm 2: Inference with FlowDAS
    observations: [L_obs, Dim_obs] (Sequence of y)
    init_state: [Dim_state] (x_L-1)
    """
    model.eval()
    
    # Grid s
    N = Config.N_INTERPOLANT_STEPS
    # ds = 1.0 / N
    steps = torch.linspace(0, 1, N+1, device=Config.DEVICE)
    
    trajectories = []
    # Initial state x_{L-1}
    # Ensure initial state is on the correct device
    current_x = init_state.clone().to(Config.DEVICE).unsqueeze(0) # [1, 3]
    
    trajectories.append(current_x.cpu().numpy())
    
    num_steps = observations.shape[0]
    
    for k in range(num_steps):
        # FIX: Move the observation for this step to the correct DEVICE
        y_obs = observations[k].unsqueeze(0).to(Config.DEVICE) # [1, 1]
        
        # Start of transition for this step (s=0)
        X_s = current_x.clone()
        X_0 = current_x.clone() # Condition variable stays fixed
        
        for n in range(N):
            s_n = steps[n]
            # Time delta
            dt = steps[n+1] - steps[n]
            
            # We need to compute gradients w.r.t X_s for guidance (Line 13)
            # So we enable grad tracking on the current state variable
            X_s = X_s.detach().requires_grad_(True)
            
            # 1. Posterior Estimation (Eq 9 / Algorithm 2 Line 8)
            # Calculate drift b(Xs, X0)
            # Note: We use the model once for both the projection and the step
            s_tensor = torch.ones(X_s.shape[0], 1, device=Config.DEVICE) * s_n
            drift = model(X_s, s_tensor, X_0)
            
            # Generate J samples for Monte Carlo likelihood estimation
            J = Config.MC_SAMPLES_J
            
            # Replicate tensors for parallel sampling [J, 3]
            X_s_rep = X_s.repeat(J, 1)
            drift_rep = drift.repeat(J, 1)
            
            z_samples = torch.randn_like(X_s_rep)
            
            # Projection: X1_hat = X_s + drift * (1-s) + noise
            # We project from the CURRENT state X_s to s=1.
            X1_hat = X_s_rep + drift_rep * (1.0 - s_n) + \
                     Config.SIGMA_PROCESS * torch.sqrt(1.0 - s_n) * z_samples
            
            # 2. Gradient Computation (Algorithm 2 Line 13 & Eq 8)
            # We must differentiate through the observation map
            y_pred = observation_map(X1_hat) # [J, 1]
            
            # Squared Error ||y - A(X1)||^2
            # NOW BOTH TENSORS ARE ON THE SAME DEVICE
            dist_sq = (y_pred - y_obs)**2 # [J, 1]
            
            # Log Likelihood for weights ( - ||y - A(X)||^2 / 2gamma^2 )
            log_likelihood = -0.5 * dist_sq / (Config.SIGMA_OBS**2)
            
            # Weights w_j (Softmax over J samples)
            weights = torch.softmax(log_likelihood, dim=0) # [J, 1]
            
            # The Guidance Term: sum( w_j * || y - A(X1_j) ||^2 )
            weighted_energy = (weights.detach() * dist_sq).sum()
            
            # Compute gradient w.r.t X_s (Backprop through projection)
            grad_guidance = torch.autograd.grad(weighted_energy, X_s)[0] # [1, 3]
            
            # Detach X_s to finalize the previous computational graph
            X_s = X_s.detach()
            drift = drift.detach()
            
            # 3. Update X_s (Algorithm 2 Line 7 & 13)
            # Standard Euler-Maruyama Step (SDE)
            z_step = torch.randn_like(X_s)
            sigma_s = get_sigma(s_n) # s(1-s)
            
            # Base SDE step
            X_s_prime = X_s + drift * dt + sigma_s * torch.sqrt(dt) * z_step
            
            # Apply Guidance Correction
            X_s = X_s_prime - Config.GUIDANCE_STEP_ZETA * grad_guidance
            
        current_x = X_s
        trajectories.append(current_x.cpu().numpy())
        
    return np.array(trajectories)

def main():
    # 1. Generate Data
    print("Generating Training Data...")
    raw_data = generate_dataset(Config.N_TRAIN_TRAJ, Config.LEN_TRAIN_TRAJ)
    dataset = LorenzDataset(raw_data)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    # 2. Train or Load Model
    model_path = "flowdas_lorenz.pth"
    model = DriftNetwork().to(Config.DEVICE)
    
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        # Load weights and map to the correct device
        model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    else:
        print("No pre-trained model found. Starting training...")
        model = train(model, dataloader, save_path=model_path)
    
    # 3. Test / Inference
    print("Running Inference...")
    test_data = generate_dataset(Config.N_TEST_TRAJ, Config.LEN_TEST_TRAJ + Config.L_LAG)
    
    # Initial state is index 0
    # Observations start from index 1 to 15 (15 steps)
    init_state = test_data[:, 0, :]
    truth_trajectory = test_data[:, 1:, :]
    
    # Generate noisy observations for the truth
    obs_truth = observation_map(truth_trajectory)
    obs_noise = torch.randn_like(obs_truth) * Config.SIGMA_OBS
    observations = obs_truth + obs_noise
    
    # Run FlowDAS for the first trajectory
    idx = 0
    y_seq = observations[idx] # [15, 1]
    x_init = init_state[idx] # [3]
    gt = truth_trajectory[idx] # [15, 3]
    
    estimated_traj = inference(model, y_seq, x_init)
    
    # estimated_traj includes initial state, so length is 16. Drop init to compare.
    estimated_steps = estimated_traj[1:, 0, :] 
    
    # Metrics: Calculate RMSE
    mse = np.mean((estimated_steps - gt.cpu().numpy())**2)
    rmse = np.sqrt(mse)
    
    print(f"Inference RMSE (Traj {idx}): {rmse:.6f}")
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(gt.cpu().numpy()[:, 0], label='True State (a)', color='black', marker='x')
    plt.plot(estimated_steps[:, 0], label='FlowDAS (a)', color='red', marker='.')
    plt.plot(y_seq.cpu().numpy(), label='Observations (Arctan a)', color='green', alpha=0.3)
    plt.legend()
    plt.title(f'Lorenz 63 Data Assimilation (RMSE: {rmse:.4f})')
    plt.savefig('lorenz_results.png')
    print("Results saved to lorenz_results.png")

if __name__ == "__main__":
    main()