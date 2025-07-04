import torch
import torch.nn as nn

def compute_es(ens_states, true_states, norm_p=1):
    """
    Computes the Energy Score (ES).
    ES(F, y) = E_F[||X - y||^norm_p] - (1/2) * E_F[||X - X'||^norm_p]
    where ||.|| is the L2 norm. For classical ES, norm_p is 1.

    Args:
        ens_states (torch.Tensor): Ensemble predictions. Shape: [T, B, N, D].
        true_states (torch.Tensor): Ground truth. Shape: [T, B, D].
        norm_p (int, float): The power to which the L2 norm of distances is raised. Defaults to 1.

    Returns:
        torch.Tensor: Energy Score. Shape: [T, B].
    """
    T, B, N, D = ens_states.shape

    if N <= 1: # ES is not well-defined or is trivially zero for N <= 1.
        return torch.zeros(T, B, device=ens_states.device, dtype=ens_states.dtype)

    # First term: E_F[||X - y||^norm_p]
    # Approximate E_F by averaging over ensemble members.
    true_expanded = true_states.unsqueeze(2)  # Shape: [T, B, 1, D]
    # L2 norm for distances, then raise to norm_p.
    dist_to_true = torch.norm(ens_states - true_expanded, p=2, dim=-1) # Shape: [T, B, N]
    if norm_p != 1:
        dist_to_true = torch.pow(dist_to_true, norm_p)
    
    term_obs = torch.mean(dist_to_true, dim=2)  # Shape: [T, B]
    
    # Second term: (1/2) * E_F[||X - X'||^norm_p]
    # Approximate E_F by averaging over distinct pairs of ensemble members.
    # sum_{i!=j} ||x_i - x_j||^p / (N*(N-1))
    
    # Efficiently calculate sum of pairwise distances between ensemble members
    # This avoids explicit loops for better performance if N is large,
    # but for clarity and given typical N, loops are acceptable as in original code.
    # Here, we stick to the loop for clarity and consistency with original.
    
    sum_pairwise_dist = torch.zeros(T, B, device=ens_states.device, dtype=ens_states.dtype)
    for i in range(N):
        for j in range(i + 1, N): # Iterate over distinct pairs (i < j)
            dist_pair = torch.norm(ens_states[:, :, i, :] - ens_states[:, :, j, :], p=2, dim=-1) # Shape: [T, B]
            if norm_p != 1:
                dist_pair = torch.pow(dist_pair, norm_p)
            sum_pairwise_dist += dist_pair
            
    # Average over N*(N-1)/2 distinct pairs for E_F[||X - X'||^norm_p]
    # The sum_{i!=j} has N*(N-1) terms. sum_{i<j} has N*(N-1)/2 terms.
    # E_F[||X - X'||^p] is approximated by ( sum_{i<j} ||x_i - x_j||^p ) / (N*(N-1)/2)
    # which is (2 * sum_pairwise_dist) / (N * (N-1))
    term_pair_expectation = (2 * sum_pairwise_dist) / (N * (N - 1)) # Shape: [T, B]
    
    es = term_obs - 0.5 * term_pair_expectation  # Shape: [T, B]
    return es

def compute_kernel_es(ens_states, true_states, sigma=None):
    """
    Computes the Kernel Energy Score (kES) using a Gaussian kernel.
    kES(F, y) = -E_F[k(X, y)] + (1/2) * E_F[k(X, X')]
    where k(x,y) = exp(-||x-y||_L2^2 / (2*sigma^2)). Lower is better.

    Args:
        ens_states (torch.Tensor): Ensemble predictions. Shape: [T, B, N, D].
        true_states (torch.Tensor): Ground truth. Shape: [T, B, D].
        sigma (float, optional): Kernel bandwidth. If None, it's estimated as the median
                                 of L2 distances between ensemble members and true_states per (T,B).

    Returns:
        torch.Tensor: Kernel Energy Score. Shape: [T, B].
    """
    T, B, N, D = ens_states.shape
    device = ens_states.device
    dtype = ens_states.dtype

    if N == 0 : # Cannot compute if ensemble is empty
        return torch.zeros(T, B, device=device, dtype=dtype)
    if N == 1 and sigma is None: # Median distance for sigma estimation needs at least 1 point,
                                 # but pairwise term is ill-defined for N=1
        # Handle N=1 case: pairwise term is zero or undefined.
        # If sigma is not provided, need a fallback or error for N=1.
        # For MMD(F, delta_y) with N=1, E_F[k(X,X')] = k(x1,x1)=1. So kES = -k(x1,y) + 0.5
        # If sigma is None and N=1, true_expanded - ens_states will be used for median, which works.
        pass


    current_sigma_val = None
    if sigma is None:
        true_expanded_for_sigma = true_states.unsqueeze(2)  # Shape: [T, B, 1, D]
        distances_for_sigma = torch.norm(ens_states - true_expanded_for_sigma, p=2, dim=-1)  # Shape: [T, B, N]
        # Median over N members for each (T,B)
        current_sigma_val = torch.median(distances_for_sigma, dim=2, keepdim=False)[0] + 1e-8 # Shape: [T, B]
    else:
        current_sigma_val = torch.tensor(sigma, device=device, dtype=dtype).expand(T, B) + 1e-8 # Shape: [T, B]
    
    # Reshape sigma for broadcasting: [T, B, 1, 1]
    current_sigma_val_sq = (current_sigma_val.unsqueeze(-1).unsqueeze(-1)) ** 2


    # First term: -E_F[k(X, y)]
    true_expanded = true_states.unsqueeze(2)  # Shape: [T, B, 1, D]
    diff_obs = ens_states - true_expanded      # Shape: [T, B, N, D]
    dist_sq_obs = torch.sum(diff_obs ** 2, dim=-1, keepdim=True)  # Shape: [T, B, N, 1]
    k_obs = torch.exp(-dist_sq_obs / (2 * current_sigma_val_sq))  # Shape: [T, B, N, 1]
    # Average over ensemble members N
    term1_ef_k_xy = torch.mean(k_obs, dim=2).squeeze(-1)  # Shape: [T, B]


    # Second term: (1/2) * E_F[k(X, X')]
    if N <= 1: # Pairwise term is zero or ill-defined
        term2_ef_k_xx_prime_avg = torch.zeros_like(term1_ef_k_xy)
        if N == 1: # E_F[k(X,X')] for N=1 could be k(x1,x1) = 1
             term2_ef_k_xx_prime_avg = torch.ones_like(term1_ef_k_xy)


    else: # N > 1
        sum_pairwise_kernel = torch.zeros(T, B, device=device, dtype=dtype)
        for i in range(N):
            xi = ens_states[:, :, i:i+1, :]  # Shape: [T, B, 1, D]
            for j in range(i + 1, N): # Iterate over distinct pairs (i < j)
                xj = ens_states[:, :, j:j+1, :]  # Shape: [T, B, 1, D]
                diff_pair = xi - xj              # Shape: [T, B, 1, D]
                dist_sq_pair = torch.sum(diff_pair ** 2, dim=-1, keepdim=True)  # Shape: [T, B, 1, 1]
                k_pair = torch.exp(-dist_sq_pair / (2 * current_sigma_val_sq))  # Shape: [T, B, 1, 1]
                sum_pairwise_kernel += k_pair.squeeze(-1).squeeze(-1)  # Accumulate [T, B]

        # E_F[k(X, X')] approximated by average over N*(N-1)/2 distinct pairs
        term2_ef_k_xx_prime_avg = (2 * sum_pairwise_kernel) / (N * (N - 1)) # Shape: [T, B]
    
    kernel_es_val = -term1_ef_k_xy + 0.5 * term2_ef_k_xx_prime_avg
    return kernel_es_val


def compute_loss(ens_tensor, batch_v, loss_type, ignore_first=0, end_ind=None, 
                 valid_B_mask=None, norm_p=1, kes_sigma=1, return_sum=False):
    """
    Computes loss. Supports various types including L2, ES, and kernel ES.

    Args:
        ens_tensor (torch.Tensor): Ensemble predictions. Shape: [T, B, N, D].
        batch_v (torch.Tensor): Ground truth. Shape: [T, B, D].
        loss_type (str): Type of loss: 'l2', 'nl2' (normalized L2), 'rmse', 
                         'es', 'nes' (normalized ES), 'tnes' (trajectory normalized ES),
                         'kes' (kernel ES), 'nkes' (normalized kES), 'tnkes' (trajectory normalized kES).
        ignore_first (int): Number of initial time steps to ignore.
        end_ind (int, optional): Last time step index to consider. Defaults to end of trajectory.
        valid_B_mask (torch.Tensor, optional): Boolean mask for valid batch items.
                                               Shape: [B] or [T, B]. Defaults to all valid.
        norm_p (int, float): Exponent for distances in ES (classical ES uses 1 for this exponent,
                             meaning the L2 norm itself, not L2 norm squared).
                             Also used as p for torch.norm in normalization terms for NES, NKES.
        kes_sigma (float): Bandwidth sigma for kernel ES.
        return_sum (bool): If True, returns sum of losses over valid elements. Else, returns mean.

    Returns:
        torch.Tensor: Computed loss (scalar).
    """
    full_time_steps = batch_v.size(0)
    
    if end_ind is None:
        end_ind = full_time_steps - 1 
    end_ind = min(end_ind, full_time_steps - 1)

    if ignore_first > end_ind:
        return torch.tensor(0.0, device=batch_v.device, requires_grad=True)

    if valid_B_mask is None:
        valid_B_mask = torch.ones(full_time_steps, batch_v.size(1), dtype=torch.bool, device=batch_v.device)
    elif valid_B_mask.ndim == 1: # Batch dimension only
        valid_B_mask = valid_B_mask.unsqueeze(0).expand(full_time_steps, -1)

    valid_B_mask_sliced = valid_B_mask[ignore_first:end_ind + 1, :]

    ens_states_timed = ens_tensor[ignore_first:end_ind + 1, :, :, :] 
    true_states_timed = batch_v[ignore_first:end_ind + 1, :, :]   
    
    if not valid_B_mask_sliced.any():
        return torch.tensor(0.0, device=batch_v.device, requires_grad=True)

    ens_mean_timed = torch.mean(ens_states_timed, dim=2)  # Shape: [selected_time, B, D]
    
    loss_values_per_element = None # Will store [selected_time, B] shaped losses

    if loss_type == "l2":
        loss_values_per_element = torch.sum((ens_mean_timed - true_states_timed) ** 2, dim=2)
    elif loss_type == 'nl2':
        error_norm_2 = torch.sum((ens_mean_timed - true_states_timed) ** 2, dim=2)
        true_norm_2 = torch.sum(true_states_timed ** 2, dim=2)
        loss_values_per_element = error_norm_2 / (true_norm_2 + 1e-8)
    elif loss_type == 'rmse':
        mse_features = (ens_mean_timed - true_states_timed) ** 2
        loss_values_per_element = torch.sqrt(torch.sum(mse_features, dim=2) + 1e-8)
    elif loss_type == 'es':
        loss_values_per_element = compute_es(ens_states_timed, true_states_timed, norm_p=norm_p)
    elif loss_type == 'nes' or loss_type == 'tnes':
        es_vals = compute_es(ens_states_timed, true_states_timed, norm_p=norm_p) # Shape [T_slice, B]
        true_norm_vals = torch.norm(true_states_timed, p=norm_p, dim=2) # Shape [T_slice, B]
        if loss_type == 'nes':
            loss_values_per_element = es_vals / (true_norm_vals + 1e-8)
        else: # tnes
            sum_es_per_batch = torch.zeros(batch_v.size(1), device=batch_v.device, dtype=ens_tensor.dtype)
            sum_norm_per_batch = torch.zeros(batch_v.size(1), device=batch_v.device, dtype=ens_tensor.dtype)
            for b_idx in range(batch_v.size(1)):
                valid_time_for_b = valid_B_mask_sliced[:, b_idx]
                if valid_time_for_b.any():
                    sum_es_per_batch[b_idx] = torch.sum(es_vals[valid_time_for_b, b_idx])
                    sum_norm_per_batch[b_idx] = torch.sum(true_norm_vals[valid_time_for_b, b_idx])
            
            final_batch_mask = valid_B_mask_sliced.any(dim=0) # Batches with at least one valid time step
            if not final_batch_mask.any(): return torch.tensor(0.0, device=batch_v.device, requires_grad=True)
            
            ratios = sum_es_per_batch[final_batch_mask] / (sum_norm_per_batch[final_batch_mask] + 1e-8)
            if return_sum: return torch.sum(ratios)
            return torch.mean(ratios)
            
    elif loss_type == 'kes':
        loss_values_per_element = compute_kernel_es(ens_states_timed, true_states_timed, sigma=kes_sigma)
    elif loss_type == 'nkes' or loss_type == 'tnkes':
        kes_vals = compute_kernel_es(ens_states_timed, true_states_timed, sigma=kes_sigma) # Shape [T_slice, B]
        true_norm_vals = torch.norm(true_states_timed, p=norm_p, dim=2) # Shape [T_slice, B]
        if loss_type == 'nkes':
            loss_values_per_element = kes_vals / (true_norm_vals + 1e-8)
        else: # tnkes
            sum_kes_per_batch = torch.zeros(batch_v.size(1), device=batch_v.device, dtype=ens_tensor.dtype)
            sum_norm_per_batch = torch.zeros(batch_v.size(1), device=batch_v.device, dtype=ens_tensor.dtype)
            for b_idx in range(batch_v.size(1)):
                valid_time_for_b = valid_B_mask_sliced[:, b_idx]
                if valid_time_for_b.any():
                    sum_kes_per_batch[b_idx] = torch.sum(kes_vals[valid_time_for_b, b_idx])
                    sum_norm_per_batch[b_idx] = torch.sum(true_norm_vals[valid_time_for_b, b_idx])

            final_batch_mask = valid_B_mask_sliced.any(dim=0)
            if not final_batch_mask.any(): return torch.tensor(0.0, device=batch_v.device, requires_grad=True)

            ratios = sum_kes_per_batch[final_batch_mask] / (sum_norm_per_batch[final_batch_mask] + 1e-8)
            if return_sum: return torch.sum(ratios)
            return torch.mean(ratios)
    else:
        raise NotImplementedError(f"Loss type '{loss_type}' is not implemented")
    
    masked_loss_values = loss_values_per_element[valid_B_mask_sliced]
    if masked_loss_values.numel() == 0:
        return torch.tensor(0.0, device=batch_v.device, requires_grad=True)

    if return_sum:
        return torch.sum(masked_loss_values)
    else:
        return torch.mean(masked_loss_values)
    
import torch

def compute_mean_pen(
    ens_tensor,       # [B, N, D]
    true_v,           # [B, D]
    valid_B_mask=None,# None or [B] bool mask
    return_sum=False,
    H_info=None,
    A = None,
    B_mat = None,
    a = None,
    args = None,
    lambda1 = 0.0,
):
    """
    Compute per‐batch loss for a single (latest) time‐step.
    """
    B, N, D = ens_tensor.shape

    # 1) build mask over B
    if valid_B_mask is None:
        mask = torch.ones(B, dtype=torch.bool, device=ens_tensor.device)
    else:
        mask = valid_B_mask
        if mask.ndim != 1 or mask.size(0) != B:
            raise ValueError("valid_B_mask must be shape [B]")

    # 2) collapse ensemble dim → [B, D]
    ens_mean = ens_tensor.mean(dim=1)   # (B, D)
    if lambda1 > 0.0:
        if H_info is None or A is None or B_mat is None or a is None:
            raise ValueError("Must pass H_info, A_mat, B_mat, a_vec to use lambda1>0")
        H_fun, H = H_info
        d = H.shape[0]

        # True observation at batch: [B,d]
        y_obs = H_fun(true_v.unsqueeze(1)).squeeze(1)  # (B,d)

        # Predicted obs from ensemble: [B,N,d]
        hv = H_fun(ens_tensor)                         # (B,N,d)

        # Sample means
        v_bar = ens_mean                              # (B,D)
        y_bar = hv.mean(dim=1)                        # (B,d)

        # Sample covariances
        Vp = ens_tensor - v_bar.unsqueeze(1)          # (B,N,D)
        Hp = hv         - y_bar.unsqueeze(1)          # (B,N,d)

        Cvv = torch.bmm(Vp.transpose(1,2), Vp)/(N-1)   # (B,D,D)
        Cyy = torch.bmm(Hp.transpose(1,2), Hp)/(N-1)   # (B,d,d)
        Cvy = torch.bmm(Vp.transpose(1,2), Hp)/(N-1)   # (B,D,d)

        # Invert Cyy safely
        eps = getattr(args, "cov_eps", 1e-3)
        Cyy_j = Cyy + eps * torch.eye(Cyy.shape[-1], device = args.device).unsqueeze(0)
        Cyy_inv = torch.inverse(Cyy_j)

        # Analytic posterior mean: v_bar + Cvy Cyy^{-1} (y_obs - y_bar)
        innov = (y_obs - y_bar).unsqueeze(-1)          # (B,d,1)
        m_th = v_bar + torch.bmm(Cvy, Cyy_inv).bmm(innov).squeeze(-1)  # (B,D)

        # learned mean
        m_nn = (
            torch.bmm(A, v_bar.unsqueeze(-1)).squeeze(-1) +
            torch.bmm(B_mat, y_bar.unsqueeze(-1)).squeeze(-1) +
            a
        )  # (B,D)

        # L2 norm penalty
        mean_diff = m_th - m_nn                    # (B,D)
        # L = L + lambda1 * torch.norm(mean_diff, dim=1)/torch.norm(m_th, dim = 1)
        # mean_pen = lambda1 * torch.norm(mean_diff, dim=1)/torch.norm(m_th, dim = 1)
        mean_pen = lambda1 * torch.norm(mean_diff, dim=1)
        if return_sum:
            return mean_pen.sum()
        else:
            return mean_pen.mean()
    return torch.tensor(0.0, device=ens_tensor.device, requires_grad=False)

    # 5) If desired, analytic vs learned covariance matching
def compute_cov_pen(
    ens_tensor,       # [B, N, D]
    valid_B_mask=None,# None or [B] bool mask
    return_sum=False,
    H_info=None,
    A = None,
    B_mat = None,
    a = None,
    args = None,
    lambda2 = 0.0
):
    B, N, D = ens_tensor.shape

    # 1) build mask over B
    if valid_B_mask is None:
        mask = torch.ones(B, dtype=torch.bool, device=ens_tensor.device)
    else:
        mask = valid_B_mask
        if mask.ndim != 1 or mask.size(0) != B:
            raise ValueError("valid_B_mask must be shape [B]")

    # 2) collapse ensemble dim → [B, D]
    ens_mean = ens_tensor.mean(dim=1)   # (B, D)
    if lambda2 > 0.0:
        if A is None or B_mat is None:
            raise ValueError("Must pass A_mat, B_mat to use lambda2>0")

        # reuse Cvv, Cyy, Cvy from above, or recompute if lambda1==0
        H_fun, H = H_info
        if 'Cvv' not in locals():
            # recompute as in step 4
            hv = H_fun(ens_tensor); y_bar = hv.mean(dim=1)
            v_bar = ens_mean
            Vp, Hp = ens_tensor - v_bar.unsqueeze(1), hv - y_bar.unsqueeze(1)
            Cvv = torch.bmm(Vp.transpose(1,2), Vp)/(N-1)
            Cyy = torch.bmm(Hp.transpose(1,2), Hp)/(N-1)
            Cvy = torch.bmm(Vp.transpose(1,2), Hp)/(N-1)
            # Invert Cyy safely
            eps = getattr(args, "cov_eps", 1e-3)
            Cyy_j = Cyy + eps * torch.eye(Cyy.shape[-1], device = args.device).unsqueeze(0)
            Cyy_inv = torch.inverse(Cyy_j)

        # Predicted covariance
        term1 = A.bmm(Cvv).bmm(A.transpose(-2,-1))
        term2 = A.bmm(Cvy).bmm(B_mat.transpose(-2,-1))
        term3 = B_mat.bmm(Cvy.transpose(-2,-1)).bmm(A.transpose(-2,-1)) #just transpose term 2
        term4 = B_mat.bmm(Cyy).bmm(B_mat.transpose(-2,-1))
        Cov_pred = term1 + term2 + term3 + term4            # (B,D,D)

        # True covariance
        Cov_true = Cvv - torch.bmm(Cvy, Cyy_inv).bmm(Cvy.transpose(-2,-1))
        cov_diff = Cov_pred - Cov_true             # (B,D,D)

        # global Frobenius norm per batch
        cov_fro = torch.norm(cov_diff, p='fro', dim=(1,2))  # (B,)
        # L = L + lambda2 * cov_fro/torch.norm(Cov_true, dim = (1,2)) #divide by torch.norm(Cov_true, dim = (1, 2))
        # cov_pen = lambda2 * cov_fro/torch.norm(Cov_true, dim = (1,2))
        cov_pen = lambda2 * cov_fro
        if return_sum:
            return cov_pen.sum()
        else:
            return cov_pen.mean()
    return torch.tensor(0.0, device=ens_tensor.device, requires_grad=False)


class MultiLossUncertaintyWeight(nn.Module):
    def __init__(self, num_losses):
        super(MultiLossUncertaintyWeight, self).__init__()
        self.log_sigma = nn.Parameter(torch.zeros(num_losses)) # Learnable log(variance) for each loss

    def forward(self, losses): # losses: list or tensor of individual losses
        total_loss = 0
        for i, loss_val in enumerate(losses):
            precision = torch.exp(-self.log_sigma[i]) # Corresponds to 1/sigma^2
            total_loss += precision * loss_val + 0.5 * self.log_sigma[i] # Maximize likelihood formulation
        return total_loss

if __name__ == "__main__":
    time_steps = 8
    batch_size = 4
    ensemble_size = 10
    feature_dim = 3
    
    ens_states = torch.randn(time_steps, batch_size, ensemble_size, feature_dim)
    true_states = torch.randn(time_steps, batch_size, feature_dim)
    
    sample_valid_B_mask = torch.ones(batch_size, dtype=torch.bool)
    sample_valid_B_mask[batch_size // 2:] = False # First half of batches valid
    
    print("--- Testing ES and kES (lower is better) ---")
    # Classical ES uses norm_p=1 for the L2 distances
    es_loss_classical_mean = compute_loss(ens_states, true_states, loss_type='es', norm_p=1, valid_B_mask=sample_valid_B_mask)
    # ES with squared L2 distances (norm_p=2)
    es_loss_squared_mean = compute_loss(ens_states, true_states, loss_type='es', norm_p=2, valid_B_mask=sample_valid_B_mask)
    
    kes_loss_sigma1_mean = compute_loss(ens_states, true_states, loss_type='kes', kes_sigma=1.0, valid_B_mask=sample_valid_B_mask)
    kes_loss_sigma_auto_mean = compute_loss(ens_states, true_states, loss_type='kes', kes_sigma=None, valid_B_mask=sample_valid_B_mask)

    print(f"Classical Energy Score (mean, exponent=1): {es_loss_classical_mean.item():.6f}")
    print(f"Energy Score (mean, exponent=2)        : {es_loss_squared_mean.item():.6f}")
    print(f"Kernel ES (mean, sigma=1.0)            : {kes_loss_sigma1_mean.item():.6f}")
    print(f"Kernel ES (mean, auto sigma)           : {kes_loss_sigma_auto_mean.item():.6f}")

    print("\n--- Testing return_sum=True ---")
    es_loss_classical_sum = compute_loss(ens_states, true_states, loss_type='es', norm_p=1, valid_B_mask=sample_valid_B_mask, return_sum=True)
    kes_loss_sigma_auto_sum = compute_loss(ens_states, true_states, loss_type='kes', kes_sigma=None, valid_B_mask=sample_valid_B_mask, return_sum=True)

    print(f"Classical Energy Score (sum, exponent=1) : {es_loss_classical_sum.item():.6f}")
    print(f"Kernel ES (sum, auto sigma)            : {kes_loss_sigma_auto_sum.item():.6f}")

    # Verification
    num_valid_elements = torch.sum(sample_valid_B_mask).item() * time_steps
    if num_valid_elements > 0:
        print(f"\nVerification (Classical ES): sum/N_elements = {es_loss_classical_sum.item()/num_valid_elements:.6f}, mean = {es_loss_classical_mean.item():.6f}")
        assert torch.isclose(es_loss_classical_sum/num_valid_elements, es_loss_classical_mean, atol=1e-5)

    print("\n--- Testing Trajectory Normalized Scores (lower is better) ---")
    tnes_loss_classical_mean = compute_loss(ens_states, true_states, loss_type='tnes', norm_p=1, valid_B_mask=sample_valid_B_mask)
    tnkes_loss_sigma_auto_mean = compute_loss(ens_states, true_states, loss_type='tnkes', norm_p=2, kes_sigma=None, valid_B_mask=sample_valid_B_mask)
    print(f"Trajectory Norm ES (mean, exponent=1)  : {tnes_loss_classical_mean.item():.6f}")
    print(f"Trajectory Norm kES (mean, auto sigma): {tnkes_loss_sigma_auto_mean.item():.6f}")
    
    tnes_loss_classical_sum = compute_loss(ens_states, true_states, loss_type='tnes', norm_p=1, valid_B_mask=sample_valid_B_mask, return_sum=True)
    print(f"Trajectory Norm ES (sum, exponent=1)   : {tnes_loss_classical_sum.item():.6f}")
    num_valid_batch_elements = torch.sum(sample_valid_B_mask).item()
    if num_valid_batch_elements > 0:
         print(f"Verification (TNES): sum/N_batch = {tnes_loss_classical_sum.item()/num_valid_batch_elements:.6f}, mean = {tnes_loss_classical_mean.item():.6f}")
         assert torch.isclose(tnes_loss_classical_sum/num_valid_batch_elements, tnes_loss_classical_mean, atol=1e-5)
    

    # dummy testing for compute_loss_last
    print("\n--- Testing compute_loss_last ---")
    ens_tensor = torch.randn(batch_size, ensemble_size, feature_dim)
    true_v = torch.randn(batch_size, feature_dim)
    valid_B_mask = torch.ones(batch_size, dtype=torch.bool)
    loss_type = 'l2'  # Example loss type
    #create dummy values for H_info, A, B_mat, a, args
    H_info = (lambda x: x, torch.eye(feature_dim))  # Dummy H
    # A and B must be 3d tensors for batch matrix multiplication
    A = torch.randn(batch_size, feature_dim, feature_dim)
    B_mat = torch.randn(batch_size, feature_dim, feature_dim)
    a = torch.randn(feature_dim)

    args = type('', (), {})()  # Create a dummy args object
    # add device to args
    args.device = ens_tensor.device

    loss_last_mean, mean_diff, cov_fro = compute_loss_last(
        ens_tensor, true_v, loss_type, valid_B_mask=valid_B_mask,
        norm_p=1, kes_sigma=1.0, return_sum=False,
        ignore_first=0, lambda1=0.1, lambda2=0.5, H_info=H_info, A=A, B_mat=B_mat, a=a, args=args
    )
    print(f"Loss (last mean): {loss_last_mean.item():.6f}")
    print(f"Mean difference: {mean_diff.item():.6f}")
    print(f"Covariance Frobenius norm: {cov_fro.item():.6f}")



    print("\nMain tests completed. Review output values.")