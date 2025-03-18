import torch
import numpy as np

def adjust_ld_step(current_ld_step, current_acceptance_rate, target_acceptance_rate=0.574, adjustment_factor=0.01):
    """
    Adjust the Langevin dynamics step size based on the current acceptance rate.
    """
    if current_acceptance_rate > target_acceptance_rate:
        return current_ld_step + adjustment_factor * current_ld_step
    else:
        return current_ld_step - adjustment_factor * current_ld_step

def langevin_dynamics(x, log_reward, device, T, args):
    """
    Produce a trajectory of shape [T, N, ndim],
    where T = args.max_iter_ls, N = x.shape[0], and ndim = x.shape[1].
    If a proposal is rejected at step i, that chain remains at its old state.
    """
    # Number of parallel samples (chains)
    N, D = x.shape  
    #T = args.max_iter_ls  # total MCMC steps

    # We'll store states at each iteration:
    # trajectories[i] will be the states of all chains after step i
    trajectories = torch.zeros(T, N, D, device=device)
    log_r_trajectories = torch.zeros(T, N, device=device)

    # Initial states
    x_current = x.clone().to(device)
    log_r_current = log_reward(x_current)
    
    # Variables for acceptance stats
    acceptance_count = 0
    total_proposals = 0
    acceptance_rate = 0.0
    ld_step = args.ld_step  # initial step size

    for i in range(T):
        x_current = x_current.requires_grad_(True)

        # Gradient wrt x
        r_grad_original = torch.autograd.grad(log_reward(x_current).sum(), x_current)[0]
        
        # Possibly adjust ld_step (if scheduling is enabled and i>0)
        if args.ld_schedule and i > 0:
            ld_step = adjust_ld_step(ld_step, acceptance_rate, target_acceptance_rate=args.target_acceptance_rate)

        # Propose new_x
        noise = torch.randn_like(x_current, device=device)
        new_x = x_current + ld_step * r_grad_original.detach() + np.sqrt(2 * ld_step) * noise
        log_r_new = log_reward(new_x)

        # Gradient at new_x
        r_grad_new = torch.autograd.grad(log_reward(new_x).sum(), new_x)[0]

        # Forward/backward proposal log-prob
        log_q_fwd = -(torch.norm(new_x - x_current - ld_step * r_grad_original, p=2, dim=1)**2) / (4 * ld_step)
        log_q_bck = -(torch.norm(x_current - new_x - ld_step * r_grad_new, p=2, dim=1)**2) / (4 * ld_step)

        # Acceptance log-prob
        log_accept = (log_r_new - log_r_current) + (log_q_bck - log_q_fwd)
        accept_mask = (torch.rand(N, device=device) < torch.exp(torch.clamp(log_accept, max=0)))

        # Update acceptance stats
        acceptance_count += accept_mask.sum().item()
        total_proposals += N

        # Accept or reject
        x_current = x_current.detach()
        x_current[accept_mask] = new_x.detach()[accept_mask]
        log_r_current[accept_mask] = log_r_new.detach()[accept_mask]

        # Store the chain states at step i
        trajectories[i] = x_current
        log_r_trajectories[i] = log_r_current

        # Recompute acceptance rate every 5 steps
        if (i + 1) % 5 == 0:
            acceptance_rate = acceptance_count / total_proposals
            acceptance_count = 0
            total_proposals = 0

    # Return the entire trajectory [T, N, D] and the corresponding log_r
    return trajectories, log_r_trajectories
