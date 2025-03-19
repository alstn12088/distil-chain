import argparse
import torch
import os

import matplotlib.pyplot as plt
from tqdm import trange
import wandb

from plot_utils import *
from utils import set_seed, cal_subtb_coef_matrix, fig_to_image
from mcmc_teachers.mala import langevin_dynamics
from energies import *
from models.architectures import DistilledSampler

parser = argparse.ArgumentParser(description='GFN Linear Regression')
parser.add_argument('--lr_policy', type=float, default=1e-4)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--s_emb_dim', type=int, default=128)
parser.add_argument('--t_emb_dim', type=int, default=128)
parser.add_argument('--harmonics_dim', type=int, default=128)




################################################################
# Number of parallel chains
parser.add_argument('--batch_size', type=int, default=500)



################################################################
# Number of bootstrap samples for Model(x, t) + Langevin Dynamics K steps \approx Model(x, t+K)
parser.add_argument('--bootstrap_K', type=int, default=100)


################################################################


parser.add_argument('--epochs', type=int, default=25000)

parser.add_argument('--energy', type=str, default='9gmm',
                    choices=('9gmm', '25gmm', 'hard_funnel', 'easy_funnel', 'many_well'))

# For MALA
################################################################

# maximum iteration
parser.add_argument('--max_iter', type=int, default=2000)
parser.add_argument('--ld_step', type=float, default=0.001)


## ld step size scheduling
parser.add_argument('--ld_schedule', action='store_true', default=True)
parser.add_argument('--target_acceptance_rate', type=float, default=0.574)
################################################################



parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

set_seed(args.seed)
if 'SLURM_PROCID' in os.environ:
    args.seed += int(os.environ["SLURM_PROCID"])

eval_data_size = 2000
final_eval_data_size = 2000
plot_data_size = 2000
final_plot_data_size = 2000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def moment_loss(samples_a, samples_b, eps=1e-6):
    mean_a = samples_a.mean(dim=0)
    mean_b = samples_b.mean(dim=0)
    mean_loss = torch.sum((mean_a - mean_b) ** 2)

    a_centered = samples_a - mean_a
    b_centered = samples_b - mean_b
    cov_a = (a_centered.T @ a_centered) / (samples_a.shape[0] + eps)
    cov_b = (b_centered.T @ b_centered) / (samples_b.shape[0] + eps)
    cov_loss = torch.sum((cov_a - cov_b) ** 2)
    return mean_loss + cov_loss

def get_energy():
    if args.energy == '9gmm':
        energy = NineGaussianMixture(device=device)
    elif args.energy == '25gmm':
        energy = TwentyFiveGaussianMixture(device=device)
    elif args.energy == 'hard_funnel':
        energy = HardFunnel(device=device)
    elif args.energy == 'easy_funnel':
        energy = EasyFunnel(device=device)
    elif args.energy == 'many_well':
        energy = ManyWell(device=device)
    return energy

def plot_step(energy, name, samples):
    """
    Plot 'samples' on CPU. Also, if we generate ground-truth or reference
    samples from 'energy.sample', move them onto 'device' for any necessary
    computation, then move to CPU to pass to plotting.
    """
    samples_cpu = samples.detach().cpu()  # for plotting

    if args.energy == 'many_well':
        batch_size = plot_data_size
        # many_well plotting expects everything on CPU
        # Move 'samples' to CPU
        vizualizations = viz_many_well(energy, samples_cpu)
        (fig_samples_x13, ax_samples_x13, fig_kde_x13, ax_kde_x13, 
         fig_contour_x13, ax_contour_x13, fig_samples_x23, ax_samples_x23, 
         fig_kde_x23, ax_kde_x23, fig_contour_x23, ax_contour_x23) = vizualizations

        fig_samples_x13.savefig(f'{name}samplesx13.pdf', bbox_inches='tight')
        fig_samples_x23.savefig(f'{name}samplesx23.pdf', bbox_inches='tight')
        fig_kde_x13.savefig(f'{name}kdex13.pdf', bbox_inches='tight')
        fig_kde_x23.savefig(f'{name}kdex23.pdf', bbox_inches='tight')
        fig_contour_x13.savefig(f'{name}contourx13.pdf', bbox_inches='tight')
        fig_contour_x23.savefig(f'{name}contourx23.pdf', bbox_inches='tight')

        return {
            "visualization/contourx13": wandb.Image(fig_to_image(fig_contour_x13)),
            "visualization/contourx23": wandb.Image(fig_to_image(fig_contour_x23)),
            "visualization/kdex13": wandb.Image(fig_to_image(fig_kde_x13)),
            "visualization/kdex23": wandb.Image(fig_to_image(fig_kde_x23)),
            "visualization/samplesx13": wandb.Image(fig_to_image(fig_samples_x13)),
            "visualization/samplesx23": wandb.Image(fig_to_image(fig_samples_x23))
        }

    elif energy.data_ndim != 2:
        return {}

    else:
        batch_size = plot_data_size
        # ground-truth samples from energy, on 'device'
        gt_samples = energy.sample(batch_size).to(device)
        gt_samples_cpu = gt_samples.detach().cpu()

        fig_contour, ax_contour = get_figure(bounds=(-13., 13.))
        fig_kde, ax_kde = get_figure(bounds=(-13., 13.))
        fig_kde_overlay, ax_kde_overlay = get_figure(bounds=(-13., 13.))

        # Plot using CPU arrays
        plot_contours(energy.log_reward, ax=ax_contour, bounds=(-13., 13.), 
                      n_contour_levels=150, device=device)
        plot_kde(gt_samples_cpu, ax=ax_kde_overlay, bounds=(-13., 13.))
        plot_kde(samples_cpu, ax=ax_kde, bounds=(-13., 13.))
        plot_samples(samples_cpu, ax=ax_contour, bounds=(-13., 13.))
        plot_samples(samples_cpu, ax=ax_kde_overlay, bounds=(-13., 13.))

        fig_contour.savefig(f'{name}contour.pdf', bbox_inches='tight')
        fig_kde_overlay.savefig(f'{name}kde_overlay.pdf', bbox_inches='tight')
        fig_kde.savefig(f'{name}kde.pdf', bbox_inches='tight')

        return {
            "visualization/contour": wandb.Image(fig_to_image(fig_contour)),
            "visualization/kde_overlay": wandb.Image(fig_to_image(fig_kde_overlay)),
            "visualization/kde": wandb.Image(fig_to_image(fig_kde))
        }

def train():
    name = "test"
    os.makedirs(name, exist_ok=True)

    set_seed(args.seed)
    energy = get_energy()

    wandb.init(project="distil-sampler", config=args.__dict__)

    # 1) Initialize DistilledSampler
    distil_model = DistilledSampler(
        s_dim=energy.data_ndim,
        harmonics_dim=args.harmonics_dim,
        hidden_dim=args.hidden_dim
    ).to(device)

    optim_policy = torch.optim.Adam(distil_model.parameters(), lr=args.lr_policy)


    T = args.max_iter
    for i in trange(args.epochs):
        optim_policy.zero_grad()

        x0_samples = torch.randn(args.batch_size, energy.data_ndim, device=device) * 3

        t_int = torch.randint(0, T - args.bootstrap_K - 1, (1,)).item()
        
        # 0 < t < 1
        t_batch = torch.ones(args.batch_size, 1, device=device) * t_int / T


        x_pred = distil_model(x0_samples, t_batch)
        x_target, _ = langevin_dynamics(x_pred, energy.log_reward, device, args.bootstrap_K, args)
        x_target = x_target[-1].squeeze(0)

        x_bootstrap = distil_model(x0_samples, t_batch + args.bootstrap_K/T)

        bootstrap_loss = moment_loss(x_bootstrap, x_target.detach())

        bootstrap_loss.backward()
        optim_policy.step()

        if i % 100 == 0:
            wandb.log({"step": i, "moment_loss": bootstrap_loss.item()})

            distil_model.eval()
            with torch.no_grad():
                # Suppose your model has a .sample method returning [N, D]
                samples = distil_model.sample(2000).to(device)

            images = plot_step(energy, f"{name}/step_{i}_", samples)
            if images is not None:
                wandb.log(images, step=i)
            distil_model.train()

def eval():
    pass

if __name__ == '__main__':
    train()
