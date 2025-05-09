import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from sympy.physics.vector.tests.test_printing import alpha
from torch.utils.data import DataLoader
sys.path.append('..')
from dataset import FunctionExplorationDataset
from models import Autoencoder

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams['axes.labelsize']=14
plt.rcParams['xtick.labelsize']=11
plt.rcParams['ytick.labelsize']=11
plt.rcParams['axes.grid']=True
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['axes.xmargin']=0
plt.rcParams['legend.fontsize'] = 14

def test_model(test_loader, model_path, input_dim, latent_dim, x_bounds, device):
    """
    Load the trained model, evaluate on the test set, and compare global optima
    with optimization performed over `x` and `z`.

    :param test_loader: DataLoader for the test dataset
    :param model_path: Path to the saved model checkpoint
    :param input_dim: Dimensionality of the input data
    :param latent_dim: Dimensionality of the latent space
    :param device: Device (CPU or GPU)
    """
    autoencoder = Autoencoder(input_dim=input_dim, latent_dim=latent_dim, lat_lb=0.0, lat_ub=1.0, out_bounds=x_bounds)

    try:
        checkpoint = torch.load(model_path, weights_only=True)
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        autoencoder.load_state_dict(state_dict)
    except:
        print("Could not load pretrained model")
        try:
            state_dict = torch.load(model_path, weights_only=True)
            autoencoder.load_state_dict(state_dict)
        except:
            raise Exception("Could not load pretrained model")

    autoencoder.to(device)
    autoencoder.eval()

    # Initialize lists to store results
    true_optima = []
    pred_optima = []
    true_f_vals = []

    with torch.no_grad():
        for batch in test_loader:

            xopt_true = batch['Xopt'].to(device)
            fopt_true = batch['fopt'].to(device)
            xopt_pred = autoencoder(xopt_true)

            true_optima.append(xopt_true.cpu().numpy())
            pred_optima.append(xopt_pred.cpu().numpy())
            true_f_vals.append(fopt_true.cpu().numpy())

    # Concatenate all batches into single arrays
    true_optima = np.concatenate(true_optima, axis=0)
    pred_optima = np.concatenate(pred_optima, axis=0)
    true_f_vals = np.concatenate(true_f_vals, axis=0)

    # Use the indices to extract the corresponding points from Xopt
    true_global_optima = []
    for i in range(true_optima.shape[0]):
        best_idx = np.argmin(true_f_vals[i,:,0])
        true_global_optima.append(true_optima[i,best_idx,:].reshape(1,-1))

    true_global_optima = np.concatenate(true_global_optima, axis=0)

    rmse = np.sqrt(np.mean((true_optima - pred_optima) ** 2))

    # Print and compare the results

    # idx_fixed = [(0, 1), (2, 12), (4, 12), (15, 16)]  # Only 4 pairs
    idx_fixed = [(0, 1), (2, 12), (15, 16)]

    fig, axes = plt.subplots(1, 3, figsize=(6, 2), sharex=True, sharey=True)
    axes = axes.flatten()  # Flatten for easy iteration

    for i, ax in enumerate(axes):
        first_idx, second_idx = idx_fixed[i]

        ax.scatter(true_optima[:, :, first_idx], true_optima[:, :, second_idx], label='_nolegend_', s=2, alpha=0.05,
                   color='tab:blue', rasterized=True)
        ax.scatter(pred_optima[:, :, first_idx], pred_optima[:, :, second_idx], label='_nolegend_', s=2, alpha=0.05,
                   color='tab:orange', rasterized=True)
        ax.scatter(true_global_optima[:, first_idx], true_global_optima[:, second_idx], label='_nolegend_', marker='*',
                   s=10, alpha=0.5, color='limegreen', rasterized=True)

        ax.set_xlabel(f'$x_{{{first_idx + 1}}}$')
        ax.set_ylabel(f'$x_{{{second_idx + 1}}}$')
        ax.set_xlim([-2.5, 2.5])
        ax.set_ylim([-2.5, 2.5])
        # Force display of -1 and 1 on the y-axis
        ax.set_xticks([-1, 1])
        ax.set_yticks([-1, 1])
        ax.set_aspect('equal')

    # Create a single legend for the entire figure
    handles = [
        plt.Line2D([], [], marker='o', linestyle='None', markersize=2, color='tab:blue', label='$x$'),
        plt.Line2D([], [], marker='o', linestyle='None', markersize=2, color='tab:orange',
                   label='$\\mathcal{A}\\mathcal{E}(x)$'),
        plt.Line2D([], [], marker='*', linestyle='None', markersize=4, color='limegreen', label='$x^\\star$'),
    ]

    # fig.legend(handles=handles, loc='upper center', ncol=3, bbox_to_anchor=(0.53, 1.08))
    # plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust layout to accommodate the legend
    plt.tight_layout()
    plt.savefig('rosenbrock_autoencoder_1x3.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function for testing the model and evaluating optimization performance.
    """
    # Configuration
    perturb_pct = 1000
    input_dim = 20
    latent_dim = 3
    x_bounds = [(-2.5, 2.5)] * input_dim
    alpha = .9
    batch_size = 128
    test_data_path = f"../../data/rosenbrock/with_exploration/test/rosenbrock_{input_dim}d_{perturb_pct}pct_optima_data.pt"
    model_path = f"../../out/rosenbrock/with_exploration/model_{input_dim}d_{latent_dim}l_alpha_{alpha:.2f}.pt"

    # Choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the test dataset
    test_dataset = FunctionExplorationDataset(test_data_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Evaluate the model and optimization approaches
    test_model(test_loader, model_path, input_dim, latent_dim, x_bounds, device)

if __name__ == "__main__":
    main()
