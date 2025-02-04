import numpy as np
import matplotlib.pyplot as plt
import torch
from sympy.physics.units import second
from torch.utils.data import DataLoader
from dataset import FunctionDataset, FunctionExplorationDataset
from models import Autoencoder, Decoder
from benchmarks import rosenbrock_function_torch, rosenbrock_function_np
from glis.solvers import GLIS  # Assuming GLIS is a library or custom module you're using

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams['axes.labelsize']=14
plt.rcParams['xtick.labelsize']=11
plt.rcParams['ytick.labelsize']=11
plt.rcParams['axes.grid']=True
plt.rcParams['axes.xmargin']=0

def test_model(test_loader, model_path, input_dim, latent_dim, device):
    """
    Load the trained model, evaluate on the test set, and compare global optima
    with optimization performed over `x` and `z`.

    :param test_loader: DataLoader for the test dataset
    :param model_path: Path to the saved model checkpoint
    :param input_dim: Dimensionality of the input data
    :param latent_dim: Dimensionality of the latent space
    :param device: Device (CPU or GPU)
    """
    # Load the trained Autoencoder model
    autoencoder_naive = Autoencoder(input_dim=input_dim, latent_dim=latent_dim)
    autoencoder_naive.to(device)
    autoencoder_naive.eval()

    # Apply scaling
    with torch.no_grad():
        for param in autoencoder_naive.parameters():
            param.data *= 2

    # Extract the decoder from the Autoencoder
    decoder_naive = autoencoder_naive.decoder

    autoencoder = Autoencoder(input_dim=input_dim, latent_dim=latent_dim)

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
    pred_optima_naive = []
    true_f_vals = []

    with torch.no_grad():
        for batch in test_loader:

            xopt_true = batch['Xopt'].to(device)
            fopt_true = batch['fopt'].to(device)
            xopt_pred = autoencoder(xopt_true)
            xopt_pred_naive = autoencoder_naive(xopt_true)

            true_optima.append(xopt_true.cpu().numpy())
            pred_optima.append(xopt_pred.cpu().numpy())
            pred_optima_naive.append(xopt_pred_naive.cpu().numpy())
            true_f_vals.append(fopt_true.cpu().numpy())

    # Concatenate all batches into single arrays
    true_optima = np.concatenate(true_optima, axis=0)
    pred_optima = np.concatenate(pred_optima, axis=0)
    pred_optima_naive = np.concatenate(pred_optima_naive, axis=0)
    true_f_vals = np.concatenate(true_f_vals, axis=0)

    # Find the index of the minimum value along the `n_points` dimension
    min_indices = np.argmin(true_f_vals, axis=1)  # Shape: (n_samples,)


    # Use the indices to extract the corresponding points from Xopt
    true_global_optima = []
    for i, idx in enumerate(min_indices):
        true_global_optima.append(true_optima[i,idx,:])

    true_global_optima = np.concatenate(true_global_optima, axis=0)

    rmse = np.sqrt(np.mean((true_optima - pred_optima) ** 2))
    rmse_naive = np.sqrt(np.mean((true_optima - pred_optima_naive) ** 2))

    # Print and compare the results

    idx_fixed = [(7, 11),(13,5),(7,5),(12,1),(12,3),(1,5)]

    for i in range(6):
        # first_idx, second_idx = np.random.choice(true_optima.shape[-1], size=2, replace=False)
        first_idx, second_idx = idx_fixed[i]

        plt.figure(figsize=(5, 5))
        plt.scatter(true_optima[:, :, first_idx], true_optima[:, :, second_idx], label='_nolegend_', s=5, alpha=0.05, color='tab:blue')
        plt.scatter(pred_optima[:, :, first_idx], pred_optima[:, :, second_idx], label='_nolegend_', s=5, alpha=0.05, color='tab:orange')
        plt.scatter(pred_optima_naive[:, :, first_idx], pred_optima_naive[:, :, second_idx], label='_nolegend_', s=5, alpha=0.05, color='tab:purple')
        plt.scatter(true_global_optima[:, first_idx], true_global_optima[:, second_idx], label='_nolegend_', marker='*', s=10, alpha=0.5, color='tab:red')

        # Add high-alpha points just for the legend
        plt.scatter([], [], label='true', s=20, alpha=1, color='tab:blue')
        plt.scatter([], [], label='pred', s=20, alpha=1, color='tab:orange')
        plt.scatter([], [], label='pred naive', s=20, alpha=1, color='tab:purple')
        plt.scatter([], [], label='true global', s=20, alpha=1, color='tab:red')

        plt.xlabel(f'$x_{{{first_idx}}}$')
        plt.ylabel(f'$x_{{{second_idx}}}$')
        plt.title(f'RMSE:, {rmse:.2f}, RMSE naive: {rmse_naive:.2f}')
        plt.legend()
        plt.axis('equal')
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.show()

def main():
    """
    Main function for testing the model and evaluating optimization performance.
    """
    # Configuration
    perturb_pct = 0.2
    input_dim = 15
    latent_dim = 5
    batch_size = 128
    test_data_path = f"data/rosenbrock/with_exploration/test/rosenbrock_{input_dim}d_{perturb_pct}pct_optima_data.pt"
    model_path = f"out/rosenbrock/with_exploration/model_{input_dim}d_{latent_dim}l_softmax_f.pt"

    # Choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the test dataset
    test_dataset = FunctionExplorationDataset(test_data_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Evaluate the model and optimization approaches
    test_model(test_loader, model_path, input_dim, latent_dim, device)

if __name__ == "__main__":
    main()
