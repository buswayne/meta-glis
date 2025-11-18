import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from numerical_examples.robot.dataset import RobotDataset
from numerical_examples.robot.models import Autoencoder
from numerical_examples.robot.plot_utils import plot_outs


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

    x_bounds = (
            [(0, 500)] * 7 +
            [(0, 200)] * 7 +
            [(0, 15000)] * 7
    )

    # Load the trained Autoencoder model
    autoencoder = Autoencoder(input_dim=input_dim, latent_dim=latent_dim, out_bounds=x_bounds)

    try:
        checkpoint = torch.load(model_path, weights_only=True)
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        autoencoder.load_state_dict(state_dict)
    except:
        "Could not load pretrained model"
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
    J = []

    with torch.no_grad():
        for batch in test_loader:

            xopt_true = batch['Xopt'].to(device)
            fopt = batch['fopt'].to(device)
            xopt_pred = autoencoder(xopt_true)

            true_optima.append(xopt_true.cpu().numpy())
            pred_optima.append(xopt_pred.cpu().numpy())
            J.append(fopt.cpu().numpy())

    # Concatenate all batches into single arrays
    true_optima = np.concatenate(true_optima, axis=0)
    pred_optima = np.concatenate(pred_optima, axis=0)
    J = np.concatenate(J, axis=0)

    # Find index of maximum J for each sample along iterations
    best_idx = np.argmax(J, axis=1).squeeze()  # shape (N,)

    # Extract the best objective values
    best_J = J[np.arange(J.shape[0]), best_idx, 0]  # shape (N,)
    best_true_optima = true_optima[np.arange(true_optima.shape[0]), best_idx, :]  # shape (N, 21)
    best_pred_optima = pred_optima[np.arange(pred_optima.shape[0]), best_idx, :]  # shape (N, 21)

    rmse = np.sqrt(np.mean((true_optima - pred_optima) ** 2))

    # # Print and compare the results
    plot_outs(best_true_optima, best_J)
    plot_outs(best_pred_optima, best_J)

def main():
    """
    Main function for testing the model and evaluating optimization performance.
    """
    # Configuration
    input_dim = 21
    latent_dim = 10
    alpha = 0
    batch_size = 128

    home_dir = os.path.expanduser('~')
    model_dir = 'meta-glis/numerical_examples/robot/out/robot'
    model_path = os.path.join(home_dir, model_dir, f'model_{input_dim}d_{latent_dim}l_300iter_exp_decay_alpha_{alpha:02}.pt')

    # Choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the test dataset
    test_dataset = RobotDataset('test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Evaluate the model and optimization approaches
    test_model(test_loader, model_path, input_dim, latent_dim, device)

if __name__ == "__main__":
    main()
