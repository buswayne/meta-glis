import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from dataset import FunctionDataset
from models import Autoencoder, Decoder
from benchmarks import rosenbrock_function_torch, rosenbrock_function_np
from glis.solvers import GLIS  # Assuming GLIS is a library or custom module you're using

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
    autoencoder_naive = Autoencoder(input_dim=input_dim, latent_dim=3)
    autoencoder_naive.to(device)
    autoencoder_naive.eval()
    # Extract the decoder from the Autoencoder
    decoder_naive = autoencoder_naive.decoder

    autoencoder = Autoencoder(input_dim=input_dim, latent_dim=latent_dim)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    autoencoder.load_state_dict(checkpoint['model'])
    autoencoder.to(device)
    autoencoder.eval()

    # Extract the decoder from the Autoencoder
    decoder = autoencoder.decoder

    # Initialize lists to store results
    true_optima = []
    pred_optima = []
    pred_optima_naive = []

    with torch.no_grad():
        for batch in test_loader:

            xopt_true = batch['xopt'].to(device)
            xopt_pred = autoencoder(xopt_true)
            xopt_pred_naive = autoencoder_naive(xopt_true)

            true_optima.append(xopt_true.cpu().numpy())
            pred_optima.append(xopt_pred.cpu().numpy())
            pred_optima_naive.append(xopt_pred_naive.cpu().numpy())

    # Concatenate all batches into single arrays
    true_optima = np.concatenate(true_optima, axis=0)
    pred_optima = np.concatenate(pred_optima, axis=0)
    pred_optima_naive = np.concatenate(pred_optima_naive, axis=0)

    # Print and compare the results
    plt.figure(figsize=(8, 6))
    plt.scatter(true_optima[::100, 0], true_optima[::100, 1], label='true', s=5)
    plt.scatter(pred_optima[::100, 0], pred_optima[::100, 1], label='pred', s=5)
    plt.scatter(pred_optima_naive[::100, 0], pred_optima_naive[::100, 1], label='pred naive', s=5)
    plt.legend()
    plt.show()

def main():
    """
    Main function for testing the model and evaluating optimization performance.
    """
    # Configuration
    input_dim = 2
    latent_dim = 1
    batch_size = 128
    test_data_path = "data/rosenbrock/test/rosenbrock_2d_optima_data.pt"
    model_path = "out/model_2d_checkpoint.pt"

    # Choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the test dataset
    test_dataset = FunctionDataset(test_data_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Evaluate the model and optimization approaches
    test_model(test_loader, model_path, input_dim, latent_dim, device)

if __name__ == "__main__":
    main()
