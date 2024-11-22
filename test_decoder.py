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
    glis_x_optima = []
    glis_z_optima_naive = []
    glis_z_optima = []

    with torch.no_grad():
        for batch in test_loader:
            xopt = batch['xopt'].to(device)
            A = batch['params']['A'].to(device)
            B = batch['params']['B'].to(device)
            shifts = batch['params']['shifts'].to(device)

            # Create the Rosenbrock function
            max_evals = 100

            # Compare three approaches
            for i in range(xopt.size(0)):  # Iterate over batch

                fun = rosenbrock_function_np(A[i].cpu().numpy(),
                                             B[i].cpu().numpy(),
                                             shifts[i].cpu().numpy(),
                                             xopt.shape[1])

                # True global optimum from dataset
                true_optimum = fun(xopt[i].cpu().numpy()).item()
                true_optima.append(true_optimum)

                # GLIS optimization directly over x
                glis_x = GLIS(bounds=(-10*np.ones(xopt.shape[1]), 10*np.ones(xopt.shape[1])), delta=0.1)
                x_glis_opt, _ = glis_x.solve(fun, max_evals)
                glis_x_optima.append(glis_x.fbest_seq)

                # GLIS optimization in latent space (z)
                fun_z = lambda z: fun(decoder(torch.tensor(z, dtype=torch.float32, device=device)).cpu().numpy())
                glis_z = GLIS(bounds=(-10*np.ones(latent_dim), 10*np.ones(latent_dim)), delta=0.1)
                z_glis_opt, _ = glis_z.solve(fun_z, max_evals)
                glis_z_optima.append(glis_z.fbest_seq)

                # NAIVE GLIS optimization in latent space (z)
                fun_z_naive = lambda z: fun(decoder_naive(torch.tensor(z, dtype=torch.float32, device=device)).cpu().numpy())
                glis_z_naive = GLIS(bounds=(-10*np.ones(3), 10*np.ones(3)), delta=0.1)
                z_glis_opt_naive, _ = glis_z_naive.solve(fun_z_naive, max_evals)
                glis_z_optima_naive.append(glis_z_naive.fbest_seq)

            break

    # Print and compare the results
    plt.figure(figsize=(8, 6))

    # True optima (constant across iterations)
    plt.plot(np.arange(max_evals), np.repeat(np.reshape(true_optima, (-1,1)), max_evals, axis=1).T, color='blue', label="True Optimum (Mean)", alpha=0.3)

    # GLIS over x
    plt.plot(np.arange(max_evals), np.array(glis_x_optima).T, label="GLIS over x", color='green', alpha=0.3)

    # GLIS over z
    plt.plot(np.arange(max_evals), np.array(glis_z_optima).T, label="GLIS over z", color='orange', alpha=0.3)

    # GLIS over z naive
    plt.plot(np.arange(max_evals), np.array(glis_z_optima_naive).T, label="GLIS over z naive", color='purple', alpha=0.3)

    plt.yscale('log')  # Log scale for y-axis
    plt.xlabel("Iterations")
    plt.ylabel("Function Value (log scale)")
    plt.legend()
    plt.grid()
    plt.show()

    # Mean - Std plot
    true_mean = np.mean(true_optima)
    true_std = np.std(true_optima)

    glis_x_means = np.mean(glis_x_optima, axis=0)
    glis_x_stds = np.std(glis_x_optima, axis=0)

    glis_z_means = np.mean(glis_z_optima, axis=0)
    glis_z_stds = np.std(glis_z_optima, axis=0)

    plt.figure(figsize=(8, 6))

    # True optima (constant across iterations)
    plt.hlines(true_mean, 0, max_evals - 1, colors='blue', linestyles='--', label="True Optimum (Mean)")
    plt.fill_between(
        np.arange(max_evals),
        true_mean - true_std,
        true_mean + true_std,
        color='blue',
        alpha=0.2,
        label="True Optimum (Std)"
    )

    # GLIS over x
    plt.plot(np.arange(max_evals), glis_x_means, label="GLIS over x (Mean)", color='green')
    plt.fill_between(
        np.arange(max_evals),
        glis_x_means - glis_x_stds,
        glis_x_means + glis_x_stds,
        color='green',
        alpha=0.2,
        label="GLIS over x (Std)"
    )

    # GLIS over z
    plt.plot(np.arange(max_evals), glis_z_means, label="GLIS over z (Mean)", color='orange')
    plt.fill_between(
        np.arange(max_evals),
        glis_z_means - glis_z_stds,
        glis_z_means + glis_z_stds,
        color='orange',
        alpha=0.2,
        label="GLIS over z (Std)"
    )

    plt.yscale('log')  # Log scale for y-axis
    plt.title("Mean and Standard Deviation of Optimization Results")
    plt.xlabel("Iterations")
    plt.ylabel("Function Value (log scale)")
    plt.legend()
    plt.grid()
    plt.show()

def main():
    """
    Main function for testing the model and evaluating optimization performance.
    """
    # Configuration
    input_dim = 2
    latent_dim = 1
    batch_size = 50
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
