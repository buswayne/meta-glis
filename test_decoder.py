import numpy as np
import matplotlib.pyplot as plt
import torch
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

def test_model(test_loader, model_path_1, model_path_2, input_dim, latent_dim, device):
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
    autoencoder_1 = Autoencoder(input_dim=input_dim, latent_dim=latent_dim)
    try:
        checkpoint = torch.load(model_path_1, weights_only=True)
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        autoencoder_1.load_state_dict(state_dict)
    except:
        raise Exception("Could not load pretrained model")
    autoencoder_1.to(device)
    autoencoder_1.eval()

    # Apply scaling
    # with torch.no_grad():
    #     for param in autoencoder_1.parameters():
    #         param.data *= 2

    # Extract the decoder from the Autoencoder
    decoder_1 = autoencoder_1.decoder

    # Optimized over X
    # autoencoder_2 = Autoencoder(input_dim=input_dim, latent_dim=latent_dim)
    # try:
    #     checkpoint = torch.load(model_path_2, weights_only=True)
    #     state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
    #     autoencoder_2.load_state_dict(state_dict)
    # except:
    #     raise Exception("Could not load pretrained model")
    #
    # autoencoder_2.to(device)
    # autoencoder_2.eval()
    #
    # # Extract the decoder from the Autoencoder
    # decoder_2 = autoencoder_2.decoder

    decoder_1 = Decoder(latent_dim=latent_dim, output_dim=input_dim)
    decoder_1.to(device)
    decoder_1.eval()

    decoder_2 = Decoder(latent_dim=latent_dim, output_dim=input_dim)

    try:
        checkpoint = torch.load(model_path_2, weights_only=True)
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        decoder_2.load_state_dict(state_dict)
    except:
        raise Exception("Could not load pretrained model")

    decoder_2.to(device)
    decoder_2.eval()

    # Initialize lists to store results
    true_optima = []
    glis_x_optima = []
    glis_z_optima_1 = []
    glis_z_optima_2 = []

    with torch.no_grad():
        for batch in test_loader:
            xopt = batch['Xopt'].to(device)
            fopt = batch['fopt'].to(device)

            A = batch['params']['A'].to(device)
            B = batch['params']['B'].to(device)
            shifts = batch['params']['shifts'].to(device)

            # Create the Rosenbrock function
            max_evals = 200

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
                fun_z_2 = lambda z: fun(decoder_2(torch.tensor(z, dtype=torch.float32, device=device)).cpu().numpy())
                glis_z_2 = GLIS(bounds=(-10*np.ones(latent_dim), 10*np.ones(latent_dim)), delta=0.1)
                z_glis_opt_2, _ = glis_z_2.solve(fun_z_2, max_evals)
                glis_z_optima_2.append(glis_z_2.fbest_seq)

                # NAIVE GLIS optimization in latent space (z)
                fun_z_1 = lambda z: fun(decoder_1(torch.tensor(z, dtype=torch.float32, device=device)).cpu().numpy())
                glis_z_1 = GLIS(bounds=(-10*np.ones(latent_dim), 10*np.ones(latent_dim)), delta=0.1)
                z_glis_opt_1, _ = glis_z_1.solve(fun_z_1, max_evals)
                glis_z_optima_1.append(glis_z_1.fbest_seq)

            break

    # Print and compare the results
    plt.figure(figsize=(9, 6))

    # True optima (constant across iterations)
    true_optima_arr =  np.repeat(np.reshape(true_optima, (-1,1)), max_evals, axis=1).T
    plt.plot(np.arange(max_evals), true_optima_arr[:,0], color='tab:blue', label="True Optimum (Mean)", alpha=0.3)
    plt.plot(np.arange(max_evals), true_optima_arr[:,1:], color='tab:blue', label='', alpha=0.3)

    # # GLIS over x
    plt.plot(np.arange(max_evals), np.array(glis_x_optima)[0,:], label="GLIS over x", color='tab:green', alpha=0.3)
    plt.plot(np.arange(max_evals), np.array(glis_x_optima)[1:,:].T, label='', color='tab:green', alpha=0.3)
    #
    # # GLIS over z
    plt.plot(np.arange(max_evals), np.array(glis_z_optima_2)[0,:], label="GLIS over z ($a$)", color='tab:orange', alpha=0.3)
    plt.plot(np.arange(max_evals), np.array(glis_z_optima_2)[1:,:].T, label='', color='tab:orange', alpha=0.3)
    #
    # # GLIS over z naive
    plt.plot(np.arange(max_evals), np.array(glis_z_optima_1)[0,:], label="GLIS over z ($b$)", color='tab:purple', alpha=0.3)
    plt.plot(np.arange(max_evals), np.array(glis_z_optima_1)[1:,:].T, label='', color='tab:purple', alpha=0.3)

    plt.yscale('log')  # Log scale for y-axis
    plt.xlabel("Iterations")
    plt.ylabel("Function Value (log scale)")
    plt.legend()
    plt.show()

    # Mean - Std plot
    true_mean = np.mean(true_optima)
    true_std = np.std(true_optima)

    glis_x_means = np.mean(glis_x_optima, axis=0)
    glis_x_stds = np.std(glis_x_optima, axis=0)

    glis_z_2_means = np.mean(glis_z_optima_2, axis=0)
    glis_z_2_stds = np.std(glis_z_optima_2, axis=0)

    glis_z_1_means = np.mean(glis_z_optima_1, axis=0)
    glis_z_1_stds = np.std(glis_z_optima_1, axis=0)

    plt.figure(figsize=(8, 6))

    # True optima (constant across iterations)
    plt.hlines(true_mean, 0, max_evals - 1, colors='tab:blue', linestyles='--', label="True Optimum")
    plt.fill_between(
        np.arange(max_evals),
        true_mean - true_std,
        true_mean + true_std,
        color='tab:blue',
        alpha=0.2,
        label="_no_legend_"
    )

    # GLIS over x
    plt.plot(np.arange(max_evals), glis_x_means, label="GLIS over x", color='tab:green')
    plt.fill_between(
        np.arange(max_evals),
        glis_x_means - glis_x_stds,
        glis_x_means + glis_x_stds,
        color='tab:green',
        alpha=0.2,
        label="_no_legend_"
    )

    # GLIS over z
    plt.plot(np.arange(max_evals), glis_z_2_means, label="GLIS over z ($a$)", color='tab:orange')
    plt.fill_between(
        np.arange(max_evals),
        glis_z_2_means - glis_z_2_stds,
        glis_z_2_means + glis_z_2_stds,
        color='tab:orange',
        alpha=0.2,
        label="_no_legend_"
    )

    # GLIS over z
    plt.plot(np.arange(max_evals), glis_z_1_means, label="GLIS over z ($b$)", color='tab:purple')
    plt.fill_between(
        np.arange(max_evals),
        glis_z_1_means - glis_z_1_stds,
        glis_z_1_means + glis_z_1_stds,
        color='tab:purple',
        alpha=0.2,
        label="_no_legend_"
    )

    plt.yscale('log')  # Log scale for y-axis
    plt.title("Mean and Standard Deviation of Optimization Results")
    plt.xlabel("Iterations")
    plt.ylabel("Function Value (log scale)")
    plt.legend()
    plt.show()

def main():
    """
    Main function for testing the model and evaluating optimization performance.
    """
    # Configuration
    perturb_pct = 0.2
    input_dim = 15
    latent_dim = 5
    batch_size = 10
    test_data_path = f"data/rosenbrock/with_exploration/valid/rosenbrock_{input_dim}d_{perturb_pct}pct_optima_data.pt"
    model_path_1 = f"out/rosenbrock/with_exploration/model_{input_dim}d_{latent_dim}l_softmax_X.pt"
    model_path_2 = f"out/rosenbrock/with_exploration/model_decoder_only_{input_dim}d_{latent_dim}l.pt"

    # Choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the test dataset
    test_dataset = FunctionDataset(test_data_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Evaluate the model and optimization approaches
    test_model(test_loader, model_path_1, model_path_2, input_dim, latent_dim, device)

if __name__ == "__main__":
    main()
