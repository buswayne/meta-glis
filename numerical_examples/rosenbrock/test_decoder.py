import sys, os
import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
from torch.utils.data import DataLoader
sys.path.append('..')
from dataset import FunctionDataset
from models import Autoencoder, Decoder
from benchmarks import rosenbrock_function_np
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
    # Load the trained Autoencoder model
    autoencoder = Autoencoder(input_dim=input_dim, latent_dim=latent_dim, lat_lb=0.0, lat_ub=1.0, out_bounds=x_bounds)

    # Scale the weights by 2
    # for param in autoencoder.parameters():
    #     if param.requires_grad:
    #         param.data *= 2
    try:
        checkpoint = torch.load(model_path, weights_only=True)
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        autoencoder.load_state_dict(state_dict)
    except:
        raise Exception("Could not load pretrained model")
    autoencoder.to(device)
    autoencoder.eval()

    # Extract the decoder from the Autoencoder
    decoder = autoencoder.decoder

    # Initialize dictionary to store all optimization results
    results_dict = {
        'true_points': [],
        'true_optima': [],
        'glis_x_points': [],
        'glis_x_values': [],
        'glis_z_points': [],
        'glis_z_values': [],
        'glis_z_decoded_points': [],
    }

    with torch.no_grad():
        for batch in test_loader:
            xopt = batch['Xopt'].to(device)
            fopt = batch['fopt'].to(device)

            A = batch['params']['A'].to(device)
            B = batch['params']['B'].to(device)
            shifts = batch['params']['shifts'].to(device)

            # Create the Rosenbrock function
            max_evals = 100

            TIME_x = []
            TIME_z = []

            # Compare three approaches
            for i in range(xopt.size(0)):  # Iterate over batch

                fun = rosenbrock_function_np(A[i].cpu().numpy(),
                                             B[i].cpu().numpy(),
                                             shifts[i].cpu().numpy(),
                                             xopt.shape[1])

                # True global optimum from dataset
                true_point = xopt[i].cpu().numpy()
                true_optimum = fun(true_point).item()
                results_dict['true_points'].append(true_point)
                results_dict['true_optima'].append(true_optimum)

                # GLIS optimization directly over x
                glis_x = GLIS(bounds=(-2.5*np.ones(xopt.shape[1]), 2.5*np.ones(xopt.shape[1])), delta=0.1, obj_transform=lambda f: np.log(f+1))
                glis_x.solve(fun, max_evals)
                total_time = np.array(glis_x.time_fun_eval)
                TIME_x.append(np.mean(total_time[41:]))
                results_dict['glis_x_points'].append(glis_x.X)  # Store the points explored
                results_dict['glis_x_values'].append(glis_x.F)  # Store the values

                # NAIVE GLIS optimization in latent space (z)
                fun_z = lambda z: fun(decoder(torch.tensor(z, dtype=torch.float32, device=device)).cpu().numpy())
                glis_z = GLIS(bounds=(0*np.ones(latent_dim), 1*np.ones(latent_dim)), delta=0.1, obj_transform=lambda f: np.log(f+1))
                glis_z.solve(fun_z, max_evals)
                results_dict['glis_z_points'].append(glis_z.X)  # Store the points explored
                results_dict['glis_z_values'].append(glis_z.F)  # Store the values
                total_time = np.array(glis_z.time_fun_eval)
                TIME_z.append(np.mean(total_time[41:]))

                # Save the sequence of reconstructed x_position from latent space z
                decoded_x = decoder(torch.tensor(glis_z.X, dtype=torch.float32, device=device)).cpu().numpy()
                results_dict['glis_z_decoded_points'].append(decoded_x)

            break

    print('Time x', np.mean(TIME_x))
    print('Time z', np.mean(TIME_z))
    # Save results to a pickle file
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(results_dir, f'rosenbrock_{input_dim}d_{latent_dim}l_v2.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results_dict, f)

def main():
    """
    Main function for testing the model and evaluating optimization performance.
    """
    # Configuration
    perturb_pct = 1000
    input_dim = 20
    latent_dim = 3
    x_bounds = [(-2.5, 2.5)] * input_dim
    batch_size = 100
    test_data_path = f"../../data/rosenbrock/with_exploration/test/rosenbrock_{input_dim}d_{perturb_pct}pct_optima_data.pt"
    model_path = f"../../out/rosenbrock/with_exploration/model_{input_dim}d_{latent_dim}l_alpha_0.90.pt"

    # Choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the test dataset
    test_dataset = FunctionDataset(test_data_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Evaluate the model and optimization approaches
    test_model(test_loader, model_path, input_dim, latent_dim, x_bounds, device)

if __name__ == "__main__":
    main()
