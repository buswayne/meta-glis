import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from dataset_mpc import FunctionExplorationDataset
import os, sys
sys.path.append('..')
from models import Autoencoder, AutoencoderTrivial
# from models_complex import Autoencoder


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

dict_x0 = {
    'Q_{\Delta u}': 0.1,
    'q_{y_{11}}': 0.1,
    'q_{y_{22}}': 0.9,
    'N_{p}': 40,
    '\epsilon_{c}': 1.0,
    'T_{s}^{MPC}': 25e-3,
    r'{\text{QP}_{{\epsilon}_{abs}}}': -3,
    r'{\text{QP}_{{\epsilon}_{rel}}}': -2,
    # r'\log{\text{QP}_{{\epsilon}_{abs}}}': -3,
    # r'\log{\text{QP}_{{\epsilon}_{rel}}}': -2,
    'W_{w_{11}}': 0.1,
    'W_{w_{22}}': 0.4,
    'W_{w_{33}}': 0.1,
    'W_{w_{44}}': 0.4,
    'W_{v_{11}}': 0.5,
    'W_{v_{22}}': 0.5
}

var_names = list(dict_x0.keys())

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

    x_bounds = [
        (1e-16, 1),  # 0
        (1e-16, 1),  # 1
        (1e-16, 1),  # 2
        (5, 300),  # 3
        (0.3, 1),  # 4
        (1e-3, 50e-3),  # 5
        (-7, -1),  # 6
        (-7, -1),  # 7
        (1e-16, 1),  # 8
        (1e-16, 1),  # 9
        (1e-16, 1),  # 10
        (1e-16, 1),  # 11
        (1e-16, 1),  # 12
        (1e-16, 1),  # 13
    ]

    # Load the trained Autoencoder model
    autoencoder_naive = Autoencoder(input_dim=input_dim, latent_dim=3)
    autoencoder_naive.to(device)
    autoencoder_naive.eval()
    # Extract the decoder from the Autoencoder
    # decoder_naive = autoencoder_naive.decoder

    autoencoder = AutoencoderTrivial(input_dim=input_dim, latent_dim=latent_dim, out_bounds=x_bounds)

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
    true_f_vals = []
    pred_optima_naive = []

    with torch.no_grad():
        for batch in test_loader:

            xopt_true = batch['Xopt'].to(device)
            fopt_true = batch['fopt'].to(device)
            xopt_pred = autoencoder(xopt_true)
            xopt_pred_naive = autoencoder_naive(xopt_true)

            true_optima.append(xopt_true.cpu().numpy())
            true_f_vals.append(fopt_true.cpu().numpy())
            pred_optima.append(xopt_pred.cpu().numpy())
            pred_optima_naive.append(xopt_pred_naive.cpu().numpy())

    # Concatenate all batches into single arrays
    true_optima = np.concatenate(true_optima, axis=0)
    pred_optima = np.concatenate(pred_optima, axis=0)
    pred_optima_naive = np.concatenate(pred_optima_naive, axis=0)
    true_f_vals = np.concatenate(true_f_vals, axis=0)


    # Use the indices to extract the corresponding points from Xopt
    true_global_optima = []

    for i in range(true_optima.shape[0]):
        best_idx = np.argmin(true_f_vals[i,:,0])
        true_global_optima.append(true_optima[i,best_idx,:].reshape(1,-1))

    true_global_optima = np.concatenate(true_global_optima, axis=0)

    rmse = np.sqrt(np.mean((true_optima - pred_optima) ** 2))
    rmse_naive = np.sqrt(np.mean((true_optima - pred_optima_naive) ** 2))

    print(rmse)

    # dim_pairs = [tuple(np.random.choice(true_optima.shape[-1], size=2, replace=False)) for _ in range(16)]
    dim_pairs = [(9, 6),
                 (12, 6),
                 (9, 2),
                 (1, 8),
                 (5, 8),
                 (13, 5),
                 (8, 9),
                 (9, 4),
                 (1, 11),
                 (3, 1)]

    # Define subplot grid (2 rows, 5 columns)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 5, figsize=(9, 3))
    axes = axes.flatten()  # Flatten for easier iteration

    # Loop through each selected pair and plot
    for ax, (first_idx, second_idx) in zip(axes, dim_pairs):
        ax.scatter(
            true_optima[:, :, first_idx], true_optima[:, :, second_idx],
            s=.5, alpha=0.05, color='tab:blue', rasterized=True
        )
        ax.scatter(
            pred_optima[:, :, first_idx], pred_optima[:, :, second_idx],
            s=.5, alpha=0.05, color='tab:orange', rasterized=True
        )
        ax.scatter(
            true_global_optima[:, first_idx], true_global_optima[:, second_idx],
            s=5, alpha=1, color='green', marker='*', rasterized=True
        )
        ax.set_xlabel(f'${var_names[first_idx]}$', fontsize=14)
        ax.set_ylabel(f'${var_names[second_idx]}$', fontsize=14)

    # Adjust layout
    fig.align_ylabels(axes)

    # Adjust layout and add a title
    # fig.subplots_adjust(wspace=0.5, hspace=0.7)  # Reduce horizontal and vertical
    #
    plt.tight_layout()
    # plt.savefig('mpc_manifold.pdf', dpi=300)
    plt.show()

    # Extract weights from the model
    import seaborn as sns
    # Extract decoder weights

    # Access the single Linear layer inside the Sequential block
    decoder_layer = autoencoder.decoder.decoder[0]  # Access the Linear layer inside .decoder# Correct way to get the Linear layer

    # Extract decoder weights
    decoder_weight = decoder_layer.weight.detach().cpu().numpy()  # (14, 3)
    decoder_bias = decoder_layer.bias.detach().cpu().numpy()  # Access the Linear layer inside .decoder# Correct way to get the Linear layer

    # Example: Normalize each row of decoder_weight
    print('Decoder:', decoder_weight.shape)
    print('Norma', np.linalg.norm(decoder_weight, axis=0, keepdims=True).shape)
    stacked = np.hstack((decoder_weight, decoder_bias[:, np.newaxis]))

    decoder_weight_normalized = decoder_weight / np.linalg.norm(decoder_weight, axis=0, keepdims=True)

    print('Sum:', decoder_weight_normalized.sum(axis=0))  # Should print an array of ones

    # Plotting the normalized matrix
    plt.figure(figsize=(6, 1.5))
    im = plt.imshow(np.abs(decoder_weight_normalized.T), cmap='viridis', aspect='auto')
    plt.colorbar(im, label=None)
    plt.yticks(ticks=[0, 1, 2], labels=['$z_1$', '$z_2$', '$z_3$'])
    plt.xticks(ticks=np.arange(len(var_names)), labels=[f"${name}$" for name in var_names], rotation=90)
    plt.grid(False)
    plt.tight_layout(w_pad=0, h_pad=0)
    plt.subplots_adjust(right=1.05)  # Ensures there is no space on the right
    plt.savefig('mpc_normed_weights.pdf', dpi=300)
    plt.show()

    # Perform QR decomposition
    Q, R = np.linalg.qr(decoder_weight)

    # Print the orthogonal matrix Q and upper triangular matrix R
    print("Orthogonal matrix Q (rows form orthogonal basis):\n", Q)
    print("\nUpper triangular matrix R:\n", R)

    # Plot the diagonal of R to understand the contributions of the orthogonal components
    import matplotlib.pyplot as plt
    plt.bar(range(1, len(R) + 1), np.diag(R))
    plt.xlabel('Orthogonal Component Index')
    plt.ylabel('Magnitude of Contribution (Diagonal of R)')
    plt.title('Contribution of Each Orthogonal Component to A')
    plt.savefig('mpc_q_magnitude.pdf', dpi=300)
    plt.show()

    # You can also visualize the rows of Q to understand how the output dimensions are structured
    plt.figure(figsize=(6,2))
    im = plt.imshow(np.abs(Q.T), cmap='viridis', aspect='auto')
    plt.colorbar(im, label='Magnitude')
    # plt.xlabel('Latent Dimension')
    # plt.ylabel('Output Dimension')

    # Add labels for the latent dimensions (z1, z2, z3)
    plt.yticks(ticks=[0, 1, 2], labels=['$z_1$', '$z_2$', '$z_3$'])
    plt.xticks(ticks=np.arange(len(var_names)), labels=[f"${name}$" for name in var_names], rotation=90)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('mpc_q_magnitude.pdf', dpi=300)
    plt.show()

    # for i in range(decoder_weight.shape[1]):
    #     # Plot histogram of weights
    #     plt.figure(figsize=(8, 5))
    #     plt.bar(np.arange(decoder_weight.shape[0]), decoder_weight[:,i])
    #     plt.xlabel("Weight Value")
    #     plt.ylabel("Frequency")
    #     plt.title("Histogram of Decoder Weights")
    #     plt.show()

    # # Print and compare the results
    # plt.figure(figsize=(6, 6))
    # plt.scatter(true_optima[:, :, 0], true_optima[:, :, 3], label='true', s=5, alpha=0.1)
    # plt.scatter(pred_optima[:, :, 0], pred_optima[:, :, 3], label='pred', s=5, alpha=0.1)
    # # plt.scatter(pred_optima_naive[:, :, 0], pred_optima_naive[:, :, 1], label='pred naive', s=5)
    # plt.xlabel('$x_1$')
    # plt.ylabel('$x_2$')
    # plt.title(f'RMSE:, {rmse:.2f}, RMSE naive: {rmse_naive:.2f}')
    # plt.legend()
    # # plt.xlim([-1,1])
    # # plt.ylim([-1,1])
    # # plt.axis('equal')
    # plt.show()

def main():
    """
    Main function for testing the model and evaluating optimization performance.
    """
    # Configuration
    input_dim = 14
    latent_dim = 3
    batch_size = 128
    model_path = f"../../out/mpc/model_{input_dim}d_{latent_dim}l_500iter_exp_decay_alpha_05_trivial.pt"

    # Choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the test dataset
    test_dataset = FunctionExplorationDataset('test', scale_data=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Evaluate the model and optimization approaches
    test_model(test_loader, model_path, input_dim, latent_dim, device)

if __name__ == "__main__":
    main()
