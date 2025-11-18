import torch
import torch.optim as optim
import time
from torch.utils.data import DataLoader

import torch.optim.lr_scheduler as lr_scheduler
import os, sys

# Add the root directory (2 levels up from this file)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from numerical_examples.robot.dataset import RobotDataset
from numerical_examples.robot.models import Autoencoder
from numerical_examples.robot.losses import WeightedSoftmaxMSELoss

# Helper functions for training and evaluation
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train for one epoch.

    :param model: The autoencoder model
    :param dataloader: DataLoader for the training data
    :param optimizer: Optimizer for updating the model's weights
    :param criterion: Loss function (e.g., MSE Loss)
    :param device: Device (CPU or GPU)
    :return: Average loss for the epoch
    """
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for (idx, batch) in enumerate(dataloader):
        xopt = batch['Xopt'].to(device)
        fopt = batch['fopt'].to(device)

        # if idx == 0:
        #     # print("DEBUG Xopt[0]:", xopt[0])
        #     print("DEBUG fopt[0]:", fopt[0])
        #
        #     with torch.no_grad():
        #         recon = model(xopt)
        #         err = (xopt - recon) ** 2
        #         print("Reconstruction error[0]:", err[0].mean().item())

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass through the Autoencoder
        reconstructed_x = model(xopt)

        # Calculate the loss (MSE between original and reconstructed)
        weights = fopt
        loss = criterion(reconstructed_x, xopt, weights)

        running_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def eval_one_epoch(model, dataloader, criterion, device):
    """
    Evaluate the model on the validation dataset for one epoch.

    :param model: The autoencoder model
    :param dataloader: DataLoader for the validation data
    :param criterion: Loss function (e.g., MSE Loss)
    :param device: Device (CPU or GPU)
    :return: Average loss for the epoch
    """
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0

    with torch.no_grad():  # Disable gradient calculation during evaluation
        for batch in dataloader:
            xopt = batch['Xopt'].to(device)
            fopt = batch['fopt'].to(device)

            # Forward pass through the Autoencoder
            reconstructed_x = model(xopt)

            # Calculate the loss (MSE between original and reconstructed)
            weights = fopt
            loss = criterion(reconstructed_x, xopt, weights)

            running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)

    return avg_loss

def train(model, train_loader, val_loader, optimizer, scheduler, criterion, device, epochs, checkpoint_path):
    """
    Train the autoencoder model with both training and validation loops, saving a single checkpoint.

    :param model: The autoencoder model
    :param train_loader: DataLoader for the training data
    :param val_loader: DataLoader for the validation data
    :param optimizer: Optimizer for updating the model's weights
    :param criterion: Loss function (e.g., MSE Loss)
    :param device: Device (CPU or GPU)
    :param epochs: Number of epochs to train
    :param model_dir: Directory to save the model
    :param cfg: Configuration object with settings
    :param checkpoint_path: Path for saving the checkpoint
    """
    best_val_loss = float('inf')  # Initialize best validation loss as infinity

    LOSS_TRAIN = []
    LOSS_VALID = []

    time_start = time.time()

    for epoch in range(epochs):
        # Training
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Evaluation
        val_loss = eval_one_epoch(model, val_loader, criterion, device)

        LOSS_TRAIN.append(train_loss)
        LOSS_VALID.append(val_loss)

        # Save checkpoint if the validation loss improves
        if train_loss < best_val_loss:
            best_val_loss = train_loss
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'train_time': time.time() - time_start,
                'train_loss': LOSS_TRAIN,
                'val_loss': LOSS_VALID,
            }
            torch.save(checkpoint, checkpoint_path)  # Save the checkpoint

            print(f"Model checkpoint saved to {checkpoint_path}")

        # Adjust the learning rate using the scheduler
        scheduler.step()

        # Print progress every 10 epochs
        if epoch % 10 == 0:
            lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {lr:.6f}")
            print('-' * 100)


def main():
    """
    Main function to execute the training of the Autoencoder.
    """
    # Configuration (example)
    input_dim = 21  # Dimensionality of the input data (e.g., for the Rosenbrock function)
    latent_dim = 10  # Latent space dimension (reduced dimensionality)
    alpha = 0

    x_bounds = (
            [(0, 500)] * 7 +
            [(0, 200)] * 7 +
            [(0, 15000)] * 7
    )

    learning_rate = 1e-5
    batch_size = 100
    epochs = 100_000

    home_dir = os.path.expanduser('~')
    model_dir = os.path.join(home_dir, 'meta-glis/numerical_examples/robot/out/robot')
    os.makedirs(model_dir, exist_ok=True)

    checkpoint_path = os.path.join(model_dir, f'model_{input_dim}d_{latent_dim}l_300iter_exp_decay_alpha_{alpha:02}.pt')

    # Load the dataset (training and validation)
    train_dataset = RobotDataset('train', best_only=True)
    val_dataset = RobotDataset('valid', best_only=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim, lat_lb=0.0, lat_ub=1.0, out_bounds=x_bounds)
    try:
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint['model'])
    except Exception as e:
        print("No previous checkpoint found", e)

    # Choose device (GPU if available, else CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize the cosine annealing scheduler
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)

    criterion = WeightedSoftmaxMSELoss(alpha=alpha, x_bounds=x_bounds, maximize=True)  # Mean Squared Error loss for reconstruction

    # Start training
    train(model, train_loader, val_loader, optimizer, scheduler, criterion, device, epochs, checkpoint_path)


if __name__ == "__main__":
    main()
