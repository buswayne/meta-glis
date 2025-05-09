import torch
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from dataset_mpc import FunctionExplorationDataset, FunctionDataset
import torch.optim.lr_scheduler as lr_scheduler
import os, sys
sys.path.append('..')
from models import Autoencoder, AutoencoderTrivial
# from models_complex import Autoencoder
from losses import WeightedSoftmaxMSELoss

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

    for batch in dataloader:
        xopt = batch['Xopt'].to(device)
        fopt = batch['fopt'].to(device)

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
    input_dim = 14  # Dimensionality of the input data (e.g., for the Rosenbrock function)
    latent_dim = 3  # Latent space dimension (reduced dimensionality)

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

    learning_rate = 1e-4
    batch_size = 100
    epochs = 100_000
    checkpoint_path = f"../../out/mpc/model_{input_dim}d_{latent_dim}l_500iter_exp_decay_alpha_05_trivial.pt"
    model_dir = '../../out'
    os.makedirs(model_dir, exist_ok=True)

    # Load the dataset (training and validation)
    train_dataset = FunctionExplorationDataset('train')
    val_dataset = FunctionExplorationDataset('valid')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = AutoencoderTrivial(input_dim=input_dim, latent_dim=latent_dim, lat_lb=0.0, lat_ub=1.0, out_bounds=x_bounds)
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
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    criterion = WeightedSoftmaxMSELoss(alpha=0., x_bounds=x_bounds)  # Mean Squared Error loss for reconstruction

    # Start training
    train(model, train_loader, val_loader, optimizer, scheduler, criterion, device, epochs, checkpoint_path)


if __name__ == "__main__":
    main()
