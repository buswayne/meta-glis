from turtledemo.sorting_animate import start_qsort

import torch
import torch.optim as optim
import time
from torch import nn
from torch.utils.data import DataLoader
from models import Autoencoder
from dataset import FunctionDataset
from benchmarks import rosenbrock_function_torch
import os

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
        xopt = batch['xopt'].to(device)
        A = batch['params']['A'].to(device)
        B = batch['params']['B'].to(device)
        shifts = batch['params']['shifts'].to(device)

        # Create Rosenbrock function
        rosenbrock_func = rosenbrock_function_torch(A, B, shifts, xopt.shape[1])

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass through the Autoencoder
        reconstructed_x = model(xopt)

        # Calculate the loss (MSE between original and reconstructed)
        loss = criterion(rosenbrock_func(reconstructed_x), rosenbrock_func(xopt))

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
            xopt = batch['xopt'].to(device)
            A = batch['params']['A'].to(device)
            B = batch['params']['B'].to(device)
            shifts = batch['params']['shifts'].to(device)

            # Create Rosenbrock function
            rosenbrock_func = rosenbrock_function_torch(A, B, shifts, xopt.shape[1])

            # Forward pass through the Autoencoder
            reconstructed_x = model(xopt)

            # Calculate the loss (MSE between original and reconstructed)
            loss = criterion(rosenbrock_func(reconstructed_x), rosenbrock_func(xopt))

            running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def train(model, train_loader, val_loader, optimizer, criterion, device, epochs, checkpoint_path):
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
        if val_loss < best_val_loss:
            best_val_loss = val_loss
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

        if epoch % 10 == 0:
            # Print loss for current epoch
            print(f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print('-'*100)


def main():
    """
    Main function to execute the training of the Autoencoder.
    """
    # Configuration (example)
    input_dim = 20  # Dimensionality of the input data (e.g., for the Rosenbrock function)
    latent_dim = 5  # Latent space dimension (reduced dimensionality)
    learning_rate = 1e-4
    batch_size = 64
    epochs = 100000
    checkpoint_path = "out/model_checkpoint.pt"
    model_dir = 'out'
    os.makedirs(model_dir, exist_ok=True)

    # Load the dataset (training and validation)
    train_data_path = "data/rosenbrock/train/rosenbrock_optima_data.pt"
    valid_data_path = "data/rosenbrock/valid/rosenbrock_optima_data.pt"

    train_dataset = FunctionDataset(train_data_path)
    val_dataset = FunctionDataset(valid_data_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim)

    # Choose device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()  # Mean Squared Error loss for reconstruction

    # Start training
    train(model, train_loader, val_loader, optimizer, criterion, device, epochs, checkpoint_path)


if __name__ == "__main__":
    main()
