import torch
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from models import Decoder
from dataset import FunctionExplorationDataset
from benchmarks import rosenbrock_function_torch
from losses import SoftplusLoss
import os
import torch.optim.lr_scheduler as lr_scheduler

# Helper functions for training and evaluation
def train_one_epoch(model, dataloader, optimizer, criterion, device, minimize_over_X=True):
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
        A = batch['params']['A'].to(device)
        B = batch['params']['B'].to(device)
        shifts = batch['params']['shifts'].to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass through the Autoencoder
        # reconstructed_x = model(xopt)
        # Sample latent variables
        z = 0.1*torch.randn(xopt.shape[0], 1, 5).to("cuda")  # Randomly sampled z
        reconstructed_x = model(z)

        # Calculate the loss (MSE between original and reconstructed)
        rosenbrock_func = rosenbrock_function_torch(A, B, shifts, z.shape[1])
        reconstructed_f = torch.empty_like(fopt)
        fopt_vect = fopt.min() * torch.ones_like(fopt)

        for i in range(reconstructed_x.shape[1]):
            point = reconstructed_x[:, i, :]
            f = rosenbrock_func(point)
            # print('f value:', f)
            # print('f log value:', torch.log1p(f))
            reconstructed_f[:, i, 0] = f.view(-1)

        loss = criterion(torch.log1p(reconstructed_f), torch.log1p(fopt_vect))

        running_loss += loss.item()

        # Backward pass and optimization
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def eval_one_epoch(model, dataloader, criterion, device, minimize_over_X=True):
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
            A = batch['params']['A'].to(device)
            B = batch['params']['B'].to(device)
            shifts = batch['params']['shifts'].to(device)

            # Forward pass through the Autoencoder
            # reconstructed_x = model(xopt)
            # Sample latent variables
            z = 0.1*torch.randn(xopt.shape[0], 1, 5).to("cuda")  # Randomly sampled z
            reconstructed_x = model(z)

            # Calculate the loss (MSE between original and reconstructed)
            rosenbrock_func = rosenbrock_function_torch(A, B, shifts, z.shape[1])
            reconstructed_f = torch.empty_like(fopt)
            fopt_vect = fopt.min() * torch.ones_like(fopt)

            for i in range(reconstructed_x.shape[1]):
                point = reconstructed_x[:, i, :]
                f = rosenbrock_func(point)
                reconstructed_f[:, i, 0] = f.view(-1)

            loss = criterion(torch.log1p(reconstructed_f), torch.log1p(fopt_vect))

            running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)

    return avg_loss

def train(model, train_loader, val_loader, optimizer, scheduler, criterion, device, epochs, checkpoint_path, minimize_over_X=True):
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
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, minimize_over_X)

        # Evaluation
        val_loss = eval_one_epoch(model, val_loader, criterion, device, minimize_over_X)

        LOSS_TRAIN.append(train_loss)
        LOSS_VALID.append(val_loss)

        # Save checkpoint if the validation loss improves
        if val_loss < best_val_loss:
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
    minimize_over_X = False
    input_dim = 15  # Dimensionality of the input data (e.g., for the Rosenbrock function)
    latent_dim = 5  # Latent space dimension (reduced dimensionality)
    learning_rate = 1e-6
    batch_size = 64
    epochs = 10_000
    X_or_f = 'X' if minimize_over_X else 'f'
    checkpoint_path = f"out/rosenbrock/with_exploration/model_decoder_only_{input_dim}d_{latent_dim}l.pt"
    model_dir = '../../out'
    os.makedirs(model_dir, exist_ok=True)

    # Load the dataset (training and validation)
    train_data_path = f"data/rosenbrock/with_exploration/train/rosenbrock_{input_dim}d_0.2pct_optima_data.pt"
    valid_data_path = f"data/rosenbrock/with_exploration/valid/rosenbrock_{input_dim}d_0.2pct_optima_data.pt"

    train_dataset = FunctionExplorationDataset(train_data_path)
    val_dataset = FunctionExplorationDataset(valid_data_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = Decoder(latent_dim=latent_dim, output_dim=input_dim)
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

    # criterion = nn.MSELoss()  # Mean Squared Error loss for reconstruction
    # criterion = WeightedMSELoss()
    # criterion = WeightedSoftmaxMSELoss()
    # criterion = SoftmaxLoss()
    criterion = SoftplusLoss()

    # Start training
    train(model, train_loader, val_loader, optimizer, scheduler, criterion, device, epochs, checkpoint_path, minimize_over_X)


if __name__ == "__main__":
    main()
