import logging

import torch
import torch.optim as optim
import time
from torch import nn
from torch.utils.data import DataLoader
from models import Autoencoder
from dataset import FunctionDataset
from benchmarks import rosenbrock_function_torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
import wandb
import os

# Set default values for distributed variables
os.environ.setdefault('MASTER_ADDR', 'localhost')
os.environ.setdefault('MASTER_PORT', '12346')
os.environ.setdefault('RANK', '0')
os.environ.setdefault('WORLD_SIZE', '1')


# Initialize WandB on rank 0
def initialize_wandb(rank, project_name='project_name', run_name='run_name'):
    if rank == 0:
        wandb.init(project=project_name, name=run_name)

# Log metrics using WandB
def log_metrics(epoch, train_loss, val_loss, rank):
    if rank == 0:
        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})

# Setup distributed training
def setup(rank, world_size):
    init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    print(f"[Rank {rank}] Using device: {torch.cuda.get_device_name(rank)}")

def cleanup():
    destroy_process_group()

# Helper functions for training and evaluation
def train_one_epoch(model, dataloader, criterion, optimizer, scheduler=None, rank=0):
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
        xopt = batch['xopt'].to(rank)
        A = batch['params']['A'].to(rank)
        B = batch['params']['B'].to(rank)
        shifts = batch['params']['shifts'].to(rank)

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

        # Step the scheduler (for learning rate adjustment)
        if scheduler is not None:
            scheduler.step()

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def eval_one_epoch(model, dataloader, criterion, rank):
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
            xopt = batch['xopt'].to(rank)
            A = batch['params']['A'].to(rank)
            B = batch['params']['B'].to(rank)
            shifts = batch['params']['shifts'].to(rank)

            # Create Rosenbrock function
            rosenbrock_func = rosenbrock_function_torch(A, B, shifts, xopt.shape[1])

            # Forward pass through the Autoencoder
            reconstructed_x = model(xopt)

            # Calculate the loss (MSE between original and reconstructed)
            loss = criterion(rosenbrock_func(reconstructed_x), rosenbrock_func(xopt))

            running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def train(model, train_loader, valid_loader, criterion, optimizer, scheduler, rank, epochs, checkpoint_path):
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

        current_lr = optimizer.param_groups[0]['lr']

        # Training
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, rank)

        # Evaluation
        val_loss = eval_one_epoch(model, valid_loader, criterion, rank)

        LOSS_TRAIN.append(train_loss)
        LOSS_VALID.append(val_loss)

        if rank == 0:

            log_metrics(epoch + 1, train_loss, val_loss, rank)

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

            if epoch % 10 == 0:
                # Print loss for current epoch
                print(f"Epoch {epoch + 1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f} - "
                      f"Val Loss: {val_loss:.4f} - "
                      f"LR: {current_lr:.6f} - ")
                print('-'*100)


def main(rank, world_size):
    """
    Main function to execute the training of the Autoencoder.
    """
    setup(rank, world_size)

    # Configuration (example)
    input_dim = 10  # Dimensionality of the input data (e.g., for the Rosenbrock function)
    latent_dim = 3  # Latent space dimension (reduced dimensionality)
    learning_rate = 1e-4
    batch_size = 64
    epochs = 1_000_000
    checkpoint_path = f"out/model_{input_dim}d_{latent_dim}l_checkpoint.pt"
    model_dir = '../../out'
    os.makedirs(model_dir, exist_ok=True)

    if rank == 0:
        initialize_wandb(rank, project_name='meta-glis', run_name=f"model_{input_dim}d_{latent_dim}l")

    # Load the dataset (training and validation)
    train_data_path = "data/rosenbrock/train/rosenbrock_10d_0.5pct_optima_data.pt"
    valid_data_path = "data/rosenbrock/valid/rosenbrock_10d_0.5pct_optima_data.pt"

    train_dataset = FunctionDataset(train_data_path)
    valid_dataset = FunctionDataset(valid_data_path)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler, pin_memory=True, num_workers=0)

    # Initialize the model
    model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim).to(rank)

    try:
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        model.load_state_dict(state_dict)
    except:
        logging.warning("Could not load pretrained model")
        raise Exception("Could not load pretrained model")

    model = DDP(model, device_ids=[rank])

    # Define the optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # scheduler = optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=learning_rate,
    #     steps_per_epoch=len(train_loader),
    #     epochs=epochs,
    #     pct_start=0.2,
    #     anneal_strategy='cos',
    #     cycle_momentum=True,
    #     max_momentum=0.95,
    # )
    scheduler = None

    criterion = nn.MSELoss()  # Mean Squared Error loss for reconstruction

    # Start training
    train(model, train_loader, valid_loader, criterion, optimizer, scheduler, rank, epochs, checkpoint_path)
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size)
    main()
