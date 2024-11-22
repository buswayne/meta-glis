import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from pyswarm import pso  # Assuming you have the pyswarm library installed for PSO



class FunctionDataset(Dataset):
    def __init__(self, data_path):
        """
        Dataset class to load precomputed optima and their associated parameters.

        :param data_path: Path to the file containing precomputed optima and parameters.
        """
        # Load the data
        data = torch.load(data_path, weights_only=True)
        self.Xopt = data["Xopt"]  # Optimized points
        self.params = data["params"]  # Parameters (A, B, shifts)

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.Xopt)

    def __getitem__(self, idx):
        """
        Return a sample from the dataset: both optimized points (xopt) and associated parameters.
        """
        # Fetch the optimized point and its associated parameters
        xopt = self.Xopt[idx]
        params = self.params[idx]

        # Return both optimized point and parameters (A, B, shifts)
        return {"xopt": xopt, "params": params}

if __name__ == "__main__":
    batch_size = 128

    # Step 2: Load data in the dataset
    dataset = FunctionDataset("data/rosenbrock/train/rosenbrock_2d_optima_data.pt")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for batch in dataloader:
        print(batch["xopt"].shape)
        print(batch["params"]['A'].shape)
        print(batch["params"]['B'].shape)
        print(batch["params"]['shifts'].shape)
        break
