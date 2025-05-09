import os
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from pyswarm import pso  # Assuming you have the pyswarm library installed for PSO
from sklearn.preprocessing import StandardScaler
import pickle

from numerical_examples.dataset import FunctionExplorationDataset


class FunctionExplorationDataset(Dataset):
    def __init__(self, fold_type, scale_data=False):
        """
        Dataset class to load precomputed optima and their associated parameters.

        :param data_path: Path to the file containing precomputed optima and parameters.
        """
        X = []  # will be of shape (B, n_expl, n_dim)
        J = []  # will be of shape (B, n_expl, 1)
        # Xopt = []
        # Jopt = []
        self.params = []

        if fold_type == 'train':
            samples = range(1000,1200)
        elif fold_type == 'valid':
            samples = range(1200,1300)
        elif fold_type == 'test':
            samples = range(1200,1300)

        for i in samples:
            experiment = i
            with open(f'../../efficient-calibration-embedded-MPC/results/params/{experiment:04}.pkl', 'rb') as f:
                current_params = pickle.load(f)
            with open(
                    f'../../efficient-calibration-embedded-MPC/results/experiment_{experiment:04}_res_slower1_500iter_GLIS_PC.pkl',
                    'rb') as f:
                experiment = pickle.load(f)
            self.params.append(current_params)
            X.append(experiment['X_sample'][:500,:])
            J.append(experiment['J_sample'][:500,:])
            # Xopt.append(experiment['x_opt'])
            # Jopt.append(experiment['J_opt'])

        self.X = torch.tensor(np.stack(X, axis=0), dtype=torch.float32)
        self.J = np.stack(J, axis=0)


    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        Return a sample from the dataset: both optimized points (xopt) and associated parameters.
        """
        # Fetch the optimized point and its associated parameters
        X = self.X[idx]#[-5:,:]
        J = self.J[idx]#[-5:,:]
        params = self.params[idx]

        # Return both optimized point and parameters (A, B, shifts)
        return {"Xopt": X, "fopt": J, "params": params}

class FunctionDataset(Dataset):
    def __init__(self, fold_type):
        """
        Dataset class to load precomputed optima and their associated parameters.

        :param data_path: Path to the file containing precomputed optima and parameters.
        """
        # Load the data
        X = []  # will be of shape (B, n_expl, n_dim)
        J = []  # will be of shape (B, n_expl, 1)
        self.params = []

        if fold_type == 'train':
            samples = range(1000,1200)
        elif fold_type == 'valid':
            samples = range(1200,1290)
        elif fold_type == 'test':
            samples = range(1200,1290)

        for i in samples:
            experiment = i
            with open(f'../../efficient-calibration-embedded-MPC/results/params/{experiment:04}.pkl', 'rb') as f:
                current_params = pickle.load(f)
            with open(
                    f'../../efficient-calibration-embedded-MPC/results/experiment_{experiment:04}_res_slower1_500iter_GLIS_PC.pkl',
                    'rb') as f:
                experiment = pickle.load(f)
            self.params.append(current_params)
            X.append(experiment['x_opt'].reshape(1,-1))
            J.append(experiment['J_opt'].reshape(1,-1))
            # Xopt.append(experiment['x_opt'])
            # Jopt.append(experiment['J_opt'])

        self.X = torch.tensor(np.stack(X, axis=0), dtype=torch.float32)
        self.J = np.stack(J, axis=0)

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        Return a sample from the dataset: both optimized points (xopt) and associated parameters.
        """
        # Fetch the optimized point and its associated parameters
        X = self.X[idx]
        J = self.J[idx]
        params = self.params[idx]

        # Return both optimized point and parameters (A, B, shifts)
        return {"Xopt": X, "fopt": J, "params": params}

if __name__ == "__main__":
    batch_size = 32

    # Step 2: Load data in the dataset
    dataset = FunctionExplorationDataset('train')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for batch in dataloader:
        print(batch["Xopt"].shape)
        print(batch["fopt"].shape)
        # print(batch["params"]['M'].shape)
        # print(batch["params"]['A'].shape)
        # print(batch["params"]['B'].shape)
        # print(batch["params"]['shifts'].shape)
        break
