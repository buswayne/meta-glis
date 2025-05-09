import os
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from pyswarm import pso  # Assuming you have the pyswarm library installed for PSO
from sklearn.preprocessing import StandardScaler
import pickle

class FunctionExplorationDataset(Dataset):
    def __init__(self, data_path, scale_data=False):
        """
        Dataset class to load precomputed optima and their associated parameters.

        :param data_path: Path to the file containing precomputed optima and parameters.
        """
        # Load the data
        data = torch.load(data_path, weights_only=True)
        data_path = data_path.split('../')[2]
        function_name = data_path.split('/')[1]
        with_exploration = data_path.split('/')[2]
        fold_type = data_path.split('/')[3]
        input_dim = data_path.split('/')[4].split('_')[1].split('d')[0]
        perturb_pct = data_path.split('/')[4].split('_')[2].split('pct')[0]

        self.X = data["X"] # all points
        self.f = data["f"] # all function vals
        self.params = data["params"]  # Parameters (A, B, shifts)

        if scale_data:
            scaler_f_filename = os.path.join("../data",
                                             function_name,
                                             with_exploration,
                                             "scalers",
                                             "f",
                                             f"{input_dim}d_{perturb_pct}pct.pkl")
            if fold_type == "train":
                scaler_f = StandardScaler()
                self.f = torch.tensor(scaler_f.fit_transform(self.f.squeeze(-1)), dtype=torch.float32).unsqueeze(-1)
                with open(scaler_f_filename, "wb") as file:
                    pickle.dump(scaler_f, file)
            else:
                with open(scaler_f_filename, "rb") as file:
                    scaler_f = pickle.load(file)
                self.f = torch.tensor(scaler_f.transform(self.f.squeeze(-1)), dtype=torch.float32).unsqueeze(-1)

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
        f = self.f[idx]
        params = self.params[idx]

        # Return both optimized point and parameters (A, B, shifts)
        return {"Xopt": X, "fopt": f, "params": params}

class FunctionDataset(Dataset):
    def __init__(self, data_path):
        """
        Dataset class to load precomputed optima and their associated parameters.

        :param data_path: Path to the file containing precomputed optima and parameters.
        """
        # Load the data
        data = torch.load(data_path, weights_only=True)
        self.Xopt = data["Xopt"]  # Optimized points
        self.fopt = data["fopt"]  # Optimized points
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
        Xopt = self.Xopt[idx]
        fopt = self.fopt[idx]
        params = self.params[idx]

        # Return both optimized point and parameters (A, B, shifts)
        return {"Xopt": Xopt, "fopt": fopt, "params": params}

if __name__ == "__main__":
    batch_size = 128

    # Step 2: Load data in the dataset
    dataset = FunctionExplorationDataset(
        "../data/rosenbrock/with_exploration/valid/rosenbrock_2d_0.5pct_optima_data.pt", scale_data=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for batch in dataloader:
        print(batch["Xopt"].shape)
        print(batch["fopt"].shape)
        print(batch["params"]['A'].shape)
        print(batch["params"]['B'].shape)
        print(batch["params"]['shifts'].shape)
        break
