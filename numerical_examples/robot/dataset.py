import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from numerical_examples.robot.plot_utils import plot_outs


class RobotDataset(Dataset):
    def __init__(self, fold_type='train', data_path='/home/rbusetto/meta-glis/data/robot', best_only=False):
        """
        Dataset for robot run data using .npy files.
        - fold_type: 'train', 'valid', or 'test'
        - data_path: path to the directory with .npy files
        """
        assert fold_type in ['train', 'valid', 'test'], "Invalid fold_type"

        # Load .npy files
        X = np.load(f"{data_path}/runs_outputs.npy")  # (N, n_iter, n_features)
        J = np.load(f"{data_path}/runs_targets.npy")  # (N, n_iter)
        J = np.expand_dims(J, axis=-1)
        total = X.shape[0]
        train_end = int(0.7 * total)
        valid_end = int(0.85 * total)

        if fold_type == 'train':
            indices = slice(0, train_end)
        elif fold_type == 'valid':
            indices = slice(train_end, valid_end)
        else:  # 'test'
            indices = slice(valid_end, total)

        self.X = torch.tensor(X[indices], dtype=torch.float32)
        self.J = torch.tensor(J[indices], dtype=torch.float32)

        if best_only:
            # Find index of maximum J for each sample along iterations
            best_idx = np.argmax(self.J, axis=1).squeeze()  # shape (N,)

            # Extract the best objective values
            self.J = self.J[np.arange(self.J.shape[0]), best_idx, 0]  # shape (N,)
            self.X = self.X[np.arange(self.X.shape[0]), best_idx, :]  # shape (N, 21)
            self.J = self.J[:, None, None]
            self.X = self.X[:, None, :]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "Xopt": self.X[idx],  # Shape: (n_iter, n_features)
            "fopt": self.J[idx]   # Shape: (n_iter,)
        }

# Example usage
if __name__ == "__main__":
    dataset = RobotDataset(fold_type='train', best_only=True)
    dataloader = DataLoader(dataset, batch_size=500, shuffle=True)

    for batch in dataloader:
        print(batch["Xopt"].shape)  # torch.Size([32, n_iter, n_features])
        print(batch["fopt"].shape)  # torch.Size([32, n_iter])
        break

    plot_outs(batch["Xopt"], batch["fopt"])