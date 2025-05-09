import torch
import torch.nn as nn
from sympy.physics.vector.tests.test_printing import alpha


class WeightedMSELoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(WeightedMSELoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true, f_vals):
        # Invert weights: lower values → higher importance
        weights = 1 / (f_vals + self.epsilon)

        # Normalize weights (so they sum to 1)
        normalized_weights = weights / weights.sum()

        # Compute weighted loss
        loss = normalized_weights * (y_true - y_pred) ** 2

        return loss.sum()  # Use `.sum()` instead of `.mean()` to respect the normalization

class WeightedSoftmaxMSELoss(nn.Module):
    def __init__(self, alpha=0.1, x_bounds=None):
        super(WeightedSoftmaxMSELoss, self).__init__()
        self.alpha = alpha
        self.x_bounds = x_bounds

    def forward(self, y_pred, y_true, f_vals):
        # Sort f_vals and get ranking (lower f_vals = better)
        # print('f:', f_vals.tolist())
        sorted_indices = torch.argsort(f_vals, dim=1)  # Shape: (batch, n_iter)
        # print('sorted_indices:', sorted_indices.tolist())
        ranks = torch.argsort(sorted_indices, dim=1).float()  # Shape: (batch, n_iter)
        # print('ranks:', ranks.tolist())
        # Compute exponentially decaying weights
        weights = self.alpha**ranks  # Shape: (batch, n_iter)
        # print('weights:', weights.tolist())
        # Compute weights using softmax on -f_vals
        # weights = torch.softmax(-f_vals, dim=1)

        # Compute weighted loss
        squared_diff = (y_true - y_pred) ** 2

        # Rescale the squared difference for each feature by its range
        if self.x_bounds is not None:
            for i in range(squared_diff.shape[-1]):  # Iterate over each feature
                lower_bound, upper_bound = self.x_bounds[i]
                range_ = upper_bound - lower_bound
                if range_ != 0:  # To avoid division by zero
                    squared_diff[:, :, i] /= range_  # Scale squared difference by the range

        # Find the index of the minimum value in f_vals across the iterations (dim=1)
        # min_idx = torch.argmin(f_vals, dim=1, keepdim=True)
        #
        # # Initialize weights as zeros
        # weights = torch.zeros_like(f_vals)

        # Set the weight corresponding to the minimum f_vals value to 1 (or any other value you prefer)
        # weights.scatter_(1, min_idx, 1.0)



        loss = weights * squared_diff

        # print('loss:', loss.tolist())

        return loss.mean()  # Sum to match the normalization



class SoftplusLoss(nn.Module):
    def __init__(self):
        super(SoftplusLoss, self).__init__()
        self.softplus = nn.Softplus(beta=1, threshold=20)  # Softplus activation
        self.mse_loss = nn.MSELoss()

    def forward(self, y_pred, y_true):
        loss = self.mse_loss(y_pred, y_true)
        return loss  # Sum over batch
