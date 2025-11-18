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
    def __init__(self, alpha=0.1, x_bounds=None, maximize=True):
        """
        :param alpha: decay rate for weighting. Lower alpha -> stronger focus on top-ranked samples.
        :param x_bounds: optional list of (min, max) tuples per feature, used to normalize squared error.
        :param maximize: if True, higher f_vals are better; if False, lower f_vals are better.
        """
        super(WeightedSoftmaxMSELoss, self).__init__()
        self.alpha = alpha
        self.x_bounds = x_bounds
        self.maximize = maximize

    def forward(self, y_pred, y_true, f_vals):
        # Sort f_vals depending on whether we maximize or minimize
        sorted_indices = torch.argsort(f_vals, dim=1, descending=self.maximize)
        ranks = torch.argsort(sorted_indices, dim=1).float()

        # Compute exponentially decaying weights
        if self.alpha == 0:
            # Hard selection: one-hot mask at best rank (rank == 0)
            weights = (ranks == 0).float()
        else:
            weights = self.alpha ** (ranks.max(dim=1, keepdim=True).values - ranks)
            weights = weights / weights.sum(dim=1, keepdim=True)

        # Compute squared error
        squared_diff = (y_true - y_pred) ** 2

        # Optionally normalize by feature range
        if self.x_bounds is not None:
            for i in range(squared_diff.shape[-1]):
                lower_bound, upper_bound = self.x_bounds[i]
                range_ = upper_bound - lower_bound
                if range_ != 0:
                    squared_diff[:, :, i] /= range_**2

        # Apply weights and compute final loss
        loss = weights * squared_diff  # Match shape for broadcasting
        active_elements = (weights > 0).float()  # (N, T, 1)
        total_active = active_elements.sum()
        loss_sum = (loss * active_elements).sum()
        return loss_sum / total_active


class SoftplusLoss(nn.Module):
    def __init__(self):
        super(SoftplusLoss, self).__init__()
        self.softplus = nn.Softplus(beta=1, threshold=20)  # Softplus activation
        self.mse_loss = nn.MSELoss()

    def forward(self, y_pred, y_true):
        loss = self.mse_loss(y_pred, y_true)
        return loss  # Sum over batch
