import torch
import torch.nn as nn


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
    def __init__(self):
        super(WeightedSoftmaxMSELoss, self).__init__()

    def forward(self, y_pred, y_true, f_vals):
        # Compute weights using softmax on -f_vals
        weights = torch.softmax(-f_vals, dim=0)

        # Compute weighted loss
        squared_diff = (y_true - y_pred) ** 2

        loss = weights * squared_diff

        return loss.sum()  # Sum to match the normalization

class SoftmaxLoss(nn.Module):
    def __init__(self):
        super(SoftmaxLoss, self).__init__()

    def forward(self, y_pred, y_true, f_vals):

        min_f = f_vals.min()

        # Compute weights using softmax on -f_vals
        weights = torch.softmax(-f_vals, dim=0)

        # Compute weighted loss
        squared_diff = (min_f - y_pred) ** 2

        loss = weights * squared_diff

        return loss.sum()  # Sum to match the normalization


class SoftplusLoss(nn.Module):
    def __init__(self):
        super(SoftplusLoss, self).__init__()
        self.softplus = nn.Softplus(beta=1, threshold=20)  # Softplus activation
        self.mse_loss = nn.MSELoss()

    def forward(self, y_pred, y_true):
        loss = self.mse_loss(y_pred, y_true)
        return loss  # Sum over batch
