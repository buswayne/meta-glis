import torch
from numerical_examples.robot.dataset import RobotDataset

maximize = True
alpha = 0.0
threshold = 1e-6

# Load datasets
ds_best = RobotDataset(fold_type='train', best_only=True)
ds_all = RobotDataset(fold_type='train', best_only=False)

X_best = ds_best.X.squeeze(1)  # (N, d)
J_best = ds_best.J.squeeze(1).squeeze(1)  # (N,)

X_all = ds_all.X  # (N, T, d)
J_all = ds_all.J.squeeze(-1)  # (N, T)

# Ranking logic
sorted_indices = torch.argsort(J_all, dim=1, descending=maximize)
ranks = torch.argsort(sorted_indices, dim=1).float()

# Compute weights
if alpha == 0:
    best_mask = (ranks == 0).float()
    weights = best_mask
else:
    weights = alpha ** (ranks.max(dim=1, keepdim=True).values - ranks)
    weights = weights / weights.sum(dim=1, keepdim=True)

# Compute weighted X from alpha=0 logic
weights_expanded = weights.unsqueeze(-1)  # (N, T, 1)
X_weighted = (X_all * weights_expanded).sum(dim=1)  # (N, d)


i = 0  # Check first sample
print("\n--- Sample 0 Deep Dive ---")

# Step 1: Show raw objective values
print("J_all[0]:", J_all[i])

# Step 2: Show computed ranks
print("ranks[0]:", ranks[i])

# Step 3: Show weights
print("weights[0]:", weights[i])

# Step 4: Show best index from weights (where weight is max)
print("alpha-based best index:", torch.argmax(weights[i]).item())

# Step 5: Compare with best_only=True value
print("X_best[0]:", X_best[i])
print("X_weighted[0]:", X_weighted[i])
print("||X_weighted - X_best|| =", torch.norm(X_weighted[i] - X_best[i]).item())


# Compare with X_best
diff = torch.norm(X_weighted - X_best, dim=1)  # (N,)
mismatched = (diff > threshold).nonzero().squeeze()

print(f"\nTotal samples: {X_best.shape[0]}")
print(f"Number of mismatched weighted X vectors: {mismatched.numel()}")

if mismatched.numel() > 0:
    for i in mismatched[:5]:
        i = i.item()
        print(f"\nSample {i}:")
        print(f"||X_weighted - X_best|| = {diff[i].item()}")
        print("X_best[i]:", X_best[i])
        print("X_weighted[i]:", X_weighted[i])

print("\nCheck complete.")
