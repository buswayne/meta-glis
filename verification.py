import numpy as np
import torch
from benchmarks import quadratic_function_torch, quadratic_function_np

# Quadratic function implementations as provided

# Set random seeds for consistency
np.random.seed(16)
torch.manual_seed(16)

# Define dimensions and create matching inputs
dim = 3

# Create random symmetric A and random b
base = np.random.rand(dim, dim)
A_np = 1 * (1 + 0.1 * np.random.rand(1)) * (base + base.T) / 2  # Symmetrize and scale
b_np = 1 * (1 + 0.1 * np.random.rand(dim))  # Randomized vector, scaled
c_np = 2.5

# Convert NumPy A, b, c to PyTorch tensors
A_torch = torch.tensor(A_np, dtype=torch.float32)
b_torch = torch.tensor(b_np, dtype=torch.float32)
c_torch = torch.tensor(c_np, dtype=torch.float32)

# Define input vector
x_np = np.random.rand(dim)
x_torch = torch.tensor(x_np, dtype=torch.float32)

# Create quadratic functions
quad_np = quadratic_function_np(A=A_np, b=b_np, dim=dim, c=c_np)
quad_torch = quadratic_function_torch(A=A_torch, b=b_torch, dim=dim, c=c_torch)

# Evaluate both functions
output_np = quad_np(x_np)
output_torch = quad_torch(x_torch).item()  # Convert PyTorch tensor to scalar

# Compare results
print("NumPy output:", output_np)
print("PyTorch output:", output_torch)
print("Difference:", abs(output_np - output_torch))
