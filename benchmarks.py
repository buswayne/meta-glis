import numpy as np
import torch
from pyswarm import pso  # Assuming you have the pyswarm library installed for PSO


def rosenbrock_function_torch(A=100, B=1, shifts=None, dim=20):
    """
    A parameterized version of the Rosenbrock function with different global minimum locations along each dimension.

    :param A: Scaling factor for the quadratic term (controls the steepness).
    :param B: Scaling factor for the non-quadratic term (affects the shape).
    :param shifts: A tensor or list of shifts for each dimension (instead of a single value).
    :param dim: Number of dimensions/variables for the Rosenbrock function (default is 20)
    :return: A function to minimize
    """

    # If no specific shifts are provided, default to an array where each shift is unique
    if shifts is None:
        shifts = torch.linspace(0, 5, dim)  # Create a shift vector that is unique for each dimension
    else:
        shifts = torch.tensor(shifts, dtype=torch.float32)

    def objective(x):
        """
        Objective function that computes the Rosenbrock value for input x.
        This supports batched inputs.

        :param x: A batch of input vectors
        :return: Rosenbrock value for the batch
        """
        x = x.float()  # Ensure x is a float tensor (PyTorch tensors are more efficient with floats)

        # Calculate the Rosenbrock function value for each element in the batch
        diff = x[:, 1:] - x[:, :-1] ** 2  # Shape: [batch_size, dim-1]
        term1 = A * diff ** 2  # Shape: [batch_size, dim-1]
        term2 = B * (x[:, :-1] - shifts[:, :-1]) ** 2  # Shape: [batch_size, dim-1]

        # Sum over dimensions to get the total value per sample in the batch
        return torch.sum(term1 + term2, dim=1)  # Sum along the second axis (dim-1)

    return objective

def rosenbrock_function_np(A=100, B=1, shifts=None, dim=20):
    """
    A parameterized version of the Rosenbrock function with different global minimum locations along each dimension.

    :param A: Scaling factor for the quadratic term (controls the steepness).
    :param B: Scaling factor for the non-quadratic term (affects the shape).
    :param shifts: A list of shifts for each dimension (instead of a single value).
    :param dim: Number of dimensions/variables for the Rosenbrock function (default is 20)
    :return: A function to minimize
    """

    # If no specific shifts are provided, default to an array where each shift is unique
    if shifts is None:
        shifts = np.linspace(0, 5, dim)  # Create a shift vector that is unique for each dimension

    def objective(x):
        # Ensure x is a numpy array
        x = np.array(x)

        # Standard Rosenbrock function with parameters A, B, and shift applied independently to each dimension
        return np.sum(A * (x[1:] - x[:-1] ** 2) ** 2 + B * (x[:-1] - shifts[:-1]) ** 2)

    return objective


# Use PSO to minimize the Rosenbrock function
if __name__ == "__main__":
    dim = 20  # Dimension size
    A = 100 * (1 + 0.1 * np.random.rand(1))  # Scaling factor for the quadratic term
    B = 1 * (1 + 0.1 * np.random.rand(1))  # Scaling factor for the non-quadratic term
    shifts = np.random.rand(dim)  # Create a shift vector for each dimension

    print('A:', A, 'B:', B, 'shifts:', shifts)
    # Get the Rosenbrock function and global minimum location
    rosenbrock_func = rosenbrock_function_np(A, B, shifts, dim)

    # Define bounds (typically from -30 to 30 for the function)
    lb = -30. * np.ones(dim)  # Lower bounds
    ub = 30. * np.ones(dim)  # Upper bounds

    # Perform PSO to find the global minimum
    xopt, fopt = pso(rosenbrock_func, lb, ub, swarmsize=5000, minfunc=1e-12, maxiter=10000)

    # Print the results
    print("Optimized solution:", xopt)
    print("Objective function value at optimized solution:", fopt)
    print(rosenbrock_func(np.zeros(dim)))