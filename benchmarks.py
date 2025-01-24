import numpy as np
import torch
from pyswarm import pso  # Assuming you have the pyswarm library installed for PSO
from scipy.optimize import differential_evolution

def exact_solution(A, b, c=0):
    # Compute the exact solution for the quadratic function
    x_opt = -np.linalg.inv(A).dot(b)
    # Compute the function value at the optimal solution
    f_opt = 0.5 * x_opt.T @ A @ x_opt + b.T @ x_opt + c
    return x_opt, f_opt

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

def quadratic_function_torch(A=None, b=None, dim=10, c=0):
    """
    A parameterized quadratic function in a specified number of dimensions (PyTorch version).

    :param A: Symmetric matrix for the quadratic term (dim x dim). If None, defaults to a diagonal matrix.
    :param b: Coefficient vector for the linear term (dim,). If None, defaults to a range of integers.
    :param c: Constant scalar term.
    :param dim: Number of dimensions/variables for the quadratic function (default is 10).
    :return: A function to evaluate the quadratic function.
    """
    # Default A to a symmetric diagonal matrix with increasing values
    if A is None:
        diag = torch.arange(1, dim + 1, dtype=torch.float32)
        A = torch.diag(diag) + 0.5 * torch.eye(dim, k=1) + 0.5 * torch.eye(dim, k=-1)

    # Default b to a vector alternating between positive and negative integers
    if b is None:
        b = torch.tensor([(-1) ** i * (i + 1) for i in range(dim)], dtype=torch.float32)

    def objective(x):
        """
        Compute the quadratic function value for a given input x.

        :param x: Input vector of shape (dim,) or (batch x dim).
        :return: The evaluated quadratic function value.
        """
        # Ensure all tensors have the same dtype
        dtype = torch.float32
        A_cast = A.to(dtype)
        b_cast = b.to(dtype)
        x_cast = x.to(dtype)

        # Compute Ax
        if A_cast.ndim == 2:  # Single A
            Ax = torch.matmul(x_cast, A_cast)
        else:  # Batched A
            Ax = torch.bmm(x_cast.unsqueeze(1), A_cast).squeeze(1)

        # Compute linear term
        bx = torch.matmul(x_cast, b_cast.T) if b_cast.ndim == 1 else torch.sum(x_cast * b_cast, dim=1)

        return 0.5 * torch.sum(x_cast * Ax, dim=-1) + bx + c

    return objective

def quadratic_function_np(A=None, b=None, dim=10, c=0):
    """
    A parameterized quadratic function in a specified number of dimensions.

    :param A: Symmetric matrix for the quadratic term (dim x dim). If None, defaults to a diagonal matrix.
    :param b: Coefficient vector for the linear term (dim,). If None, defaults to a range of integers.
    :param c: Constant scalar term.
    :param dim: Number of dimensions/variables for the quadratic function (default is 10).
    :return: A function to evaluate the quadratic function.
    """
    # Default A to a symmetric diagonal matrix with increasing values
    if A is None:
        A = np.diag(np.arange(1, dim + 1)) + 0.5 * np.eye(dim, k=1) + 0.5 * np.eye(dim, k=-1)

    # Default b to a vector alternating between positive and negative integers
    if b is None:
        b = np.array([(-1) ** i * (i + 1) for i in range(dim)])

    def objective(x):
        """
        Compute the quadratic function value for a given input x.

        :param x: Input vector of length `dim`.
        :return: The evaluated quadratic function value.
        """
        x = np.array(x)  # Ensure x is a numpy array
        return 0.5 * x @ A @ x + b @ x + c

    return objective

# Use PSO to minimize the Rosenbrock function
if __name__ == "__main__":
    dim = 20  # Dimension size

    # Choose the benchmark function
    function_name = "quadratic"  # Change to "quadratic" to use the quadratic function

    if function_name == "rosenbrock":
        A = 100 * (1 + 0.1 * np.random.rand(1))  # Scaling factor for the quadratic term
        B = 1 * (1 + 0.1 * np.random.rand(1))  # Scaling factor for the non-quadratic term
        shifts = np.random.rand(dim)  # Shift vector for Rosenbrock function
        func = rosenbrock_function_np(A=A, B=B, shifts=shifts, dim=dim)
    elif function_name == "quadratic":
        base = np.random.rand(dim, dim)
        A = 1 * (1 + 0.1 * np.random.rand(dim)) * (base + base.T) / 2  # Symmetrize and scale
        print(A.shape)
        b = 1 * (1 + 0.1 * np.random.rand(dim))  # Randomized vector, scaled
        func = quadratic_function_np(A=A, b=b, dim=dim)
    else:
        raise ValueError("Invalid function name. Choose 'rosenbrock' or 'quadratic'.")


    if function_name == "quadratic":
        # Exact solution and function value
        x_exact, f_exact = exact_solution(A, b)

        print("Exact solution:", x_exact)
        print("Function value at exact solution:", f_exact)

    else:
        # Define bounds (typically -30 to 30 for many benchmark functions)
        bounds = [(-30, 30) for _ in range(dim)]

        # Perform differential evolution optimization
        result = differential_evolution(func, bounds, strategy='best1bin', maxiter=1000, popsize=15, tol=1e-6)

        # Print results
        print("Optimized solution:", result.x)
        print("Objective function value at optimized solution:", result.fun)

