import matplotlib.pyplot as plt
import numpy as np
import torch
from pyswarm import pso
from scipy.optimize import differential_evolution
from benchmarks import rosenbrock_function_np

# List to store points and their function values
sampled_points_and_values = []



def store_points_and_values(xk, convergence, points_and_values, objective_function):
    """Callback function to store both the sampled points and their function values."""
    f_val = objective_function(xk)  # Compute function value at xk
    points_and_values.append((xk.copy(), f_val))  # Store point and value as a tuple

def generate_optima(file_path, data_size=10000, dim=2, bounds=(-30, 30), perturb_pct=0):
    lb, ub = bounds

    # Preallocate array for optimized results
    Xopt = np.empty((data_size, dim), dtype=np.float32)
    fopt = np.empty((data_size, 1), dtype=np.float32)
    params = []
    X = np.empty((data_size, 1000, dim), dtype=np.float32)
    f = np.empty((data_size, 1000, 1), dtype=np.float32)

    for i in range(data_size):

        # Ensure points and values are reset for this run
        sampled_points_and_values = []

        print('Sample %d/%d' % (i+1, data_size))
        A = 100 * (1 + perturb_pct * np.random.uniform(-1, 1, size=1))
        B = 1 * (1 + perturb_pct * np.random.uniform(-1, 1, size=1))
        shifts = 1 * (1 + perturb_pct * np.random.uniform(-1, 1, size=dim))

        # Create Rosenbrock function
        objective_function = rosenbrock_function_np(A, B, shifts, dim)

        # Perform PSO
        bounds = [(low, high) for low, high in zip(lb, ub)]
        # xopt, fopt = pso(rosenbrock_func, lb, ub, swarmsize=2000, minfunc=1e-12, maxiter=10000)
        result = differential_evolution(objective_function,
                                        bounds,
                                        callback=lambda xk, conv: store_points_and_values(xk, conv, sampled_points_and_values, objective_function),
                                        maxiter=1000,
                                        popsize=10,  # Fixed population size
                                        tol=-np.Inf,  # No early stopping
                                        updating='deferred'
                                        )
        Xopt[i] = result.x
        fopt[i] = result.fun


        # Store the parameters for this function
        params.append({"A": torch.tensor(A), "B": torch.tensor(B), "shifts": torch.tensor(shifts)})

        points = []
        values = []
        for point, value in sampled_points_and_values:
            points.append(point)
            values.append(value)

        X[i] = np.array(points).reshape(-1,dim)
        f[i] = np.array(values).reshape(-1,1)



    # Save the data and parameters to a file
    save_data = {"Xopt": torch.tensor(Xopt, dtype=torch.float32),
                 "fopt": torch.tensor(fopt, dtype=torch.float32),
                 "X": torch.tensor(X, dtype=torch.float32),
                 "f": torch.tensor(f, dtype=torch.float32),
                 "params": params}

    # torch.save(save_data, file_path)
    # print(f"Data and parameters saved to {file_path}")

    return Xopt, fopt, X, f

if __name__ == "__main__":
    # Configuration
    input_dim = 2
    data_size = 5
    bounds = (-5. * np.ones(input_dim), 5. * np.ones(input_dim))
    perturb_pct = 0.5

    file_name = f"rosenbrock_{input_dim}d_{perturb_pct}pct_optima_data.pt"

    (Xopt_train, fopt_train, X_train, f_train) = generate_optima(
                                                "data/rosenbrock/train/" + file_name,
                                                 data_size=data_size,
                                                 dim=input_dim,
                                                 bounds=bounds,
                                                 perturb_pct=perturb_pct)


    # X_opt_valid = generate_optima("data/rosenbrock/valid/" + file_name,
    #                 data_size=data_size//5,
    #                 dim=input_dim,
    #                 bounds=bounds,
    #                 perturb_pct=pertub_pct)
    #
    # X_opt_test = generate_optima("data/rosenbrock/test/" + file_name,
    #                 data_size=data_size//5,
    #                 dim=input_dim,
    #                 bounds=bounds,
    #                 perturb_pct=pertub_pct)

    # plot

    x_vals = np.linspace(-5, 5, 500)
    y_vals = np.linspace(-5, 5, 500)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)

    A = 100 * (1 + perturb_pct * np.random.uniform(-1, 1, size=1))
    B = 1 * (1 + perturb_pct * np.random.uniform(-1, 1, size=1))
    shifts = 1 * (1 + perturb_pct * np.random.uniform(-1, 1, size=input_dim))

    # Generate the Rosenbrock function based on your parameters
    rosenbrock_obj = rosenbrock_function_np(A, B, shifts, input_dim)

    # Evaluate the function over the grid
    Z_grid = np.zeros(X_grid.shape)
    for i in range(X_grid.shape[0]):
        for j in range(X_grid.shape[1]):
            Z_grid[i, j] = rosenbrock_obj([X_grid[i, j], Y_grid[i, j]])

    plt.figure()

    all_points = np.concatenate(X_train, axis=0)  # Shape: (total_points, dim)
    all_values = np.concatenate(f_train, axis=0)  # Shape: (total_points,)

    # Create a hexbin plot
    plt.hexbin(all_points[:, 0], all_points[:, 1], C=all_values, gridsize=50, cmap='gray', mincnt=1)

    plt.colorbar(label="Function Value (f)")

    contour = plt.contour(X_grid, Y_grid, Z_grid, levels=50, cmap='coolwarm', linestyles='--')
    plt.clabel(contour, fmt='%2.1f', colors='black')  # Add contour labels

    plt.scatter(Xopt_train[:, 0], Xopt_train[:, 1], c='red', s=25, label='Optimal Points', edgecolors='black')

    plt.axis("equal")
    # plt.scatter(Xopt_valid[:, 0], Xopt_valid[:, 1], label='valid')
    # plt.scatter(Xopt_test[:, 0], Xopt_test[:, 1], label='test')
    #
    # plt.subplot(1, 3, 2)
    # plt.scatter(X_opt_train[:, 0], X_opt_train[:, 2], label='train')
    # plt.scatter(X_opt_valid[:, 0], X_opt_valid[:, 2], label='valid')
    # plt.scatter(X_opt_test[:, 0], X_opt_test[:, 2], label='test')
    #
    # plt.subplot(1, 3, 3)
    # plt.scatter(X_opt_train[:, 1], X_opt_train[:, 2], label='train')
    # plt.scatter(X_opt_valid[:, 1], X_opt_valid[:, 2], label='valid')
    # plt.scatter(X_opt_test[:, 1], X_opt_test[:, 2], label='test')

    plt.legend()
    plt.show()