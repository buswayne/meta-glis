import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import differential_evolution
from benchmarks import rosenbrock_function_np


def generate_optima(file_path, data_size=10000, dim=2, bounds=(-30, 30), perturb_pct=0):
    lb, ub = bounds

    # Preallocate array for optimized results
    Xopt = np.empty((data_size, dim), dtype=np.float32)
    params = []

    for i in range(data_size):
        print('Sample %d/%d' % (i+1, data_size))
        A = 100 * (1 + perturb_pct * np.random.uniform(-1, 1, size=1))
        B = 1 * (1 + perturb_pct * np.random.uniform(-1, 1, size=1))
        shifts = 1 * (1 + perturb_pct * np.random.uniform(-1, 1, size=dim))

        # Create Rosenbrock function
        rosenbrock_func = rosenbrock_function_np(A, B, shifts, dim)

        # Perform PSO
        bounds = [(low, high) for low, high in zip(lb, ub)]
        # xopt, fopt = pso(rosenbrock_func, lb, ub, swarmsize=2000, minfunc=1e-12, maxiter=10000)
        result = differential_evolution(rosenbrock_func, bounds, maxiter=10000, tol=1e-12)
        xopt = result.x
        fopt = result.fun

        # print(xopt, fopt)

        # Store the result
        Xopt[i] = xopt

        # Store the parameters for this function
        params.append({"A": torch.tensor(A), "B": torch.tensor(B), "shifts": torch.tensor(shifts)})

    # Save the data and parameters to a file
    save_data = {"Xopt": torch.tensor(Xopt, dtype=torch.float32), "params": params}
    torch.save(save_data, file_path)
    print(f"Data and parameters saved to {file_path}")

    return Xopt

if __name__ == "__main__":
    # Configuration
    input_dim = 10
    data_size = 5000
    bounds = (-5. * np.ones(input_dim), 5. * np.ones(input_dim))
    pertub_pct = 0.5

    file_name = f"rosenbrock_{input_dim}d_{pertub_pct}pct_optima_data.pt"

    X_opt_train = generate_optima("data/rosenbrock/train/" + file_name,
                    data_size=data_size,
                    dim=input_dim,
                    bounds=bounds,
                    perturb_pct=pertub_pct)

    X_opt_valid = generate_optima("data/rosenbrock/valid/" + file_name,
                    data_size=data_size//5,
                    dim=input_dim,
                    bounds=bounds,
                    perturb_pct=pertub_pct)

    X_opt_test = generate_optima("data/rosenbrock/test/" + file_name,
                    data_size=data_size//5,
                    dim=input_dim,
                    bounds=bounds,
                    perturb_pct=pertub_pct)

    # plot
    plt.figure(figsize=(16,5))
    # plt.subplot(1, 3, 1)
    plt.scatter(X_opt_train[:, 0], X_opt_train[:, 1], label='train')
    plt.scatter(X_opt_valid[:, 0], X_opt_valid[:, 1], label='valid')
    plt.scatter(X_opt_test[:, 0], X_opt_test[:, 1], label='test')

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