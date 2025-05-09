import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import differential_evolution
sys.path.append('..')
from benchmarks import rosenbrock_function_np

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams['axes.labelsize']=12
plt.rcParams['xtick.labelsize']=10
plt.rcParams['ytick.labelsize']=10
plt.rcParams['axes.grid']=True
plt.rcParams['axes.xmargin']=0

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
        # A = 100 * (1 + perturb_pct * np.random.uniform(-1, 1, size=1))
        # B = 1 * (1 + perturb_pct * np.random.uniform(-1, 1, size=1))
        # shifts = 1 * (1 + perturb_pct * np.random.uniform(-1, 1, size=dim))
        A = np.random.uniform(10, 1000, size=1)
        B = np.random.uniform(0.1, 10, size=1)
        shifts = np.random.uniform(0.1, 10, size=dim)

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

    torch.save(save_data, file_path)
    print(f"Data and parameters saved to {file_path}")

    return Xopt, fopt, X, f

if __name__ == "__main__":
    # Configuration
    input_dim = 20
    data_size = 500
    bounds = (-2.5 * np.ones(input_dim), 2.5 * np.ones(input_dim))
    perturb_pct = 1000

    file_name = f"rosenbrock_{input_dim}d_{perturb_pct}pct_optima_data.pt"

    (Xopt_train, fopt_train, X_train, f_train) = generate_optima(
                                                "../../data/rosenbrock/with_exploration/train/" + file_name,
                                                 data_size=data_size,
                                                 dim=input_dim,
                                                 bounds=bounds,
                                                 perturb_pct=perturb_pct)


    (Xopt_valid, fopt_valid, X_valid, f_valid) = generate_optima(
                                                 "../../data/rosenbrock/with_exploration/valid/" + file_name,
                                                 data_size=data_size//5,
                                                 dim=input_dim,
                                                 bounds=bounds,
                                                 perturb_pct=perturb_pct)

    (Xopt_test, fopt_test, X_test, f_test) = generate_optima(
                                                 "../../data/rosenbrock/with_exploration/test/" + file_name,
                                                 data_size=data_size//5,
                                                 dim=input_dim,
                                                 bounds=bounds,
                                                 perturb_pct=perturb_pct)

    # plot

    # all_points = np.concatenate(X_train, axis=0)  # Shape: (total_points, dim)
    # all_values = np.concatenate(f_train, axis=0)  # Shape: (total_points,)
    #
    # for i in range(input_dim):
    #     for j in range(input_dim):
    #         plt.figure(figsize=(5, 5))
    #
    #         # Create a hexbin plot
    #         plt.hexbin(all_points[:, i], all_points[:, j], C=all_values, gridsize=50, cmap='gray', mincnt=1)
    #
    #         plt.scatter(Xopt_train[:, i], Xopt_train[:, j], c='red', s=25, label='Optimal Points', edgecolors='black')
    #
    #         plt.axis("equal")
    #
    #         plt.title(f'dim {i} vs dim {j}')
    #         plt.legend()
    #
    #         plt.show()

    idx_fixed = [(0, 1),(1,2),(2,3),(3,4),(4,5),(5,6)]

    for i in range(6):
        # first_idx, second_idx = np.random.choice(true_optima.shape[-1], size=2, replace=False)
        first_idx, second_idx = idx_fixed[i]

        plt.figure(figsize=(5, 5))
        plt.scatter(X_train[:, :, first_idx], X_train[:, :, second_idx], label='_nolegend_', s=5, alpha=0.05, color='tab:blue')
        plt.scatter(Xopt_train[:, first_idx], Xopt_train[:, second_idx], label='_nolegend_', marker='*', s=20, alpha=0.5, color='tab:red')

        # Add high-alpha points just for the legend
        plt.scatter([], [], label='$X^{*}$', s=5, alpha=1, color='tab:blue')
        plt.scatter([], [], label='$X^{\star}$', s=20, marker='*', alpha=1, color='tab:red')

        plt.xlabel(f'$x_{{{first_idx}}}$')
        plt.ylabel(f'$x_{{{second_idx}}}$')
        plt.legend()
        plt.axis('equal')
        plt.xlim([-2.5, 2.5])
        plt.ylim([-2.5, 2.5])
        plt.show()