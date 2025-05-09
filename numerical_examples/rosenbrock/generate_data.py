import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import differential_evolution

from benchmarks import rosenbrock_function_np, quadratic_function_np, exact_solution

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams['axes.labelsize']=12
plt.rcParams['xtick.labelsize']=10
plt.rcParams['ytick.labelsize']=10
plt.rcParams['axes.grid']=True
plt.rcParams['axes.xmargin']=0

def generate_optima(file_path, func_name='rosenbrock', data_size=10000, dim=2, bounds=(-30, 30), perturb_pct=0, base=None):
    lb, ub = bounds

    # Preallocate array for optimized results
    Xopt = np.empty((data_size, dim), dtype=np.float32)
    params = []

    for i in range(data_size):
        print('Sample %d/%d' % (i+1, data_size))

        if func_name == "rosenbrock":
            A = 100 * (1 + perturb_pct * np.random.uniform(-1, 1, size=1))
            B = 1 * (1 + perturb_pct * np.random.uniform(-1, 1, size=1))
            shifts = 1 * (1 + perturb_pct * np.random.uniform(-1, 1, size=dim))
            func = rosenbrock_function_np(A=A, B=B, shifts=shifts, dim=dim)

        elif func_name == "quadratic":
            A = 1 * (1 + perturb_pct * np.random.uniform(-1, 1, size=1)) * (base + base.T) / 2  # Symmetrize and scale
            b = 1 * (1 + perturb_pct * np.random.uniform(-1, 1, size=dim))  # Randomized vector, scaled
            func = quadratic_function_np(A=A, b=b, dim=dim)
        else:
            raise ValueError("Invalid function name. Choose 'rosenbrock' or 'quadratic'.")

        if func_name == "quadratic":
            # Exact solution and function value
            xopt, fopt = exact_solution(A, b)
            # Store the parameters for this function
            params.append({"A": torch.tensor(A), "b": torch.tensor(b)})
        else:
            # Perform PSO
            bounds = [(low, high) for low, high in zip(lb, ub)]
            # xopt, fopt = pso(rosenbrock_func, lb, ub, swarmsize=2000, minfunc=1e-12, maxiter=10000)
            result = differential_evolution(func, bounds, maxiter=10000, tol=1e-12)
            xopt = result.x
            fopt = result.fun
            # Store the parameters for this function
            params.append({"A": torch.tensor(A), "B": torch.tensor(B), "shifts": torch.tensor(shifts)})

        print(fopt)

        # Store the result
        Xopt[i] = xopt



    # Save the data and parameters to a file
    save_data = {"Xopt": torch.tensor(Xopt, dtype=torch.float32), "params": params}
    torch.save(save_data, file_path)
    print(f"Data and parameters saved to {file_path}")

    return Xopt

def plot_3d_quadratic(ax, A, b, bounds):
    # Generate a meshgrid for the input space
    x = np.linspace(bounds[0], bounds[1], 50)
    y = np.linspace(bounds[0], bounds[1], 50)
    X, Y = np.meshgrid(x, y)

    # Calculate the function values over the meshgrid
    Z = np.array([[quadratic_function_np(A, b, dim=2)([xi, yi]) for xi, yi in zip(X_row, Y_row)]
                  for X_row, Y_row in zip(X, Y)])

    xopt, fopt = exact_solution(A, b)

    # Scatter the points (optima) on the surface
    ax.scatter(xopt[0], xopt[1], fopt, color='r',
               label='train', alpha=0.7)

    ax.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.1)

    # Create contour plot
    # ax.contour3D(X, Y, Z, 50, cmap='viridis')

    return ax

def plot_2d_quadratic_contour(ax, A, b, bounds):
    """
    Plot 2D contour of the quadratic function.
    Args:
    - A (np.ndarray): The quadratic coefficient matrix (shape: dim x dim).
    - b (np.ndarray): The linear coefficient vector (shape: dim).
    - bounds (tuple): The bounds for the x and y axes (min, max).
    - ax (matplotlib.axes.Axes): Axes to plot on (optional).
    """
    # Generate a meshgrid for the input space
    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x, y)

    # Calculate the function values over the meshgrid
    Z = np.array([[quadratic_function_np(A, b, dim=2)([xi, yi]) for xi, yi in zip(X_row, Y_row)]
                  for X_row, Y_row in zip(X, Y)])

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    # Create the contour plot
    cp = ax.contour(X, Y, Z, 20, cmap='coolwarm', alpha=0.5)
    # ax.clabel(cp, inline=True, fontsize=10)

    # Exact solution and function value (optima)
    xopt, fopt = exact_solution(A, b)
    ax.plot(xopt[0], xopt[1], 'r*', label="Optima", markersize=7)

    # ax.legend()
    return ax

if __name__ == "__main__":
    # Configuration
    func_name = "quadratic"
    input_dim = 20
    data_size = 100000
    bounds = (-3. * np.ones(input_dim), 3. * np.ones(input_dim))
    perturb_pct = 0.5
    # For the quadratic
    # base = np.random.rand(input_dim, input_dim)
    base = np.eye(input_dim)

    # Modify the diagonal values to create anisotropic scaling
    for i in range(5):
        base[i, i] = np.random.uniform(5.0, 10.0)  # Stretch along the dimensions

    file_name = f"{func_name}_{input_dim}d_{perturb_pct}pct_optima_data.pt"

    X_opt_train = generate_optima(f"data/{func_name}/train/" + file_name,
                    func_name=func_name,
                    data_size=data_size,
                    dim=input_dim,
                    bounds=bounds,
                    perturb_pct=perturb_pct,
                    base=base)

    X_opt_valid = generate_optima(f"data/{func_name}/valid/" + file_name,
                    func_name=func_name,
                    data_size=data_size//5,
                    dim=input_dim,
                    bounds=bounds,
                    perturb_pct=perturb_pct,
                    base=base)

    X_opt_test = generate_optima(f"data/{func_name}/test/" + file_name,
                    func_name=func_name,
                    data_size=data_size//5,
                    dim=input_dim,
                    bounds=bounds,
                    perturb_pct=perturb_pct,
                    base=base)

    # Plot
    plt.figure(figsize=(5, 5))

    # Subplot 1
    ax1 = plt.subplot(1, 1, 1)
    ax1.scatter(X_opt_train[:, 0], X_opt_train[:, 1], label='train')
    ax1.scatter(X_opt_valid[:, 0], X_opt_valid[:, 1], label='valid')
    ax1.scatter(X_opt_test[:, 0], X_opt_test[:, 1], label='test')
    ax1.set_aspect('equal')  # Set aspect ratio to be equal
    ax1.legend()

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

    # 3D plot of the quadratic function
    # Create the 3D plot
    # fig = plt.figure(figsize=(2.5, 2.5))
    # ax = fig.add_subplot(111)
    #
    # for i in range(data_size):
    #     A = 1 * (1 + perturb_pct * np.random.uniform(-1, 1, size=1)) * (base + base.T) / 2  # Symmetrize and scale
    #     b = 1 * (1 + perturb_pct * np.random.uniform(-1, 1, size=input_dim))  # Randomized vector, scaled
    #     # ax = plot_3d_quadratic(ax, A, b, bounds)
    #     ax = plot_2d_quadratic_contour(ax, A, b, bounds)
    #
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_aspect('equal')  # Set aspect ratio to be equal
    # # ax.set_zlabel('Function Value')
    # ax.set_xlabel('$x_1$')
    # ax.set_ylabel('$x_2$')
    # plt.tight_layout()
    #
    # plt.savefig('quadratic_contour_plot.pdf', format='pdf')
    #
    # plt.show()