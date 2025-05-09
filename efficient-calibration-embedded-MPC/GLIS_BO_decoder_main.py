import os
import matplotlib
import sys

if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use("Agg")
import idwgopt.idwgopt
import numpy as np
from pendulum_MPC_sim import simulate_pendulum_MPC, get_parameter
import matplotlib.pyplot as plt
from objective_function import f_x, get_simoptions_x
import pickle
import time
import numba as nb

import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from numerical_examples.models import Autoencoder, AutoencoderTrivial

def main(prefix, params):

    np.random.seed(0)
    # initial random points

    # optimization parameters
    # Run the optimization
    # n_init = 10 changed to 2*n_var (GLIS default choice)
    max_iter = 30  # evaluation budget
    max_time = np.inf  # time budget
    eps = 0.0  # Minimum allows distance between the las two observations
    eps_calc = 1.0

    method = "GLIS" # GLIS or BO
    machine = 'PC' # PC or PI

    # dict_x0 = {
    #     'QDu_scale': 0.1,
    #     'Qy11': 0.1,
    #     'Qy22': 0.9,
    #     'Np': 40,
    #     'Nc_perc': 1.0,
    #     'Ts_MPC': 25e-3,
    #     'QP_eps_abs_log': -3,
    #     'QP_eps_rel_log': -2,
    #     'Q_kal_11': 0.1,
    #     'Q_kal_22': 0.4,
    #     'Q_kal_33': 0.1,
    #     'Q_kal_44': 0.4,
    #     'R_kal_11': 0.5,
    #     'R_kal_22': 0.5
    # }

    # dict_context = {}
    #
    # x0 = dict_to_x(dict_x0)

    # x_bounds = [
    #     {'name': 'QDu_scale', 'type': 'continuous', 'domain': (1e-16, 1)},  # 0
    #     {'name': 'Qy11', 'type': 'continuous', 'domain': (1e-16, 1)},  # 1
    #     {'name': 'Qy22', 'type': 'continuous', 'domain': (1e-16, 1)},  # 2
    #     {'name': 'Np', 'type': 'continuous', 'domain': (5, 300)},  # 3
    #     {'name': 'Nc_perc', 'type': 'continuous', 'domain': (0.3, 1)},  # 4
    #     {'name': 'Ts_MPC', 'type': 'continuous', 'domain': (1e-3, 50e-3)},  # 5
    #     {'name': 'QP_eps_abs_log', 'type': 'continuous', 'domain': (-7, -1)},  # 6
    #     {'name': 'QP_eps_rel_log', 'type': 'continuous', 'domain': (-7, -1)},  # 7
    #     {'name': 'Q_kal_11', 'type': 'continuous', 'domain': (1e-16, 1)},  # 8
    #     {'name': 'Q_kal_22', 'type': 'continuous', 'domain': (1e-16, 1)},  # 9
    #     {'name': 'Q_kal_33', 'type': 'continuous', 'domain': (1e-16, 1)},  # 10
    #     {'name': 'Q_kal_44', 'type': 'continuous', 'domain': (1e-16, 1)},  # 11
    #     {'name': 'R_kal_11', 'type': 'continuous', 'domain': (1e-16, 1)},  # 12
    #     {'name': 'R_kal_22', 'type': 'continuous', 'domain': (1e-16, 1)},  # 13
    # ]

    x_bounds = [
        (1e-16, 1),  # 0
        (1e-16, 1),  # 1
        (1e-16, 1),  # 2
        (5, 300),  # 3
        (0.3, 1),  # 4
        (1e-3, 50e-3),  # 5
        (-7, -1),  # 6
        (-7, -1),  # 7
        (1e-16, 1),  # 8
        (1e-16, 1),  # 9
        (1e-16, 1),  # 10
        (1e-16, 1),  # 11
        (1e-16, 1),  # 12
        (1e-16, 1),  # 13
    ]

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 14
    latent_dim = 3
    bounds = (0.0*np.ones(latent_dim), 1.0*np.ones(latent_dim))

    model_path = f"../out/mpc/model_{input_dim}d_{latent_dim}l_500iter_exp_decay_alpha_05_trivial.pt"
    autoencoder = AutoencoderTrivial(input_dim=input_dim, latent_dim=latent_dim, out_bounds=x_bounds)

    try:
        checkpoint = torch.load(model_path, weights_only=True)
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        autoencoder.load_state_dict(state_dict)
    except:
        "Could not load pretrained model"
        try:
            state_dict = torch.load(model_path, weights_only=True)
            autoencoder.load_state_dict(state_dict)
        except:
            "Could not load pretrained model"
    autoencoder.to(device)
    autoencoder.eval()

    # Extract the decoder from the Autoencoder
    decoder = autoencoder.decoder

    constraints = [
        # {'name': 'min_time', 'constraint': '-x[:,4]*x[:,6] + 0.1'}, # prediction horizon in seconds large enough
    ]

    n_var = latent_dim
    n_init = 2 * n_var
    # n_init = 1

    def f_x_calc(z):
        with torch.no_grad():
            x = decoder(torch.tensor(z, dtype=torch.float32, device=device)).cpu().numpy()
        return f_x(x, params, eps_calc)


    # feasible_region = GPyOpt.Design_space(space=bounds,
    #                                       constraints=constraints)  # , constraints=constraints_context)
    # #    unfeasible_region = GPyOpt.Design_space(space=bounds)
    # X_init = GPyOpt.experiment_design.initial_design('random', feasible_region, n_init)

    time_optimization_start = time.perf_counter()
    # if method == "BO":
    #     myBopt = GPyOpt.methods.BayesianOptimization(f_x_calc,
    #                                                  X=X_init,
    #                                                  domain=bounds,
    #                                                  model_type='GP',
    #                                                  acquisition_type='EI',
    #                                                  normalize_Y=True,
    #                                                  exact_feval=False)
    #
    #     myBopt.run_optimization(max_iter=max_iter - n_init, max_time=max_time, eps=eps, verbosity=False)
    #
    #     x_opt = myBopt.x_opt
    #     J_opt = myBopt.fx_opt
    #     X_sample = myBopt.X
    #     J_sample = myBopt.Y
    #     idx_opt = np.argmin(J_sample)

    if method == "GLIS":

        # IDWGOPT initialization
        nvars = latent_dim
        lb = np.zeros((nvars, 1)).flatten("c")
        ub = lb.copy()
        for i in range(0, nvars):
            lb[i] = bounds[0][i]
            ub[i] = bounds[1][i]

        problem = idwgopt.idwgopt.default(nvars)
        problem["nsamp"] = n_init
        problem["maxevals"] = max_iter
        #        problem["g"] = lambda x: np.array([-x[5]*x[7]+0.1])
        problem["lb"] = lb
        problem["ub"] = ub
        problem["f"] = f_x_calc
        problem["useRBF"] = 1  # use Radial Basis Functions
        # problem["useRBF"] = 0 # Inverse Distance Weighting
        if problem["useRBF"]:
            epsil = .5


            @nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:])], '(n),(n)->()')
            def fun_rbf(x1, x2, res):
                res[0] = 1 / (1 + epsil ** 2 * np.sum((x1 - x2) ** 2))


            problem['rbf'] = fun_rbf

        problem["alpha"] = 1
        problem["delta"] = .5

        problem["svdtol"] = 1e-6
        # problem["globoptsol"] = "direct"
        problem["globoptsol"] = "pswarm"
        problem["display"] = 0

        problem["scalevars"] = 1

        problem["constraint_penalty"] = 1e3
        problem["feasible_sampling"] = False

        problem["shrink_range"] = 0  # 0 = don't shrink lb/ub

        tic = time.perf_counter()
        out = idwgopt.idwgopt.solve(problem)
        toc = time.perf_counter()

        x_opt = out["xopt"]
        J_opt = out["fopt"]
        J_sample = out["F"]
        X_sample = out["X"]
        idx_opt = np.argmin(J_sample, axis=0)

    time_optimization = time.perf_counter() - time_optimization_start

    print(f"J_best_val: {J_opt.item():.3f}")

    # In[Re-simulate with the optimal point]

    with torch.no_grad():
        simopt = get_simoptions_x(decoder(torch.tensor(x_opt, dtype=torch.float32, device=device)).cpu().numpy())

    # Here we pass the params
    simout = simulate_pendulum_MPC(simopt, params)

    tsim = simout['t']
    xsim = simout['x']
    usim = simout['u']

    x_ref = simout['x_ref']
    uref = get_parameter({}, 'uref')

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    axes[0].plot(tsim, xsim[:, 0], "k", label='p')
    axes[0].plot(tsim, x_ref[:, 0], "r--", label="p_ref")
    axes[0].set_title("Position (m)")

    axes[1].plot(tsim, xsim[:, 2] * 360 / 2 / np.pi, label="phi")
    axes[1].plot(tsim, x_ref[:, 2] * 360 / 2 / np.pi, "r--", label="phi_ref")
    axes[1].set_title("Angle (deg)")

    axes[2].plot(tsim, usim[:, 0], label="u")
    axes[2].plot(tsim, uref * np.ones(np.shape(tsim)), "r--", label="u_ref")
    axes[2].set_title("Force (N)")

    for ax in axes:
        ax.grid(True)
        ax.legend()

    J_best_curr = np.zeros(np.shape(J_sample))
    J_best_val = J_sample[0]
    iter_best_val = 0

    fig, axes = plt.subplots(1, 1, figsize=(8, 5))
    axes = [axes]
    for i in range(len(J_best_curr)):
        if J_sample[i] < J_best_val:
            J_best_val = J_sample[i]
            iter_best_val = i
        J_best_curr[i] = J_best_val

    N = len(J_sample)
    iter = np.arange(0, N, dtype=int)
    axes[0].plot(iter, J_sample, 'k*', label='Current test boint')
    axes[0].plot(iter, J_best_curr, 'g', label='Current best point')
    axes[0].plot(iter_best_val, J_best_val, 's', label='Overall best point')

    for ax in axes:
        ax.grid(True)
        ax.legend()

    # In[Re-evaluate optimal controller]
    with torch.no_grad():
        J_opt = f_x(decoder(torch.tensor(x_opt, dtype=torch.float32, device=device)).cpu().numpy(), params)
    print(J_opt)

    plt.show()

    # In[Store results]

    result = {}
    # if method == 'BO':
    #     myBopt.f = None  # hack to solve the issues pickling the myBopt object
    #     myBopt.objective = None
    #
    #     results = {'X_sample': myBopt.X, 'J_sample': myBopt.Y,
    #                'idx_opt': idx_opt, 'x_opt': x_opt, 'J_opt': J_opt,
    #                'eps_calc': eps_calc,
    #                'time_iter': myBopt.time_iter, 'time_f_eval': myBopt.time_f_eval,
    #                'time_opt_acquisition': myBopt.time_opt_acquisition, 'time_fit_surrogate': myBopt.time_fit_surrogate,
    #                'myBopt': myBopt, 'method': method,
    #                }

    if method == 'GLIS':
        results = {'X_sample': X_sample, 'J_sample': J_sample,
                   'idx_opt': idx_opt, 'x_opt': x_opt, 'J_opt': J_opt,
                   'eps_calc': eps_calc,
                   'time_iter': out['time_iter'], 'time_f_eval': out['time_f_eval'],
                   'time_opt_acquisition': out['time_opt_acquisition'], 'time_fit_surrogate': out['time_fit_surrogate'],
                   'out': out, 'method': method,
                   }

    res_filename = f"results/decoder/experiment_{prefix}_res_slower{eps_calc:.0f}_{max_iter:.0f}iter_{method}_{machine}_trivial.pkl"

    with open(res_filename, "wb") as file:
        pickle.dump(results, file)
        print('Results saved')

if __name__ == '__main__':

    experiment_name = "0001"
    perturb_pct = 0.2

    params = {
        "M": 0.5 * (1 + perturb_pct * np.random.uniform(low=-1.0, high=1.0)),
        "m": 0.3 * (1 + perturb_pct * np.random.uniform(low=-1.0, high=1.0)),
        "b": 0.05 * (1 + perturb_pct * np.random.uniform(low=-1.0, high=1.0)),
        "ftheta": 0.15 * (1 + perturb_pct * np.random.uniform(low=-1.0, high=1.0)),
        "l": 0.4 * (1 + perturb_pct * np.random.uniform(low=-1.0, high=1.0)),
    }

    main(experiment_name, params)


