import matplotlib

import numpy as np
from pendulum_MPC_sim import simulate_pendulum_MPC, get_parameter

import matplotlib.pyplot as plt
from objective_function import f_x, get_simoptions_x
from pendulum_model import RAD_TO_DEG
import pickle
import os
from scipy.interpolate import interp1d

import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import Autoencoder

if __name__ == '__main__':

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
    latent_dim = 5
    bounds = (0.0*np.ones(latent_dim), 1.0*np.ones(latent_dim))

    model_path = f"../out/mpc/model_{input_dim}d_{latent_dim}l.pt"
    autoencoder = Autoencoder(input_dim=input_dim, latent_dim=latent_dim, out_bounds=x_bounds)

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


    matplotlib.rc('text', usetex=False)

    algo = 'GLIS' # GLIS or BO
    machine = 'PC' # PC or PI
    eps_calc = 1.0
    iter_max_plot = 50

    plt.close('all')

    experiment = 800
    with open(os.path.join('results/params', f'{experiment:04}.pkl'), 'rb') as f:
        params = pickle.load(f)

    res_filename = f"results/decoder/experiment_{experiment:04}_res_slower{eps_calc:.0f}_100iter_{algo}_{machine}.pkl"
    results = pickle.load(open(res_filename, "rb"))


    # In[]
    FIG_FOLDER = 'fig'
    if not os.path.isdir(FIG_FOLDER):
        os.makedirs(FIG_FOLDER)

    # In[Re-simulate]

    ## Re-simulate with the optimal point
    z_opt = results['x_opt']

    with torch.no_grad():
        x_opt = decoder(torch.tensor(z_opt, dtype=torch.float32, device=device)).cpu().numpy()
    simopt = get_simoptions_x(x_opt)
    t_ref_vec = np.array([0.0, 5.0, 10.0,  13.0,   20.0, 22.0,  25.0, 30.0, 35.0,  40.0, 100.0])
    p_ref_vec = np.array([0.0, 0.4,  0.0,   0.9,    0.9,  0.4,   0.4,  0.4,  0.0,   0.0, 0.0])
    rp_fun = interp1d(t_ref_vec, p_ref_vec, kind='linear')
    def xref_fun_def(t):
        return np.array([rp_fun(t), 0.0, 0.0, 0.0])
    simopt['xref_fun'] = xref_fun_def


    simout = simulate_pendulum_MPC(simopt, params)

    t = simout['t']
    x = simout['x']
    u = simout['u']
    y = simout['y']
    y_meas = simout['y_meas']
    x_ref = simout['x_ref']
    x_MPC_pred = simout['x_MPC_pred']
    x_fast = simout['x_fast']
    x_ref_fast = simout['x_ref_fast']
    y_ref = x_ref[:, [0, 2]]  # on-line predictions from the Kalman Filter
    uref = get_parameter({}, 'uref')
    u_fast = simout['u_fast']

    t_int = simout['t_int_fast']
    t_fast = simout['t_fast']
    t_calc = simout['t_calc']

    fig, axes = plt.subplots(3, 1, figsize=(8, 6))
#    axes[0].plot(t, y_meas[:, 0], "r", label='p_meas')
    axes[0].plot(t_fast, x_fast[:, 0], "k", label='$p$')
    axes[0].plot(t, y_ref[:, 0], "r--", label="$p^{\mathrm{ref}}$", linewidth=2)
    axes[0].set_ylim(-0.2, 1.0)
    axes[0].set_ylabel("Position (m)")

#    axes[1].plot(t, y_meas[:, 1] * RAD_TO_DEG, "r", label='phi_meas')
    axes[1].plot(t_fast, x_fast[:, 2] * RAD_TO_DEG, 'k', label="$\phi$")
    idx_pred = 0
    axes[1].set_ylim(-12, 12)
    axes[1].set_ylabel("Angle (deg)")

    axes[2].plot(t, u[:, 0], 'k', label="$u$")
    #axes[2].plot(t, uref * np.ones(np.shape(t)), "r--", label="u_ref")
    axes[2].set_ylim(-8, 8)
    axes[2].set_ylabel("Force (N)")
    axes[2].set_xlabel("Simulation time (s)")

    for ax in axes:
        ax.grid(True)
        ax.legend(loc='upper right')

    fig_name = f"BEST_{algo}_{machine}.pdf"
    fig_path = os.path.join(FIG_FOLDER, fig_name)
    fig.savefig(fig_path, bbox_inches='tight')

    plt.show()

    # MPC time check
    # In[MPC computation time ]
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(t, y_meas[:, 0], "r", label='$p_{meas}$')
    axes[0].plot(t_fast, x_fast[:, 0], "k", label='$p$')
    axes[0].step(t, y_ref[:, 0], "k--", where='post', label="$p_{ref}$")
    axes[0].set_ylim(-0.2, 1.0)
    axes[0].set_xlabel("Simulation time (s)")
    axes[0].set_ylabel("Position (m)")

    axes[1].step(t, t_calc[:, 0] * 1e3, "b", where='post', label='$T_{MPC}$')
    axes[1].set_xlabel("Simulation time (s)")
    axes[1].set_ylabel("MPC time (ms)")
    axes[1].set_ylim(0, 40)
    axes[2].step(t_fast[1:], t_int[1:, 0] * 1e3, "b", where='post', label='$T_{ODE}$')
    axes[2].set_xlabel("Simulation time (s)")
    axes[2].set_ylabel("ODE time (ms)")
    axes[2].set_ylim(0, 2)
    axes[3].step(t, u[:, 0], where='post', label="$F$")
    axes[3].step(t_fast, u_fast[:, 0], where='post', label="$F_{d}$")
    axes[3].set_xlabel("Simulation time (s)")
    axes[3].set_ylabel("Force (N)")

    for ax in axes:
        ax.grid(True)
        ax.legend()

    fig_name = f"MPC_CPUTIME_{algo}_{machine}.pdf"
    fig_path = os.path.join(FIG_FOLDER, fig_name)
    fig.savefig(fig_path, bbox_inches='tight')

    plt.show()

    # In[Iteration plot]

    J = results['J_sample']
    Ts_MPC = simout['Ts_MPC']

    J_best_curr = np.zeros(np.shape(J))
    J_best_val = J[0]
    iter_best_val = 0

    fig, axes = plt.subplots(1, 1, figsize=(6, 4))
    axes = [axes]
    for i in range(len(J_best_curr)):
        if J[i] < J_best_val:
            J_best_val = J[i]
            iter_best_val = i
        J_best_curr[i] = J_best_val

    N = len(J)
    iter = np.arange(1, N + 1, dtype=int)
    axes[0].plot(iter, J, 'k*', label='Current test point')
    #    axes[0].plot(iter, Y_best_curr, 'r', label='Current best point')
    axes[0].plot(iter, J_best_val * np.ones(J.shape), '-', label='Overall best point', color='red')
    axes[0].set_xlabel("Iteration index $n$ (-)")
    axes[0].set_ylabel(r"Performance cost $\tilde {J}^{\mathrm{cl}}$")

    for ax in axes:
        ax.grid(True)
        ax.legend(loc='upper right')

    axes[0].set_xlim((0, iter_max_plot))
    axes[0].set_ylim((-1, 25))

    fig_name = f"ITER_{algo}_{machine}_{iter_max_plot:.0f}.pdf"
    fig_path = os.path.join(FIG_FOLDER, fig_name)
    fig.savefig(fig_path, bbox_inches='tight')

    plt.show()

    # In[Recompute optimum]
    #print(Y_best_val)

    with torch.no_grad():
        x_opt = decoder(torch.tensor(z_opt, dtype=torch.float32, device=device)).cpu().numpy()
    J_opt = f_x(x_opt, params, eps_calc=results['eps_calc'])
    #print(J_opt)

    # In[Optimization computation time]

    t_unknown = results['time_iter'] - (
                results['time_f_eval'] + results['time_opt_acquisition'] + results['time_fit_surrogate'])
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.step(iter, results['time_iter'], 'k', where='post', label='Total')
    ax.step(iter, results['time_f_eval'], 'r', where='post', label='Eval')
    ax.step(iter, results['time_opt_acquisition'], 'y', where='post', label='Opt')
    ax.step(iter, results['time_fit_surrogate'], 'g', where='post', label='Fit')
    ax.grid(True)
    ax.legend()

    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.step(iter, np.cumsum(results['time_iter']), 'k', where='post', label='Total')
    ax.step(iter, np.cumsum(results['time_f_eval']), 'r', where='post', label='Function evaluation')
    ax.step(iter, np.cumsum(results['time_fit_surrogate']), 'g', where='post', label='Surrogate fitting')
    ax.step(iter, np.cumsum(results['time_opt_acquisition']), 'y', where='post', label='Surrogate optimization')
    # ax.step(iter, np.cumsum(t_unknown), 'g', where='post', label='Unknown')
    ax.set_xlabel("Iteration index i (-)")
    ax.set_ylabel("Comulative computational time (s)")
    ax.grid(True)
    ax.legend()

    fig_name = f"COMPUTATION_{algo}_{machine}.pdf"
    fig_path = os.path.join(FIG_FOLDER, fig_name)
    fig.savefig(fig_path, bbox_inches='tight')

    plt.show()

    residual_time = np.sum(results['time_iter']) - np.sum(results['time_f_eval']) - np.sum(
        results['time_opt_acquisition']) - np.sum(results['time_fit_surrogate'])
