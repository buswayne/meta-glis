import matplotlib

import numpy as np

import matplotlib.pyplot as plt
from objective_function import f_x
import pickle
import os

import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == '__main__':
    matplotlib.rc('text', usetex=False)

    algo = 'GLIS' # GLIS or BO
    machine = 'PC' # PC or PI
    eps_calc = 1.0
    iter_max_plot = 50

    plt.close('all')

    experiment = 800
    with open(os.path.join('results/params', f'{experiment:04}.pkl'), 'rb') as f:
        params = pickle.load(f)

    res_filename = f"results/experiment_{experiment:04}_res_slower{eps_calc:.0f}_100iter_{algo}_{machine}.pkl"
    results = pickle.load(open(res_filename, "rb"))

    decoder_res_filename = f"results/decoder/experiment_{experiment:04}_res_slower{eps_calc:.0f}_100iter_{algo}_{machine}.pkl"
    decoder_results = pickle.load(open(decoder_res_filename, "rb"))


    # In[]
    FIG_FOLDER = 'fig'
    if not os.path.isdir(FIG_FOLDER):
        os.makedirs(FIG_FOLDER)

    # In[Iteration plot]

    J = results['J_sample']
    J_decoder = decoder_results['J_sample']


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


