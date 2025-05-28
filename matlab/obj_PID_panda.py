import numpy as np
from panda_robot import panda_robot


def obj_PID_panda(x_var, const, return_trace=False):
    sim_var = const.copy()
    sim_var.update(x_var)

    Kp = np.diag([sim_var[f"Kp{i+1}"] for i in range(7)])
    Ki = np.diag([sim_var[f"Ki{i+1}"] for i in range(7)])
    Kd = np.diag([sim_var[f"Kd{i+1}"] for i in range(7)])

    Robot = sim_var['Robot']
    Robot_eval = panda_robot()

    Ts = sim_var['Ts']
    t = sim_var['time']
    n_DoFs = sim_var['n_DoFs']
    q_r = sim_var['r']['q_r']
    dq_r = sim_var['r']['dq_r']
    ddq_r = sim_var['r']['ddq_r']
    f = sim_var['Robot_friction']

    q_msr = np.zeros_like(q_r)
    dq_msr = np.zeros_like(dq_r)
    ddq_msr = np.zeros_like(ddq_r)
    ierr_m = np.zeros_like(q_r)

    q_msr[0, :] = q_r[0, :]
    ierr_m[0, :] = 0

    qerr = np.zeros_like(q_r)
    dqerr = np.zeros_like(dq_r)
    ddqerr = np.zeros_like(ddq_r)

    B_eval = np.zeros((len(t), n_DoFs, n_DoFs))
    g_eval = np.zeros((len(t), n_DoFs))

    penalty = np.zeros(n_DoFs)

    toll_qerr = sim_var['toll_qerr']
    exit_flag = False
    jj = 0

    while jj < len(t) - 1 and not exit_flag:
        jj += 1

        B = Robot.inertia(q_msr[jj - 1, :])
        g = Robot.gravload(q_msr[jj - 1, :])

        B_eval[jj] = Robot_eval.inertia(q_msr[jj - 1, :])
        g_eval[jj] = Robot_eval.gravload(q_msr[jj - 1, :])

        tau_l = f * dq_msr[jj - 1, :] + g
        tau_PID = B_eval[jj] @ (Kp @ (q_r[jj, :] - q_msr[jj - 1, :]) - Kd @ dq_msr[jj - 1, :] + Ki @ ierr_m[jj - 1, :])
        tau_comp = f * dq_msr[jj - 1, :] + g_eval[jj]

        ddq_msr[jj, :] = np.linalg.solve(B, -tau_l + tau_PID + tau_comp)
        dq_msr[jj, :] = dq_msr[jj - 1, :] + ddq_msr[jj, :] * Ts
        q_msr[jj, :] = q_msr[jj - 1, :] + dq_msr[jj, :] * Ts

        ierr_m[jj, :] = ierr_m[jj - 1, :] + (q_r[jj, :] - q_msr[jj, :]) * Ts

        qerr[jj, :] = q_r[jj, :] - q_msr[jj, :]
        dqerr[jj, :] = dq_r[jj, :] - dq_msr[jj, :]
        ddqerr[jj, :] = ddq_r[jj, :] - ddq_msr[jj, :]

        for k in range(n_DoFs):
            if np.isnan(q_msr[jj, k]) or abs(qerr[jj, k]) > toll_qerr:
                penalty[k] = 1e5 * np.exp(-t[jj])
                exit_flag = True
                break

    if jj == len(t) - 1:
        J = np.sum(penalty)
        J += np.sum(np.sqrt(np.mean(qerr**2, axis=0)))
        J += np.sum(np.std(qerr, axis=0))
        J += np.sum(np.sqrt(np.mean(dqerr**2, axis=0)))
        J += np.sum(np.std(dqerr, axis=0))
        J += np.sum(np.max(np.abs(qerr), axis=0))
        J += np.sum(np.abs(np.mean(qerr[-50:, :], axis=0)))
        J += np.sum(np.max(np.abs(dqerr), axis=0))
        J += np.sum(np.abs(np.mean(dqerr[-50:, :], axis=0)))
    else:
        J = np.sum(penalty)

    if return_trace:
        return J, np.array(q_msr)
    else:
        return J