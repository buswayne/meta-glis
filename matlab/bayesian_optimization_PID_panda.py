import numpy as np
from panda_robot import panda_robot
from obj_PID_panda import obj_PID_panda
from bayes_opt import BayesianOptimization
import roboticstoolbox as rtb
import matplotlib.pyplot as plt
import imageio

np.random.seed(42)

# Constants
ts = 1e-3
Tsim = 4.0
time = np.arange(0, Tsim, ts)
n_DoFs = 7
friction = np.array([2]*n_DoFs)

# Robot and initial state
Robot = panda_robot()
R_0 = np.eye(3)
x_0 = np.array([0.1, 0.0, 0.5])
x_f = np.array([0.25, 0.0, 0.6])
dx_0 = dx_f = ddx_0 = ddx_f = np.zeros(3)

T_0 = np.eye(4)
T_0[:3, :3] = R_0
T_0[:3, 3] = x_0

q_guess = np.array([-0.7160, -0.5850, 0.3504, -1.5666, 0.2241, -2.1201, -2.8398])
q_0 = Robot.ikine_LM(T_0, q0=q_guess).q

t0 = 0
tf = 2.

# Trajectory interpolation
C = np.vstack([
    [1, t, t**2, t**3, t**4, t**5] for t in [t0, tf]
] + [
    [0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4] for t in [t0, tf]
] + [
    [0, 0, 2, 6*t, 12*t**2, 20*t**3] for t in [t0, tf]
])

xM = np.vstack([x_0, x_f, dx_0, dx_f, ddx_0, ddx_f])
a = np.linalg.solve(C, xM)

x_r = []
dx_r = []
ddx_r = []
q_r = [q_0]
dq_r = [np.zeros(n_DoFs)]
ddq_r = [np.zeros(n_DoFs)]

for t in time:
    if t <= tf:
        xt = np.array([np.polyval(a[::-1, i], t) for i in range(3)])
        dxt = np.array([np.polyval(np.polyder(a[::-1, i]), t) for i in range(3)])
        ddxt = np.array([np.polyval(np.polyder(a[::-1, i], 2), t) for i in range(3)])
        x_r.append(xt)
        dx_r.append(dxt)
        ddx_r.append(ddxt)

        T_i = np.eye(4)
        T_i[:3, :3] = R_0
        T_i[:3, 3] = xt

        q_i = Robot.ikine_LM(T_i, q0=q_r[-1]).q
        q_r.append(q_i)
        dq_r.append((q_i - q_r[-2])/ts)
        ddq_r.append((dq_r[-1] - dq_r[-2])/ts)
    else:
        x_r.append(x_r[-1])
        dx_r.append(np.zeros(3))
        ddx_r.append(np.zeros(3))
        q_r.append(q_r[-1])
        dq_r.append(np.zeros(n_DoFs))
        ddq_r.append(np.zeros(n_DoFs))

# Assemble data for objective
r = dict(q_r=np.array(q_r), dq_r=np.array(dq_r), ddq_r=np.array(ddq_r))
const = dict(Ts=ts, Tsim=Tsim, time=time, n_DoFs=n_DoFs,
             r=r, Robot=Robot, Robot_friction=friction, q_0=q_0, toll_qerr=20*np.pi/180)

# Define bounds for Bayesian optimization
bounds = {}
for i in range(1, 8):
    bounds[f"Kp{i}"] = (0, 300)
    bounds[f"Ki{i}"] = (0, 500)
    bounds[f"Kd{i}"] = (0, 50)

# Wrap objective
def black_box_function(**kwargs):
    return -obj_PID_panda(kwargs, const)  # Minimize cost

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=bounds,
    verbose=2,
    random_state=42
)

optimizer.maximize(
    init_points=10,
    n_iter=90
)

print("Best Parameters:", optimizer.max['params'])

# Run one simulation with best parameters and get joint trajectories
best_params = optimizer.max['params']
cost, q_msr = obj_PID_panda(best_params, const, return_trace=True)
q_r = r['q_r']

q_r = q_r[:len(time)]
q_msr = q_msr[:len(time)]

t = const['time']
q_err = np.rad2deg(q_r - q_msr)

# Plot joint tracking errors (degrees)
plt.figure(figsize=(10, 5))
for i in range(n_DoFs):
    plt.plot(t, q_err[:, i], label=f'q{i+1}')
plt.xlabel("Time [s]")
plt.ylabel("Tracking error [deg]")
plt.title("Joint Position Tracking Errors")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
for i in range(n_DoFs):
    plt.plot(t, np.rad2deg(q_r[:, i]), linestyle='--', label=f'q_r{i+1}')
    plt.plot(t, np.rad2deg(q_msr[:, i]), linestyle='-', label=f'q_msr{i+1}')
plt.xlabel("Time [s]")
plt.ylabel("Joint Angle [deg]")
plt.title("Desired vs Measured Joint Angles")
plt.legend(ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()

# Optionally downsample for faster plotting
q_msr_sampled = q_msr[::20]

# Create GIF (or animation in window)
# anim = Robot.plot(q_msr_sampled, dt=ts, block=False)  # capture animation object
# anim.save('robot_trajectory.gif', writer='pillow', fps=int(1/ts))
Robot.plot(q_msr_sampled, dt=ts)