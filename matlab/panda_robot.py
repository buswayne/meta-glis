import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH


def panda_robot():
    mm = 1e-3

    d = np.array([333, 0, 316, 0, 384, 0, 107]) * mm
    a = np.array([0, 0, 82.5, -82.5, 0, 88, 0]) * mm
    alpha = np.array([-np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0])

    L = [RevoluteDH(d=d[i], a=a[i], alpha=alpha[i]) for i in range(7)]

    masses = [1, 0, 3, 0, 5, 0, 2.5]
    inertias = [[0.1, 0.1, 0.1, 0, 0, 0]] * 7
    centers = [
        [-d[0]/2, 0.0, 0.0],
        [0, 0, d[2]/2],
        [0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0, -d[4]/2, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ]

    for i, link in enumerate(L):
        link.m = masses[i]
        link.I = inertias[i]
        link.r = centers[i]
        link.Jm = 0

    return DHRobot(L, name='panda')