import numpy as np
import pandas as pd
from scipy.integrate import odeint

def full_model(x, g, L, c, m):
    return [x[1], -g/L * np.sin(x[0]) - c/m/L*x[1]]

def single_pendulum(L, init_velocity, c, tf=2., dt=0.02, m=0.3, g=9.81):
    ## Set constants
    init_angle = np.pi / 2.  # initial angle
    t0 = 0.  # initial time
    # g = gravity acceleration
    # m = mass
    # c = damping coefficient
    # tf, dt = final time, time step

    ## Solve ODE
    model = lambda x, t: full_model(x, g, L, c, m)

    x0 = [init_angle, init_velocity]
    t = np.arange(t0, tf + dt / 2., dt)
    x = odeint(model, x0, t)

    ## Achieve solutions in time
    angle = x[:, 0]
    velocity = x[:, 1]
    acceleration = -g/L * np.sin(x[:, 0]) - c/m/L * x[:, 1]

    # return t, angle, velocity, acceleration

    cols = ['L', 'v0', 'C', 't', 'angle', 'velocity', 'accel']
    len = t.size
    L_ary = np.array([L]*len)
    v0_ary = np.array([init_velocity]*len)
    c_ary = np.array([c]*len)

    return pd.DataFrame(np.transpose([L_ary, v0_ary, c_ary, t, angle, velocity, acceleration]), columns = cols)

