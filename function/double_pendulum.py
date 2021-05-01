import numpy as np
import pandas as pd
from scipy.integrate import odeint

def full_model(x, g, L, m):
	### x = [theta1, theta2, theta_prime1, theta_prime2]
	theta = [x[0], x[1]]
	eta = [x[2], x[3]]
	del_theta = theta[0] - theta[1]

	a = (m[0] + m[1])*L[0]
	b = m[1]*L[1]*np.cos(del_theta)
	c = m[1]*L[0]*np.cos(del_theta)
	d = m[1]*L[1]
	e = -m[1]*L[1]*(eta[1]**2.)*np.sin(del_theta) - g*(m[0]+m[1])*np.sin(theta[0])
	f = m[1]*L[0]*(eta[0]**2.)*np.sin(del_theta) - g*m[1]*np.sin(theta[1])

	xprime0 = eta[0]
	xprime1 = eta[1]
	xprime2 = (e*d - b*f)/(a*d - c*b)
	xprime3 = (a*f - c*e)/(a*d - c*b)

	return [xprime0, xprime1, xprime2, xprime3]

# 시간 간격
# dt_=5e-2
dt_=1e-2

def double_pendulum(L, init_vel, tf=5., dt=dt_, m=[2., 1.], g=9.8):
	## Set constants
	init_angle = [1.6, 2.2] #initial angle 
	t0 = 0.  #initial time
	# g = gravity acceleration
	# m = [mass1, mass2]
	# tf, dt = final time, time step
	
	## Solve ODE
	model = lambda x, t: full_model(x, g, L, m)

	x0 = list(np.hstack([np.array(init_angle), np.array(init_vel)]))
	t = np.arange(t0, tf+dt/2., dt)
	x = odeint(model, x0, t)

	## Achieve solutions in time 
	angle1 = x[:,0]
	angle2 = x[:,1]
	vel1 = x[:,2]
	vel2 = x[:,3]
	# return t, angle1, angle2, vel1, vel2

	cols = ['L1', 'L2', 'v1', 'v2', 't', 'angle1', 'angle2', 'vel1', 'vel2']
	len = t.size
	L1_ary = np.array([L[0]] * len)
	L2_ary = np.array([L[1]] * len)
	v1_ary = np.array([init_vel[0]] * len)
	v2_ary = np.array([init_vel[1]] * len)

	return pd.DataFrame(np.transpose([L1_ary, L2_ary, v1_ary, v2_ary, t, angle1, angle2, vel1, vel2]), columns=cols)

# ## Test Double_pendulum funtion
# L = [1., 2.]
# init_vel = [0.1, 0.4]
# t, u1, u2, v1, v2= Double_pendulum(L, init_vel)


# ### Plot the solutions
# x = [u1,u2,v1,v2]
# fig, ax = plt.subplots()
# for n in range(4):
# 	ax.plot(t, x[n], label='n='+str(n))
# plt.legend()
# plt.show()











