import numpy as np
import math
import matplotlib.pyplot as plt

# Class for simulating the car on the hill problem with the euler method
class car_on_the_hill_problem():
	def __init__(self, U, m, g, gamma, time_interval, integration_time_step, policy, p_0, s_0):
		self.U = U
		self.m = m
		self.g = g
		self.gamma = gamma
		self.time_interval = time_interval
		self.integration_time_step = integration_time_step
		self.terminal_state_reached = False

		# Compute euler for 1 step
		# Compute N
		N = int(time_interval/integration_time_step)
		u = 4
		t_0 = 0
		self.Euler_method_visual(p_0, s_0, u, t_0, time_interval, integration_time_step)
		result = self.Euler_method(p_0, s_0, u, N, integration_time_step)
		print(result)

	def Hill(self, p):
		if p < 0:
			return p**2 + p
		else:
			return p/sqrt(1+5*p**2)

	def Hill_first_der(self, p):
		if p < 0:
			return 2*p + 1
		else:
			return (1 + 5*p**2)**(-3/2)

	def Hill_second_der(self, p):
		if p < 0:
			return 2
		else:
			return (-15*p)/(1 + 5*p**2)**(5/2)

	# Acceleration
	def acc(self, p, s, u):
		hd1 = self.Hill_first_der(p)
		return u/(self.m *(1 + hd1**2)) - self.g * hd1/(1 + hd1**2) - s**2 * hd1 * self.Hill_second_der(p)/(1 + hd1**2)

	# Reward signal
	def R(self, p_next, s_next):
		# Rewards
		if p_next < -1 or np.abs(s_next) > 3 and not self.terminal_state_reached:
			self.terminal_state_reached = True
			return -1
		elif p_next > 1 and np.abs(s_next) < 3 and not self.terminal_state_reached:
			self.terminal_state_reached = True
			return 1
		else:
			return 0

	def Euler_method_visual(self, p_0, s_0, u, t_0, dt, h):
		# Compute N
		N = int(dt/h)
		#t = t_0
		#p = p_0
		#s = s_0

		t = np.zeros(N)
		t[0] = t_0
		p = np.zeros(N)
		p[0] = p_0
		s = np.zeros(N)
		s[0] = s_0
		a = np.zeros(N)
		a[0] = self.acc(p[0], s[0], u)

		for i in range(N-1):
			a[i+1] = self.acc(p[i], s[i], u)
			t[i+1] = t[i] + h
			s[i+1] = s[i] + h * self.acc(p[i], s[i], u)
			p[i+1] = p[i] + h * s[i]

		plt.plot(t,p)
		plt.show()

		return (p, s, t)

	def Euler_method(self, p_0, s_0, u, N, h):
		# Compute N
		p = p_0
		s = s_0

		for i in range(N):
			s_next = s + h * self.acc(p, s, u)
			p_next = p + h * s
			s = s_next
			p = p_next

		return (p, s)



# main class for creating a policy
class cls_policy():
	def choose_action(self, x):
		pass

# policy class for a constant direction
# give the U matrix and specify the direction desired
class policy_cst(cls_policy):
	def __init__(self, U, direction):
		self.U = U

		if direction == "right":
			self.action = U[0]

		else:
			self.action = U[1]

	def choose_action(self, x):
		return self.action

# policy class for a random action
class policy_rand(cls_policy):
	def __init__(self, U):
		self.U = U

	def choose_action(self, x):
		return self.U[np.random.randint(len(self.U))]



# apply a policy to find action and get the outgoing new state from f_transition
def get_next(g, U, x, my_policy, f_transition):
	u = my_policy.choose_action(x)
	x_next = f_transition(x, u, g.shape)

	return u, x_next



if __name__ == '__main__':

	U = [4, -4]
	m = 1
	g = 9.81
	gamma = 0.95
	time_interval = 5
	integration_time_step = 0.001
	#p_0 = np.random.rand()*0.2-0.1
	p_0 = -0.95
	s_0 = 0
	my_policy = policy_cst(U, "right")

	sect1 = car_on_the_hill_problem(U, m, g, gamma, time_interval, integration_time_step, my_policy, p_0, s_0)

	#t = 0
	#for k in range(100):
	#	t = t + 0.01
	#	print(t)