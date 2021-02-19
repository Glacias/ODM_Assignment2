import numpy as np
import math
import matplotlib.pyplot as plt

# Class for simulating the car on the hill problem with the euler method
class car_on_the_hill_problem():
	def __init__(self, U, m, g, gamma, time_interval, integration_time_step, policy, p_0, s_0, T):
		self.U = U
		self.m = m
		self.g = g
		self.gamma = gamma
		self.time_interval = time_interval
		self.integration_time_step = integration_time_step
		self.terminal_state_reached = False

		# Compute N
		N = int(time_interval/integration_time_step)
		# Get next action
		u_0 = policy.choose_action(p_0, s_0)
		# Compute first step
		next_step = self.Euler_method(p_0, s_0, u_0, N, integration_time_step)

		# Trajectory (p_t, s_t, u_t, r, p_next, s_next)
		self.traj = np.zeros([T, 6])

		# Initialize traj
		self.traj[0][0] = p_0 # p_t
		self.traj[0][1] = s_0 # s_t
		self.traj[0][2] = u_0 # u
		self.traj[0][3] = self.R(next_step[0], next_step[1]) # r
		self.traj[0][4] = next_step[0] # p_next
		self.traj[0][5] = next_step[1] # s_next

		# Compute the following steps
		for i in range(T-1):
			self.traj[i+1][0] = self.traj[i][4]
			self.traj[i+1][1] = self.traj[i][5]
			self.traj[i+1][2] = policy.choose_action(self.traj[i][4], self.traj[i][5])

			# Compute next step
			next_step = self.Euler_method(self.traj[i+1][0], self.traj[i+1][1], self.traj[i+1][2], N, integration_time_step)

			self.traj[i+1][3] = self.R(next_step[0], next_step[1])
			self.traj[i+1][4] = next_step[0]
			self.traj[i+1][5] = next_step[1]


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
		if (p_next < -1 or np.abs(s_next) > 3) and not self.terminal_state_reached:
			self.terminal_state_reached = True
			return -1
		elif p_next > 1 and np.abs(s_next) < 3 and not self.terminal_state_reached:
			self.terminal_state_reached = True
			return 1
		else:
			return 0

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
	def choose_action(self, p, s):
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

	def choose_action(self, p, s):
		return self.action

# policy class for a random action
class policy_rand(cls_policy):
	def __init__(self, U):
		self.U = U

	def choose_action(self, p, s):
		return self.U[np.random.randint(len(self.U))]



if __name__ == '__main__':

	U = [4, -4]
	m = 1
	g = 9.81
	gamma = 0.95
	time_interval = 0.1
	integration_time_step = 0.001
	p_0 = -1
	#p_0 = np.random.rand()*0.2-0.1
	s_0 = 0
	my_policy = policy_cst(U, "right")
	#my_policy = policy_rand(U)
	T = 20

	sect1 = car_on_the_hill_problem(U, m, g, gamma, time_interval, integration_time_step, my_policy, p_0, s_0, T)
	# Graph
	plt.plot(range(0,T), sect1.traj[:, 0], 'ro', label='Position')
	plt.plot(range(0,T), sect1.traj[:, 1], 'go', label='Speed')
	plt.plot(range(0,T), sect1.traj[:, 3], 'bo', label='Reward')
	plt.xlabel('Time')
	plt.ylabel('Value')
	plt.legend()
	plt.show()

	# Display on terminal
	print("Trajectory :")
	for t in range(T):
		print("(p_" + str(t) + " = " + str(sect1.traj[t][0]) +
			", s_" + str(t) + " = " + str(sect1.traj[t][1]) +
			", u_" + str(t) + " = " + str(sect1.traj[t][2]) +
			", r_" + str(t) + " = " + str(sect1.traj[t][3]) +
			", p_" + str(t+1) + " = " + str(sect1.traj[t][4]) +
			", s_" + str(t+1) + " = " + str(sect1.traj[t][5]) + ")")
