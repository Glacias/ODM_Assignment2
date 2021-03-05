import numpy as np
import math
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from section5 import *

class car_on_the_hill_online(car_on_the_hill_problem):
	def __init__(self, U, m, g, gamma, time_interval, integration_time_step, policy):
		self.U = U
		self.m = m
		self.g = g
		self.gamma = gamma
		self.time_interval = time_interval
		self.integration_time_step = integration_time_step
		self.policy = policy
		self.terminal_state_reached = True

		# Compute N
		self.N = int(time_interval/integration_time_step)

		self.last_step = np.zeros([1,6])

	def gen_init_state(self):
		p0 = np.random.rand()*0.2-0.1
		s0 = 0

		return p0, s0

	def generate_steps(self, n):
		if n <= 0:
			return np.array([])

		steps = np.zeros([n, 6])

		# Initialize traj
		# Start new traj
		if self.terminal_state_reached:
			steps[0][0], steps[0][1] = self.gen_init_state()
			self.terminal_state_reached = False

		# Continue traj
		else:
			steps[0][0] = self.last_step[4]
			steps[0][1] = self.last_step[5]

		u_ini = self.policy.choose_action(steps[0][0], steps[0][1])
		next_step = self.Euler_method(steps[0][0], steps[0][1], u_ini, self.N, self.integration_time_step)

		steps[0][2] = u_ini # u
		steps[0][3] = self.R(next_step[0], next_step[1], 0) # r
		steps[0][4] = next_step[0] # p_next
		steps[0][5] = next_step[1] # s_next

		# Compute the following steps
		for i in range(n-1):
			# Start new traj
			if self.terminal_state_reached:
				steps[i+1][0], steps[i+1][1] = self.gen_init_state()
				self.terminal_state_reached = False

			# Continue traj
			else:
				steps[i+1][0] = steps[i][4]
				steps[i+1][1] = steps[i][5]


			steps[i+1][2] = self.policy.choose_action(steps[i+1][0], steps[i+1][1])

			# Compute next step
			next_step = self.Euler_method(steps[i+1][0], steps[i+1][1], steps[i+1][2], self.N, self.integration_time_step)

			steps[i+1][3] = self.R(next_step[0], next_step[1], i+1)
			steps[i+1][4] = next_step[0]
			steps[i+1][5] = next_step[1]

		self.last_step = steps[n-1, :]

		return steps



def update_Q_param(batch, U, gamma, net, optimizer, device=torch.device("cpu"), verbose=True):

	# Set data input
	batch = torch.from_numpy(batch).float().to(device)

	X = batch[:, :3]

	# Reset the gradient
	net.zero_grad()
	# Predict Q
	Q = net(X)

	# Compute y and don't keep track of the gradient for these operations
	with torch.no_grad():
		X_next = torch.cat([batch[:, 4:], torch.ones([X.shape[0], 1]).to(device) * U[0]], dim=1)
		Q_next = net(X_next)

		# For all possible action
		for u_idx in range(len(U)-1):
			X_next = torch.cat([batch[:, 4:], torch.ones([X.shape[0], 1]).to(device) * U[u_idx+1]], dim=1)
			Q_next = torch.cat([Q_next, net(X_next)], dim=1)

			max_Q_next = torch.max(Q_next, dim=1)[0]
			y = batch[:, 3] + gamma * max_Q_next

	loss = my_loss(Q, y)
	loss.backward()

	# normalize
	nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0, norm_type=2)

	optimizer.step()


if __name__ == '__main__':
	# Display info
	verbose = True

	img_folder = "out/"

	# GPU
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("Device : {}".format(device))

	# Set up constants
	U = [4, -4]
	m = 1
	g = 9.81
	gamma = 0.95
	time_interval = 0.1
	integration_time_step = 0.001
	s_0 = 0

	## Generate a set of transition from trajectories with random policy
	n_episode_tot = 500


	Q_estimator = Net().to(device)
	# Set optimizer
	#optimizer = optim.Adam(Q_estimator.parameters(), lr=0.0001, weight_decay=0.0005)
	optimizer = optim.Adam(Q_estimator.parameters(), lr=0.001)

	policy_explore = policy_eps_greedy_estimator(U, Q_estimator, 1)
	policy_exploit = policy_estimator(U, Q_estimator)


	# parameters
	n_generation = 50000
	size_generation = 1
	final_eps = 0.1
	prtcl_name = "bonus_{}gen_{}length".format(n_generation, size_generation)

	# show some test within the training using the full exploit policy
	show_intermediate = True
	test_pol_mod = n_generation//5

	# Generator
	gen = car_on_the_hill_online(U, m, g, gamma, time_interval, integration_time_step, policy_explore)

	for i in range(n_generation):
		if (i+1)%10 == 0:
			print("Gen {}".format(i+1))

		# Generate a batch
		new_obs = gen.generate_steps(size_generation)

		# Update network
		update_Q_param(new_obs, U, gamma, Q_estimator, optimizer, device=device, verbose=True)

		# Show evolution of policy
		if (i+1)%test_pol_mod ==0 and show_intermediate:
			ep = car_on_the_hill_problem(U, m, g, gamma, time_interval, integration_time_step, policy_exploit, 0, 0, 1000, stop_terminal=True)
			plot_decision(Q_estimator, episode=ep)

		# reduce greediness
		policy_explore.eps -= (1-final_eps)/(n_generation-1)


	# Test final policy (and save)
	ep = car_on_the_hill_problem(U, m, g, gamma, time_interval, integration_time_step, policy_exploit, 0, 0, 1000, stop_terminal=True)

	plot_decision(Q_estimator, episode=ep, save_name=img_folder+prtcl_name)

	## Estimate expected return
	n_traj = 50
	T = 1000
	p_0_table = [np.random.rand()*0.2-0.1 for i in range(n_traj)]
	table_traj = [car_on_the_hill_problem(U, m, g, gamma, time_interval, integration_time_step, policy_exploit, p_0_table[i], s_0, T, stop_terminal=True) for i in range(n_traj)]
	step_last_reward = np.array([table_traj[i].terminal_state_nbr for i in range(len(table_traj))]).max()
	N = step_last_reward+1 + int(step_last_reward*0.25)
	score_mu_table = score_mu(table_traj, N)

	# graph
	plt.plot(range(0, N+1), score_mu_table)
	plt.xticks(range(0,N+1,int((N+1)/4)))
	plt.xlabel('N')
	plt.ylabel('Expected return $(J^{\mu}_N)$')
	plt.savefig(img_folder+prtcl_name+"+exp_ret_{}.png".format(n_traj))

	plt.show()

	print("Final expected return = {}".format(score_mu_table[-1]))
	file = open(img_folder+prtcl_name+"+exp_ret_{}.txt".format(n_traj), "w")
	file.write("Final expected return = {}".format(score_mu_table[-1]))
	file.close()