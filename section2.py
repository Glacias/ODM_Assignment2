import numpy as np
import math
import matplotlib.pyplot as plt
from section1 import *


def score_mu(table_traj, N):
	# Initialize score_mu_table
	score_mu_table = np.zeros(N+1)
	for traj in table_traj:
		expected_reward = traj.gamma**traj.terminal_state_nbr * traj.terminal_state_r / len(table_traj)
		score_mu_table[traj.terminal_state_nbr+1:] += expected_reward
	return score_mu_table


if __name__ == '__main__':
	U = [4, -4]
	m = 1
	g = 9.81
	gamma = 0.95
	time_interval = 0.1
	integration_time_step = 0.001
	N = 50
	n_traj = 50
	p_0_table = [np.random.rand()*0.2-0.1 for i in range(n_traj)]
	s_0 = 0
	#my_policy = policy_cst(U, "right")
	#my_policy = policy_rand(U)
	my_policy = policy_climb(U)

	# Generate trajectories
	table_traj = [car_on_the_hill_problem(U, m, g, gamma, time_interval, integration_time_step, my_policy, p_0_table[i], s_0, N, stop_terminal=False) for i in range(n_traj)]
	score_mu_table = score_mu(table_traj, N)

	# Graph
	plt.plot(range(0, N+1), score_mu_table)
	plt.xticks(range(0,N+1,int((N+1)/4)))
	plt.xlabel('N')
	plt.ylabel('Expected return $(J^{\mu}_N)$')
	plt.show()
