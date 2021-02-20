import numpy as np
import math
import matplotlib.pyplot as plt

# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
from sklearn.linear_model import LinearRegression
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html
from sklearn.ensemble import ExtraTreesRegressor
# http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
from sklearn.neural_network import MLPRegressor

from section1 import *

def build_y(observations, U, gamma, my_estimator):

	Q_prev = np.empty([observations.shape[0], len(U)])
	for u_idx in range(len(U)):
		X_predi = np.append(observations[:,4:], np.ones([observations.shape[0], 1]) * U[u_idx], axis=1)
		Q_prev[:, u_idx] = my_estimator.predict(X_predi)

	max_Q_prev = Q_prev.max(axis=1)

	return observations[:, 3] + gamma * max_Q_prev

def add_episode(observations, episode):
	return np.append(observations, episode.traj[0:episode.terminal_state_nbr+1, :], axis=0)


def compute_Q_estimator(observations, U, gamma, my_estimator, N, verbose=False):

	# output for Q_1
	if verbose:
		print("\tComputing Q_1")
	y = observations[:,3]
	my_estimator.fit(observations[:,:3], y)

	# iterate to find Q_N estimator
	for t in range(2, N+1):
		if verbose:
			print("\tComputing Q_" + str(t))
			print("\t\tBuild y")
		y = build_y(observations, U, gamma, my_estimator)
		if verbose:
			print("\t\tFit estimator")
		my_estimator.fit(observations[:,:3], y)

	return my_estimator

# compute N such that the bound on the suboptimality
# for the approximation (over an horizon limited to N steps) of the optimal policy
# is smaller or equal to a given threshold
def compute_N_Q(gamma, Br, thresh):
	return math.ceil(math.log(thresh * (1-gamma)**2 / (2*Br) , gamma))


class policy_estimator(cls_policy):
	def __init__(self, U, Q_estimator):
		self.U = U
		self.Q_estimator = Q_estimator

	def choose_action(self, p, s):
		u_idx = np.array([self.Q_estimator.predict([[p, s, u]]) for u in self.U]).argmax()

		return self.U[u_idx]



if __name__ == '__main__':
	# Set up constants
	U = [4, -4]
	m = 1
	g = 9.81
	gamma = 0.95
	time_interval = 0.1
	integration_time_step = 0.001

	T = 1000
	s_0 = 0
	my_policy = policy_rand(U)

	n_ep = 500
	observations = np.empty([0,6])
	print("Generating episodes")
	for i in range(n_ep):
		p_0 = np.random.rand()*0.2-0.1
		ep = car_on_the_hill_problem(U, m, g, gamma, time_interval, integration_time_step, my_policy, p_0, s_0, T, stop_terminal=True)
		observations = add_episode(observations, ep)

	print("\t{} tuples generated".format(observations.shape[0]))

	# Compute N
	Br = 1
	thresh = 0.1
	N_Q = compute_N_Q(gamma, Br, thresh)
	print("N_Q = " + str(N_Q))

	# Choose the regression algorithm
	#Q_estimator = LinearRegression()
	Q_estimator = ExtraTreesRegressor(n_estimators=100, random_state=0)
	#Q_estimator = MLPRegressor(random_state=0, max_iter=500, hidden_layer_sizes=(6,6))

	# Compute Q_N
	Q_estimator = compute_Q_estimator(observations, U, gamma, Q_estimator, N_Q, verbose=True)

	# Save policy inferred from Q_N
	my_policy_opt = policy_estimator(U, Q_estimator)

	# Test policy
	p_0 = 0
	ep = car_on_the_hill_problem(U, m, g, gamma, time_interval, integration_time_step, my_policy_opt, p_0, s_0, T, stop_terminal=True)

	print()
	print("Terminal state reached after {} steps with reward {}".format(ep.terminal_state_nbr+1, ep.terminal_state_r))
