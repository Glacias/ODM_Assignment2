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


# N ignored if thresh is set
def compute_Q_estimator(observations, U, gamma, my_estimator, N, thresh=0, verbose=False):

	# threshold is set -> use stopping rule 2
	if thresh > 0:
		N = np.infty

	# output for Q_1

	print("\tComputing Q_1") if verbose else ""
	y = observations[:,3]
	my_estimator.fit(observations[:,:3], y)

	y_prev = np.zeros(observations.shape[0])

	# iterate to find Q_N estimator
	t = 2
	while(True):

		# stopping rule 1 (default)
		if t > N:
			break

		# save last y
		y_prev = y


		print("\tComputing Q_" + str(t)) if verbose else ""
		print("\t\tBuild y") if verbose else ""

		# compute output to predict Q_t for each observations
		y = build_y(observations, U, gamma, my_estimator)

		# stopping rule 2
		if thresh > 0:
			norm_inf = abs(y - y_prev).max()
			print("\t\tInfinite norm with previous y = " + str(norm_inf)) if verbose else ""
			if norm_inf < thresh:
				print("\t\tQ change small enough to stop") if verbose else ""
				break

		print("\t\tFit estimator") if verbose else ""

		# fit estimator to predict Q_t
		my_estimator.fit(observations[:,:3], y)

		t +=1

	return my_estimator


def learn_Q_random(U, m, g, gamma, time_interval, integration_time_step, s_0, T, n_ep, my_estimator, N_Q, thresh=0, verbose=True):
	observations = np.empty([0,6])
	my_policy_rand = policy_rand(U)

	print("Generating episodes") if verbose else ""

	for i in range(n_ep):
		p_0 = np.random.rand()*0.2-0.1
		ep = car_on_the_hill_problem(U, m, g, gamma, time_interval, integration_time_step, my_policy_rand, p_0, s_0, T, stop_terminal=True)
		observations = add_episode(observations, ep)

	print("\t{} tuples generated".format(observations.shape[0])) if verbose else ""

	# Compute Q_N
	return compute_Q_estimator(observations, U, gamma, Q_estimator, N_Q, thresh=thresh, verbose=verbose)


def learn_Q_eps_greedy(U, m, g, gamma, time_interval, integration_time_step, s_0, T, n_fit, ep_per_fit, Q_estimator, eps, N_Q, thresh=0, verbose=True):
	observations = np.empty([0,6])
	# start with a full random policy
	my_policy = policy_eps_greedy_estimator(U, None, 1)


	for i in range(n_fit):
		n_new_obs = 0

		print("Generating episodes' serie #{}/{} ({} news)".format(i+1, n_fit, ep_per_fit)) if verbose else ""
		print("\t", end='') if verbose else ""
		for j in range(ep_per_fit):
			print(".{}".format(j+1), end='') if verbose else ""
			p_0 = np.random.rand()*0.2-0.1
			ep = car_on_the_hill_problem(U, m, g, gamma, time_interval, integration_time_step, my_policy, p_0, s_0, T, stop_terminal=True)
			n_new_obs += (ep.terminal_state_nbr+1)
			observations = add_episode(observations, ep)

		print("\t{} new tuples generated, total of {}".format(n_new_obs, observations.shape[0])) if verbose else ""

		Q_estimator = compute_Q_estimator(observations, U, gamma, Q_estimator, N_Q, thresh=thresh, verbose=verbose)
		# set the eps-greedy policy for next generations
		my_policy.Q_estimator = Q_estimator
		my_policy.eps = eps


	# Compute Q_N
	return Q_estimator


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


class policy_eps_greedy_estimator(cls_policy):
	def __init__(self, U, Q_estimator, eps):
		self.U = U
		self.Q_estimator = Q_estimator
		self.eps = eps

	def choose_action(self, p, s):
		if np.random.rand() < self.eps:
			return self.U[np.random.randint(len(self.U))]
		else:
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

	s_0 = 0


	# REGRESSION ALGORITHM
	#Q_estimator = LinearRegression()
	Q_estimator = ExtraTreesRegressor(n_estimators=100, random_state=0)
	#Q_estimator = MLPRegressor(random_state=0, max_iter=500, hidden_layer_sizes=(6,6))

	# STOPPING RULE (True = N_Q || False = threshold)
	stop_ineq = False
	thresh_ineq = 0.1
	thresh_Q = 0.7

	# POLICY (True = random || False = eps-greedy)
	use_pol_rand = False
	eps = 0.1

	# EPISODE TO GENERATE
	T = 1000
	n_ep_tot = 500
	n_fit = 10
	ep_per_fit = int(n_ep_tot/n_fit)
	print("Total number of episode to generate : " + str(n_ep_tot))

	if stop_ineq:
		Br = 1
		N_Q = compute_N_Q(gamma, Br, thresh_ineq)
		print("Stopping rule by bounding inequality :\n\tN_Q = " + str(N_Q))

		if use_pol_rand:
			print("Using random policy")
			Q_estimator = learn_Q_random(U, m, g, gamma, time_interval, integration_time_step, s_0, T, n_ep_tot, Q_estimator, N_Q)

		else:
			print("Using eps-greedy policy")
			Q_estimator = learn_Q_eps_greedy(U, m, g, gamma, time_interval, integration_time_step, s_0, T, n_fit, ep_per_fit, Q_estimator, eps, N_Q, verbose=True)

	else:
		print("Stopping rule by threshold on Q convergence :\n\tthreshold = " + str(thresh_Q))

		if use_pol_rand:
			print("Using random policy")
			Q_estimator = learn_Q_random(U, m, g, gamma, time_interval, integration_time_step, s_0, T, n_ep_tot, Q_estimator, 0, thresh = thresh_Q)

		else:
			print("Using eps-greedy policy")
			Q_estimator = learn_Q_eps_greedy(U, m, g, gamma, time_interval, integration_time_step, s_0, T, n_fit, ep_per_fit, Q_estimator, eps, 0, thresh = thresh_Q, verbose=True)


	# Save policy inferred from Q_N
	my_policy_opt = policy_estimator(U, Q_estimator)

	# Test policy
	p_0 = 0
	ep = car_on_the_hill_problem(U, m, g, gamma, time_interval, integration_time_step, my_policy_opt, p_0, s_0, T, stop_terminal=True)

	print()
	print("Terminal state reached after {} steps with reward {}".format(ep.terminal_state_nbr+1, ep.terminal_state_r))
