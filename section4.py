import numpy as np
import math
import matplotlib.pyplot as plt

# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
from sklearn.linear_model import LinearRegression
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html
from sklearn.ensemble import ExtraTreesRegressor
# http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
from sklearn.neural_network import MLPRegressor

from keras.models import Sequential
from keras.layers import Dense

from section1 import *
from section2 import *

# for a set of observation (four-tuples) which compose our training set,
# compute the value of Q_N using an estimator to compute the value for Q_N-1
def build_y(observations, U, gamma, my_estimator):

	Q_prev = np.empty([observations.shape[0], len(U)])

	# Q_N-1 value for the next states of the TS considering all possible actions
	for u_idx in range(len(U)):
		X_predi = np.append(observations[:,4:], np.ones([observations.shape[0], 1]) * U[u_idx], axis=1)
		Q_prev[:, u_idx] = my_estimator.predict(X_predi)

	# keep for each next state the best possible Q_N-1 value
	max_Q_prev = Q_prev.max(axis=1)

	# return all estimated Q_N
	return observations[:, 3] + gamma * max_Q_prev


# add a generated episode to a set of observations
def add_episode(observations, episode):
	return np.append(observations, episode.traj[0:episode.terminal_state_nbr+1, :], axis=0)


# N value still active (to set a maximum of iteration) if thresh is set
def compute_Q_estimator(observations, U, gamma, my_estimator, N, thresh=0, verbose=False):

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

# learn an estimator of Q_N with generation of episodes using a random policy
def learn_Q_random(U, m, g, gamma, time_interval, integration_time_step, s_0, T, n_ep, my_estimator, N_Q, thresh=0, verbose=True):
	observations = np.empty([0,6])
	my_policy_rand = policy_rand(U)

	print("Generating episodes") if verbose else ""

	# generate episodes
	for i in range(n_ep):
		p_0 = np.random.rand()*0.2-0.1
		ep = car_on_the_hill_problem(U, m, g, gamma, time_interval, integration_time_step, my_policy_rand, p_0, s_0, T, stop_terminal=True)
		observations = add_episode(observations, ep)

	print("\t{} tuples generated".format(observations.shape[0])) if verbose else ""

	# Compute Q_N
	return compute_Q_estimator(observations, U, gamma, Q_estimator, N_Q, thresh=thresh, verbose=verbose)

# iteratively learn several estimators of Q_N ('n_fit' times)
# with episodes generated using an eps-policy based on the previous iteration of Q_N computation
def learn_Q_eps_greedy(U, m, g, gamma, time_interval, integration_time_step, s_0, T, n_fit, ep_per_fit, Q_estimator, eps, N_Q, thresh=0, eps_adapt=False, verbose=True):
	observations = np.empty([0,6])
	# start with a full random policy (eps=1 means every choice will be random)
	my_policy = policy_eps_greedy_estimator(U, None, 1)


	for i in range(n_fit):
		n_new_obs = 0

		print("Generating episodes' serie #{}/{} ({} news)".format(i+1, n_fit, ep_per_fit)) if verbose else ""
		print("Current epsilon = {}".format(my_policy.eps)) if verbose else ""
		print("\t", end='') if verbose else ""

		# generate episodes using previous estimator of Q_N
		for j in range(ep_per_fit):
			p_0 = np.random.rand()*0.2-0.1
			ep = car_on_the_hill_problem(U, m, g, gamma, time_interval, integration_time_step, my_policy, p_0, s_0, T, stop_terminal=True)
			n_new_obs += (ep.terminal_state_nbr+1)
			observations = add_episode(observations, ep)

		print("\n\t{} new tuples generated, total of {}".format(n_new_obs, observations.shape[0])) if verbose else ""

		# estimate Q_N from the current set of observations
		Q_estimator = compute_Q_estimator(observations, U, gamma, Q_estimator, N_Q, thresh=thresh, verbose=verbose)

		# set the eps-greedy policy for next generations
		my_policy.Q_estimator = Q_estimator
		if eps_adapt and n_fit > 1:
			my_policy.eps -= (1-eps)/(n_fit-1)
		else:
			my_policy.eps = eps


	# return the final estimator
	return Q_estimator

def plot_Q(Q_estimator, action, plotname="", plot_fig=True, save_name=None):
	# create space
	p = np.arange(-1, 1, 0.01)
	s = np.arange(-3, 3, 0.01)
	pp, ss = np.meshgrid(p, s)

	# get prediction for each <p, s>
	Q_pred = pred_Q_mat(Q_estimator, pp, ss, action)

	# plot Q values
	plt.contourf(p, s, Q_pred, cmap='RdBu')
	plt.colorbar()
	plt.title(plotname)
	plt.xlabel("p")
	plt.ylabel("s")

	if save_name is not None:
		plt.savefig(save_name)
	if plot_fig:
		plt.show()

	plt.clf()

def pred_Q_mat(Q_estimator, pp, ss, action):
	# reshape to fit predict() function requirement
	ps_mat = np.append(np.expand_dims(pp, axis=2), np.expand_dims(ss, axis=2), axis=2)
	ps_mat = ps_mat.reshape([-1, 2])
	ps_mat = np.append(ps_mat, np.ones([ps_mat.shape[0] ,1])*action , axis=1)

	# use Q_estimator to predict value of the space
	Q_pred = Q_estimator.predict(ps_mat)

	# reshape back prediction to fit space again
	return Q_pred.reshape(pp.shape)

def plot_decision(Q_estimator, episode=None, plot_fig=True, save_name=None):
	# create space
	p = np.arange(-1, 1, 0.01)
	s = np.arange(-3, 3, 0.01)
	pp, ss = np.meshgrid(p, s)

   # get prediction for each <p, s> and u = -4 then u = 4
	Q_pred_back = pred_Q_mat(Q_estimator, pp, ss, -4)
	Q_pred_front = pred_Q_mat(Q_estimator, pp, ss, 4)

	# front Q - back Q => positive value means front action gives a greater Q
	Q_diff = Q_pred_front - Q_pred_back

	# plot Q_diff => back (orange) | front (purple)
	max_diff = abs(Q_diff).max()
	plt.contourf(p, s, Q_diff, cmap='RdBu', vmin=-max_diff, vmax=max_diff)
	cbar = plt.colorbar()
	cbar.set_label('$Q(p, s, 4) - Q(p, s, -4)$', rotation=270)

	# if an episode (one simulation) is given, add it to the plot
	if episode is not None:
		# Extract traj in the state space
		p_traj = np.append(episode.traj[0, 0], episode.traj[:episode.terminal_state_nbr+1, 4])
		s_traj = np.append(episode.traj[0, 1], episode.traj[:episode.terminal_state_nbr+1, 5])

		# plot traj
		plt.plot(p_traj, s_traj, 'k', label="trajectory")
		plt.plot(p_traj, s_traj, 'Dg')
		plt.legend(loc='lower right')

	# add title and labels
	plt.title("Action taken on a path")
	plt.xlabel("p")
	plt.ylabel("s")

	if save_name is not None:
		plt.savefig(save_name+"+path_taken.png")
	if plot_fig:
		plt.show()

	plt.clf()

	# policy plot
	Q_diff_pol = Q_diff
	Q_diff_pol[Q_diff_pol>0] = 1
	Q_diff_pol[Q_diff_pol<0] = -1
	plt.contourf(p, s, Q_diff, cmap='RdBu', vmin=-1, vmax=1)
	plt.title("Policy")
	plt.xlabel("p")
	plt.ylabel("s")

	if save_name is not None:
		plt.savefig(save_name+"+policy.png")
	if plot_fig:
		plt.show()

	plt.clf()


# compute N such that the bound on the suboptimality
# for the approximation (over an horizon limited to N steps) of the optimal policy
# is smaller or equal to a given threshold
def compute_N_Q(gamma, Br, thresh):
	return math.ceil(math.log(thresh * (1-gamma)**2 / (2*Br) , gamma))

# policy taking the argmax of Q_N computed with an estimator
class policy_estimator(cls_policy):
	def __init__(self, U, Q_estimator):
		self.U = U
		self.Q_estimator = Q_estimator

	def choose_action(self, p, s):
		u_idx = np.array([self.Q_estimator.predict([[p, s, u]]) for u in self.U]).argmax()

		return self.U[u_idx]

# policy taking actions at random with prob 'eps' or
# taking the argmax of Q_N computed with an estimator otherwise
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

# Class for the neural network estimator
class NN_estimator():
	def __init__(self):
		# Define the keras model
		self.model = Sequential()
		self.model.add(Dense(200, input_dim=3, activation='relu'))
		self.model.add(Dense(1, activation='linear'))
		self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

	def fit(self, X, y):
		# Re-initialize the network at each fit
		self.model = Sequential()
		self.model.add(Dense(200, input_dim=3, activation='relu'))
		self.model.add(Dense(1, activation='linear'))
		self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

		# Fit
		self.model.fit(X, y, epochs=50)

	def predict(self, X):
		# Predict
		out = self.model.predict(X)
		return out.reshape(len(out))


if __name__ == '__main__':
	# Set up constants
	U = [4, -4]
	m = 1
	g = 9.81
	gamma = 0.95
	time_interval = 0.1
	integration_time_step = 0.001

	s_0 = 0

	verbose = True
	plot_fig = True
	prtcl_name = ""
	img_folder = "out/"

	# REGRESSION ALGORITHM
	algo = ["LinearRegression", "ExtraTreesRegressor", "MLPRegressor"][1]

	if algo == "LinearRegression":
		print("Estimator used is LinearRegression")
		prtcl_name += "lin_reg"
		Q_estimator = LinearRegression()

	elif algo == "ExtraTreesRegressor":
		print("Estimator used is ExtraTreesRegressor")
		prtcl_name += "extra_trees"
		Q_estimator = ExtraTreesRegressor(n_estimators=100, random_state=0)

	else:
		print("Estimator used is MLPRegressor")
		prtcl_name += "mlp"
		#Q_estimator = MLPRegressor(max_iter=500, hidden_layer_sizes=(4,4))
		Q_estimator = NN_estimator()

	# STOPPING RULE (True = N_Q || False = threshold)
	stop_ineq = True
	thresh_ineq = 0.1
	thresh_Q = 0.1

	# POLICY (True = random || False = eps-greedy)
	use_pol_rand = False
	eps = 0.1

	# EPISODE TO GENERATE
	T = 1000
	n_ep_tot = 500
	n_fit = 5
	ep_per_fit = int(n_ep_tot/n_fit)
	print("Total number of episode to generate : " + str(n_ep_tot))

	Br = 1
	N_Q = compute_N_Q(gamma, Br, thresh_ineq)

	if stop_ineq:

		print("Stopping rule by bounding inequality :\n\tN_Q = " + str(N_Q))

		prtcl_name += "+bound_gamma_N_{}".format(N_Q)

		if use_pol_rand:
			prtcl_name += "+random_gen_{}ep".format(n_ep_tot)

			print("Using random policy")
			Q_estimator = learn_Q_random(U, m, g, gamma, time_interval, integration_time_step, s_0, T, n_ep_tot, Q_estimator, N_Q, verbose=verbose)

		else:
			prtcl_name += "+eps_greedy_gen_{}_{}".format(n_fit, ep_per_fit)

			print("Using eps-greedy policy")
			Q_estimator = learn_Q_eps_greedy(U, m, g, gamma, time_interval, integration_time_step, s_0, T, n_fit, ep_per_fit, Q_estimator, eps, N_Q, eps_adapt=True, verbose=verbose)

	else:
		prtcl_name += "+thresh_Q_diff"

		print("Stopping rule by threshold on Q convergence :\n\tthreshold = " + str(thresh_Q))
		max_iter = N_Q

		if use_pol_rand:
			prtcl_name += "+random_gen_{}ep".format(n_ep_tot)

			print("Using random policy")
			Q_estimator = learn_Q_random(U, m, g, gamma, time_interval, integration_time_step, s_0, T, n_ep_tot, Q_estimator, max_iter, thresh = thresh_Q, verbose=verbose)

		else:
			prtcl_name += "+eps_greedy_gen_{}_{}".format(n_fit, ep_per_fit)

			print("Using eps-greedy policy")
			Q_estimator = learn_Q_eps_greedy(U, m, g, gamma, time_interval, integration_time_step, s_0, T, n_fit, ep_per_fit, Q_estimator, eps, max_iter, thresh = thresh_Q, eps_adapt=True, verbose=verbose)


	# Save policy inferred from Q_N
	my_policy_opt = policy_estimator(U, Q_estimator)

	# Test policy
	p_0 = 0
	ep = car_on_the_hill_problem(U, m, g, gamma, time_interval, integration_time_step, my_policy_opt, p_0, s_0, T, stop_terminal=True)

	print()
	print("Testing policy on a simulation starting at p_0 = 0 :")
	print("\tTerminal state reached after {} steps with reward {}".format(ep.terminal_state_nbr+1, ep.terminal_state_r))

	## Plot heat map
	plot_Q(Q_estimator, -4, plotname=r'$\widehat{Q}\left(p, s, -4\right)$', plot_fig=plot_fig, save_name=img_folder+prtcl_name+"+Q_back")
	plot_Q(Q_estimator, 4, plotname=r'$\widehat{Q}\left(p, s, 4\right)$', plot_fig=plot_fig, save_name=img_folder+prtcl_name+"+Q_front")

	plot_decision(Q_estimator, episode=ep, plot_fig=plot_fig, save_name=img_folder+prtcl_name)

	## Estimate expected return
	n_traj = 50
	T = 1000
	p_0_table = [np.random.rand()*0.2-0.1 for i in range(n_traj)]
	table_traj = [car_on_the_hill_problem(U, m, g, gamma, time_interval, integration_time_step, my_policy_opt, p_0_table[i], s_0, T, stop_terminal=True) for i in range(n_traj)]
	step_last_reward = np.array([table_traj[i].terminal_state_nbr for i in range(len(table_traj))]).max()
	N = step_last_reward+1 + int(step_last_reward*0.25)
	score_mu_table = score_mu(table_traj, N)

	# graph
	plt.plot(range(0, N+1), score_mu_table)
	plt.xticks(range(0,N+1,int((N+1)/4)))
	plt.xlabel('N')
	plt.ylabel('Expected return $(J^{\mu}_N)$')
	plt.savefig(img_folder+prtcl_name+"+exp_ret_{}.png".format(n_traj))
	if plot_fig:
		plt.show()

	print("Final expected return = {}".format(score_mu_table[-1]))
	file = open(img_folder+prtcl_name+"+exp_ret_{}.txt".format(n_traj), "w")
	file.write("Final expected return = {}".format(score_mu_table[-1]))
	file.close()
