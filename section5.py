import numpy as np
import math
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from section4 import *

# Neural network used fot parametric Q-learning
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(3, 200)
		self.fc2 = nn.Linear(200, 1)
		#self.fc3 = nn.Linear(50, 1)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		#x = self.fc3(x)
		return x

	def predict(self, x):

		x = torch.from_numpy(np.array(x)).float().to(next(self.parameters()).device)

		with torch.no_grad():
			x = self.forward(x)

		return x.detach().cpu().numpy()



# Implementing the loss L = 1/2 (Q_phi - y)^2
def my_loss(output, y):
	loss = torch.mean(1/2 * (output - y.detach())**2)
	return loss

def compute_Q_param(observations, U, gamma, n_epoch, batch_size, device=torch.device("cpu"), verbose=True):

	# Create NN
	net = Net().to(device)

	# Set optimizer
	#optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.0005)
	optimizer = optim.Adam(net.parameters(), lr=0.001)

	# Set data input
	Obs = torch.from_numpy(observations).float().to(device)

	# Train the neural network
	n_batch = Obs.shape[0]//batch_size + (0 if (Obs.shape[0]%batch_size) == 0 else 1)

	for epoch in range(n_epoch):
		# Shuffle observations
		idx = torch.randperm(Obs.shape[0])
		Obs = Obs[idx]
		tot_loss = 0

		for batch_idx in range(0, Obs.shape[0], batch_size):
			# Obs[batch_idx:batch_idx+batch_size]

			X = Obs[batch_idx:batch_idx+batch_size, :3]

			# Reset the gradient
			net.zero_grad()
			# Predict Q
			Q = net(X)

			# Compute y and don't keep track of the gradient for these operations
			with torch.no_grad():
				X_next = torch.cat([Obs[batch_idx:batch_idx+batch_size, 4:], torch.ones([X.shape[0], 1]).to(device) * U[0]], dim=1)
				Q_next = net(X_next)

				# For all possible action
				for u_idx in range(len(U)-1):
					X_next = torch.cat([Obs[batch_idx:batch_idx+batch_size, 4:], torch.ones([X.shape[0], 1]).to(device) * U[u_idx+1]], dim=1)
					Q_next = torch.cat([Q_next, net(X_next)], dim=1)

				max_Q_next = torch.max(Q_next, dim=1)[0]
				y = Obs[batch_idx:batch_idx+batch_size, 3] + gamma * max_Q_next

			loss = my_loss(Q, y)
			loss.backward()
			optimizer.step()

			tot_loss += loss
			#print("Ep {}/{}, batch {}/{} : loss = {}".format(epoch+1, n_epoch, (batch_idx//batch_size)+1, n_batch, loss)) if verbose else ""
		print("Ep {}/{}, {} batches : loss = {}".format(epoch+1, n_epoch, n_batch, tot_loss)) if verbose else ""

	return net

def generate_observations(U, m, g, gamma, time_interval, integration_time_step, s_0, T, n_episode, verbose=True):

	observations = np.empty([0,6])
	my_policy_rand = policy_rand(U)

	print("Generating episodes") if verbose else ""

	# Generate episodes
	for i in range(n_episode):
		p_0 = np.random.rand()*0.2-0.1
		ep = car_on_the_hill_problem(U, m, g, gamma, time_interval, integration_time_step, my_policy_rand, p_0, s_0, T, stop_terminal=True)
		observations = add_episode(observations, ep)

	print("Number of observations = {}".format(observations.shape[0]))
	print("Number of empty = {}".format(np.count_nonzero(observations[:,3]==0)))
	print("Number of success = {}".format(np.count_nonzero(observations[:,3]==1)))
	print("Number of failure = {}".format(np.count_nonzero(observations[:,3]==-1)))

	return observations

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
	T = 1000
	n_episode_tot = 500

	# Network training param
	n_epoch = 5
	batch_size = 1

	prtcl_name = "sec5_{}epi_{}epo_{}batch_v3".format(n_episode_tot, n_epoch, batch_size)

	# Generate the observations
	obs_numb_step = [1000, 5000, 10000, 25000, 50000]
	#obs_numb_step = [100, 500, 1000]

	observations_all = np.empty([0,6])
	my_policy_rand = policy_rand(U)

	# Generate transitions until the maximum needed is reached
	while True:
		p_0 = np.random.rand()*0.2-0.1
		ep = car_on_the_hill_problem(U, m, g, gamma, time_interval, integration_time_step, my_policy_rand, p_0, s_0, T, stop_terminal=True)

		if observations_all.shape[0] + ep.terminal_state_nbr + 1 < obs_numb_step[-1]:
			observations_all = add_episode(observations_all, ep)
		else:
			ep.terminal_state_nbr = obs_numb_step[-1] - observations_all.shape[0] -1
			observations_all = add_episode(observations_all, ep)
			break

	# Array for valueof expected return
	exp_return_val = np.zeros([len(obs_numb_step),2])

	# Compute for different learning set size (number of observations)
	i = 0
	for n_transition in obs_numb_step:
		observations = observations_all[:n_transition, :]
	
		# Compute the paramatetric Q learning estimator
		Q_estimator_param = compute_Q_param(observations, U, gamma, n_epoch, batch_size, device=device)

		#torch.save(Q_estimator.state_dict(), "weight_NN.weights")

		# Test policy
		my_policy_param = policy_estimator(U, Q_estimator_param)
		p_0 = 0
		ep = car_on_the_hill_problem(U, m, g, gamma, time_interval, integration_time_step, my_policy_param, p_0, s_0, T, stop_terminal=True)

		#plot_decision(Q_estimator_param, episode=ep, save_name=img_folder+prtcl_name)


		## Launch section 4 and compare
		Q_estimator_FQI = ExtraTreesRegressor(n_estimators=100, random_state=0)

		Br = 1
		thresh_ineq = 0.1
		N_Q = compute_N_Q(gamma, Br, thresh_ineq)

		# Compute estimator
		Q_estimator_FQI = compute_Q_estimator(observations, U, gamma, Q_estimator_FQI, N_Q, verbose=True)

		# Test policy
		my_policy_FQI = policy_estimator(U, Q_estimator_FQI)
		p_0 = 0
		ep = car_on_the_hill_problem(U, m, g, gamma, time_interval, integration_time_step, my_policy_FQI, p_0, s_0, T, stop_terminal=True)

		#prtcl_name_FQI = "sec5_{}epi_{}epo_{}batch_v2_FQI".format(n_episode_tot, n_epoch, batch_size)
		#plot_decision(Q_estimator_FQI, episode=ep, save_name=img_folder+prtcl_name_FQI)

		## Estimate expected return for the two methods
		n_traj = 50
		T = 1000
		p_0_table = [np.random.rand()*0.2-0.1 for i in range(n_traj)]

		table_traj_param = [car_on_the_hill_problem(U, m, g, gamma, time_interval, integration_time_step, my_policy_param, p_0_table[i], s_0, T, stop_terminal=True) for i in range(n_traj)]
		table_traj_FQI = [car_on_the_hill_problem(U, m, g, gamma, time_interval, integration_time_step, my_policy_FQI, p_0_table[i], s_0, T, stop_terminal=True) for i in range(n_traj)]

		step_last_reward_param = np.array([table_traj_param[i].terminal_state_nbr for i in range(len(table_traj_param))]).max()
		step_last_reward_FQI = np.array([table_traj_FQI[i].terminal_state_nbr for i in range(len(table_traj_FQI))]).max()
		
		N_param = step_last_reward_param+1 + int(step_last_reward_param*0.25)
		N_FQI = step_last_reward_FQI+1 + int(step_last_reward_FQI*0.25)

		N_max = max([N_param, N_FQI])

		score_mu_table_param = score_mu(table_traj_param, N_max)
		score_mu_table_FQI = score_mu(table_traj_FQI, N_max)

		exp_return_val[i, 0] = score_mu_table_param[-1]
		exp_return_val[i, 1] = score_mu_table_FQI[-1]

		i += 1

	# graph
	plt.plot(obs_numb_step, exp_return_val[:, 0], label ='Param')
	plt.plot(obs_numb_step, exp_return_val[:, 1], label='FQI')
	#plt.xticks(range(0,N_max+1,int((N_max+1)/4)))
	plt.xlabel('Number of transition')
	plt.ylabel('Expected return $(J^{\mu})$')
	plt.legend()
	plt.savefig(img_folder+prtcl_name+"+exp_ret_{}_final.png".format(n_traj))
	#plt.savefig(img_folder+prtcl_name+"+exp_ret_{}.fig".format(n_traj))
	if True:
		plt.show()
