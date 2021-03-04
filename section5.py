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
		self.fc1 = nn.Linear(3, 50)
		self.fc2 = nn.Linear(50, 50)
		self.fc3 = nn.Linear(50, 1)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

	def predict(self, x):

		x = torch.from_numpy(x).float()

		with torch.no_grad():
			x = self.forward(x)

		return x.detach().cpu().numpy()



# Implementing the loss L = 1/2 (Q_phi - y)^2
def my_loss(output, y):
	loss = torch.mean(1/2 * (output - y.detach())**2)
	return loss

if __name__ == '__main__':
	# Display info
	verbose = True

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
	n_ep_tot = 1000
	observations = np.empty([0,6])
	my_policy_rand = policy_rand(U)

	print("Generating episodes") if verbose else ""

	# Generate episodes
	for i in range(n_ep_tot):
		p_0 = np.random.rand()*0.2-0.1
		ep = car_on_the_hill_problem(U, m, g, gamma, time_interval, integration_time_step, my_policy_rand, p_0, s_0, T, stop_terminal=True)
		observations = add_episode(observations, ep)

	# Create NN
	net = Net().to(device)

	# Set optimizer
	optimizer = optim.Adam(net.parameters(), lr=0.001)

	# Set data input
	Obs = torch.from_numpy(observations).float().to(device)

	# Train the neural network
	n_epoch = 100
	batch_size = 100

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
		print("Ep {}/{}, {} batches : loss = {}".format(epoch+1, n_epoch, n_batch, loss)) if verbose else ""

	#torch.save(net.state_dict(), "weight_NN.weights")

	net = net.cpu()

	my_policy = policy_estimator(U, net)

	plot_decision(net)