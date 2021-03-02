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


if __name__ == '__main__':
	# Display info
	verbose = True

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
	n_ep_tot = 500
	observations = np.empty([0,6])
	my_policy_rand = policy_rand(U)

	print("Generating episodes") if verbose else ""

	# Generate episodes
	for i in range(n_ep_tot):
		p_0 = np.random.rand()*0.2-0.1
		ep = car_on_the_hill_problem(U, m, g, gamma, time_interval, integration_time_step, my_policy_rand, p_0, s_0, T, stop_terminal=True)
		observations = add_episode(observations, ep)

	# Create NN
	net = Net()

	# Test to pass info trough the NN
	print(observations[:,:3].shape)
	obs_tensor = torch.from_numpy(observations[:,:3]).float()
	print(obs_tensor)
	output = net(obs_tensor)
	print(output)
	print(output.shape)

	## Train the neural network