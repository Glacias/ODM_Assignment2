import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import linear_model

# compute N such that the bound on the suboptimality
# for the approximation (over an horizon limited to N steps) of the optimal policy
# is smaller or equal to a given threshold
def compute_N_Q(gamma, Br, thresh):
	return math.ceil(math.log(thresh * (1-gamma)**2 / (2*Br) , gamma))


if __name__ == '__main__':
	# Set up constants
	U = [4, -4]
	m = 1
	g = 9.81
	gamma = 0.95
	time_interval = 0.1
	integration_time_step = 0.001
	p_0 = 0
	#p_0 = np.random.rand()*0.2-0.1
	s_0 = 0
	#my_policy = policy_cst(U, "left")
	#my_policy = policy_rand(U)
	my_policy = policy_climb(U)
	N = 50

	## Linear regression with first stopping condition (N == N_Q)
	Br = 1
	threshold = 0.1
	N_Q = compute_N_Q(gamma, Br, threshold)
	
	for n in range(N_Q):
		