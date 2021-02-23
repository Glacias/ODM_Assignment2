import numpy as np
import math
import matplotlib.pyplot as plt
from section1 import *
from display_caronthehill import *
from PIL import Image, ImageDraw


if __name__ == '__main__':
	U = [4, -4]
	m = 1
	g = 9.81
	gamma = 0.95
	time_interval = 0.1
	integration_time_step = 0.001
	p_0 = 0
	#p_0 = np.random.rand()*0.2-0.1
	s_0 = 0
	my_policy = policy_cst(U, "right")
	#my_policy = policy_rand(U)
	T = 1000

	# Simulate the trajectory
	sect1 = car_on_the_hill_problem(U, m, g, gamma, time_interval, integration_time_step, my_policy, p_0, s_0, T)

	# Create the gif
	images = []
	for t in range(T):
		p = min(max(sect1.traj[t][0], -1), 1)
		s = min(max(sect1.traj[t][1], -3), 3)
		im_arr = save_caronthehill_image(p, s, out_file=None)
		im = Image.fromarray(im_arr, 'RGB')
		images.append(im.transpose(method=Image.TRANSPOSE))
	images[0].save('visual.gif', save_all=True, append_images=images[1:], optimize=False, duration=40, loop=0)
