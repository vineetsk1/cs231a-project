from __future__ import division
from __future__ import print_function
import sys
import numpy as np
from numpy.core.numeric import inf
import time
import os
import argparse
import cv2
import numpy as np
import matplotlib
from util import *
# The 'MacOSX' backend appears to have some issues on Mavericks.
#if sys.platform.startswith('darwin'):
#	matplotlib.use('TkAgg')
import matplotlib.pyplot as pl

import ewap as ewap
import display as display
import models as models

from gp import ParametricGPModel
#from com.ntraft.gp import ParametricGPModel
from collections import namedtuple

NUM_SAMPLES = 100	# number of particles
ALPHA = 1.0			# repelling force
H = 11				# safety distance

work_time = 0
start_time = 0
total_runs = 0

Hfile =  "seq_hotel/H.txt"
obsfile = "seq_hotel/obsmat.txt"
#Hfile =  "data_zara01/H.txt"
#obsfile = "data_zara01/obsmat.txt"
#mapfile = "data_zara01/map.png"
#obs_map = ewap.create_obstacle_map(mapfile)

# Parse homography matrix.
H = np.loadtxt(Hfile)
Hinv = np.linalg.inv(H)
# Parse pedestrian annotations.
frames, timeframes, timesteps, agents = ewap.parse_annotations(Hinv, obsfile)
model = models.model1

Err=[];Edt=[];
max_frame = max(frames)


gp_model=ParametricGPModel()
robot=-1
past_plan=None

t = 1;

peds = timesteps[t]
past_paths = []
true_paths = []
prior = []

#ped=1;
for ped in peds:
	# Get the past and future paths of the agent.
	past_plus_dest, future = get_path_at_time(t, agents[ped])
	past_paths.append(past_plus_dest[:-1,1:4].copy())
	# Replace human's path with robot's path.
	if past_plan is not None and ped == robot:
		past_plus_dest[:-1,1:] = past_plan
	true_paths.append(future[:,1:4])

	# Predict possible paths for the agent.
	t_future = future[:,0]
	gp_model.recompute(past_plus_dest, t_future)
	samples = gp_model.sample(NUM_SAMPLES)
	prior.append(samples)

# Perform importance sampling and get the maximum a-posteriori path.
weights = interaction(prior)
sortdex = np.argsort(-weights)
weights = weights[sortdex]
prior = [p[:,sortdex,:] for p in prior]
posterior, plan = compute_expectation(prior, weights)