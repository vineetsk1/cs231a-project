from __future__ import division
import sys
import os
import argparse
import cv2
import numpy as np
# import matplotlib
# The 'MacOSX' backend appears to have some issues on Mavericks.
#if sys.platform.startswith('darwin'):
#	matplotlib.use('TkAgg')
# import matplotlib.pyplot as pl

import ewap as ewap
# import display as display
import models as models
import util as util
from util import *

# POS_MSEC = cv2.cv.CV_CAP_PROP_POS_MSEC
# POS_FRAMES = cv2.cv.CV_CAP_PROP_POS_FRAMES
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
ESC = 27
# import matplotlib.pyplot as pl

import ewap as ewap
# import display as display
import models as models

from gp import ParametricGPModel
#from com.ntraft.gp import ParametricGPModel
from collections import namedtuple



def testigp():
	# Hfile =  "st3_dataset/H_iw.txt"
#Hfile =  "data_zara01/H.txt"
#obsfile = "data_zara01/obsmat.txt"
#mapfile = "data_zara01/map.png"
#obs_map = ewap.create_obstacle_map(mapfile)

# Parse homography matrix.
	# H = np.loadtxt(Hfile)
	# Hinv = np.linalg.inv(H)
	# Parse pedestrian annotations.
	# obsfile = "st3_dataset/obsmat_px.txt"
	# frames, timeframes, timesteps, agents = ewap.parse_annotations(None, obsfile)

	for folder in os.listdir("seq_trajnet/gt/"):
		if "." in folder:
			continue
		for fname in os.listdir("seq_trajnet/gt/" + folder + "/"):
			if not fname.endswith(".txt"):
				continue
			obsfile = "seq_trajnet/gt/" + folder + "/" + fname
			print "Processing", folder + "/" + fname

			# obsfile = "seq_trajnet/biwi_eth.txt"
			gt = np.loadtxt(obsfile)
			frames, timeframes, timesteps, agents = ewap.parse_annotations_trajnet(obsfile)

			model = models.model1

			peds = []
			for i in range(len(agents)):
				if len(agents[i]) != 0:
					peds.append(i)

			time = []
			for i in range(len(peds)):
				time.append(int(agents[peds[i]][8][0]))

			# for all timesteps
			# time = [timeframes[0]] # timeframes

			Err=[];Edt=[];
			max_frame = max(frames)
			ind = 0
			for t in time:
			#if t % 100 == 0:
				#print "Frame {0} of {1}".format(t, max_frame)
			# we then predict all the path for the agent present in the scene till the destination. 
			# We should modify the results to only predict until someone else enter he map.
			#
			# to do that, we consider the values   "predictions.plan[k]" and select only until the step where someone come in.
			# if we are at time "t" and someone came in at time "t+n" then we only look for     "predictions.plan[k])[range(k)]"
				predictions = util.make_predictions(t, timesteps, agents, model)

				E = [];Ed=[];
				for k in range(0,len(predictions.true_paths)):
					if timesteps[t][k] != peds[ind]:
						continue
					print "PREDICTING TIME", t, "PEDESTRIAN", timesteps[t][k]
					# we compare the predicted trajectory to the true path by plootting them
					# print predictions.true_paths[k][:, 0]
					# print predictions.true_paths[k][:, 1]
					# print predictions.plan[k][:, 0]
					# print predictions.plan[k][:, 1]

					# print gt[gt[:, 1] == peds[ind], 2:4][8:, :]

					inds = np.where(gt[:, 1] == peds[ind])[0][8:]
					gt[inds, 2] = predictions.plan[k][:, 0]
					gt[inds, 3] = predictions.plan[k][:, 1]

					# print inds
					# for n in range(8, 21):
						# gt[gt[:, 1] == peds[ind], 2]
					# print gt[gt[:, 1] == peds[ind], 2:4][8:, :]
					# print predictions.plan[k][:, 0]
					# gt[gt[:, 1] == peds[ind], :][8:, 2] = predictions.plan[k][:, 0]
					# gt[gt[:, 1] == peds[ind], :][8:, 3] = predictions.plan[k][:, 1]
					# print gt[gt[:, 1] == peds[ind], 2:4][8:, :]

					# sys.exit(0)

					# pl.plot(predictions.true_paths[k][:,0],predictions.true_paths[k][:,1],'k');
					# pl.plot(predictions.plan[k][:,0],predictions.plan[k][:,1],'r');
					error= path_errors(predictions.true_paths[k], predictions.plan[k])[0:23]
					E.append(sum(error)/len(error))
					Ed.append(error[-1])
				# pl.show()

				ind += 1
				Err.append(E)
				Edt.append(Ed)

			np.savetxt("seq_trajnet/predict/" + folder + "/" + fname, gt)

# we evaluate the mean error in two ways. Em give us the average error in meters. Edm give us the average displacement of the final position. (it depends on how many frame we want to predict on)
	
	# Em=0;Edm=0;
	# for k in range(len(Err)):
	# 	if len(Err[k])>0:
	# 		Em = Em+(sum(Err[k])/len(Err[k]))/len(Err);
	# 		Edm = Edm+(sum(Edt[k])/len(Edt[k]))/len(Err);

	# return Em, Edm


testigp()

	

"""def testigp(loc):
	Hfile =  loc+"/H.txt"
	obsfile = loc+"/obsmat.txt"
	mapfile = loc+"/map.png"
	obs_map = ewap.create_obstacle_map(mapfile)

	# Parse homography matrix.
	H = np.loadtxt(Hfile)
	Hinv = np.linalg.inv(H)
	# Parse pedestrian annotations.
	frames, timeframes, timesteps, agents = ewap.parse_annotations(Hinv, obsfile)
	model = models.model1
	E = [];Ed=[];
	for t in range(max(frames)):
		
		predictions = util.make_predictions(int(t), timesteps, agents, model)

		
	
		for k in range(0,len(predictions.true_paths)):
			#pl.plot(predictions.true_paths[k][:,0],predictions.true_paths[k][:,1],'k');
			pl.plot(predictions.plan[k][:,0],predictions.plan[k][:,1],'g');
			pl.plot(predictions.true_paths[k][:,0],predictions.true_paths[k][:,1],'r')
			error= path_errors(predictions.true_paths[k], predictions.plan[k])[0:11]
			E.append(error)
			
	
	pl.show()
	return(E)"""


