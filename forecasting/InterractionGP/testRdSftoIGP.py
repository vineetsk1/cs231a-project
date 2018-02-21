from __future__ import division
import sys
import os
import argparse
import cv2
import numpy as np
import matplotlib
from numpy import linalg as LA
from numpy.core.numeric import inf
import time
import csv
import scipy.io
import matplotlib.pyplot as pl
from util import *
import matplotlib.pyplot as pl


import ewap as ewap
import display as display
import models as models
import util as util
from util import *

POS_MSEC = cv2.cv.CV_CAP_PROP_POS_MSEC
POS_FRAMES = cv2.cv.CV_CAP_PROP_POS_FRAMES

from gp import ParametricGPModel
from collections import namedtuple

DIR_PEDFILE = '/scr/alexr/ped_fileSFtoIGP';
#for ped_file in os.listdir(DIR_PEDFILE):
K = [5];

file = 'C:/Users/Alexandre/Documents/InterractionGP';
ignored_peds=[];
Hinv = np.eye(3);



for k in K:

	x=list(csv.reader(open(file+'/ped_file'+str(k)+'.csv',"rb"),delimiter=','));
	ped_start = np.array(x);  
	# ped_start(origx,dest,'start_frame','velocity','class',fist_position(x,y),last_position(x,y),last_frame,likely_velocity,id)#


	x=list(csv.reader(open(file+'/simtrajRdSF'+str(k)+'.csv',"rb"),delimiter=','));
	true_path = np.array(x) 
	#true_path(x,y,t,id,class)#

	#ped_start=np.delete(ped_start,(0),axis=0)
	true_path = true_path.astype(np.float)
	ped_start=ped_start.astype(np.float);

	mat = true_path[:,[2,3,0,1]];
	t = np.int(max(ped_start[:,2])+1);

	num_frames = mat[-1,0] + 1
	num_times = np.unique(mat[:,0]).size
	num_peds = int(np.max(mat[:,1])) + 1
	frames = [-1] * num_frames # maps frame -> timestep
	timeframes = [-1] * num_times # maps timestep -> (first) frame
	timesteps = [[] for _ in range(num_times)] # maps timestep -> ped IDs
	peds = [np.array([]).reshape(0,4) for _ in range(num_peds)] # maps ped ID -> (t,x,y,z) path
	frame = 0
	time = -1
	for row in mat:
		if row[0] != frame:
			frame = int(row[0])
			time += 1
			frames[frame] = time
			timeframes[time] = frame
		ped = int(row[1])
		if ped not in ignored_peds: # TEMP HACK - can cause empty timesteps
			timesteps[time].append(ped)
		loc = np.array([row[2], row[3], 1])
		loc = util.to_image_frame(Hinv, loc)
		loc = [time, loc[0], loc[1], loc[2]]
		peds[ped] = np.vstack((peds[ped], loc))
		
	agents = peds;



	model = models.model1
	Err=[];Edt=[];
	max_frame = max(frames)


	# we then predict all the path for the agent present in the scene till the destination. 
	# We should modify the results to only predict until someone else enter he map.
	#
	# to do that, we consider the values   "predictions.plan[k]" and select only until the step where someone come in.
	# if we are at time "t" and someone came in at time "t+n" then we only look for     "predictions.plan[k])[range(k)]"
	predictions = util.make_predictions(t, timesteps, agents, model)

	E = [];Ed=[];

	for k in range(0,len(predictions.true_paths)):
		# we compare the predicted trajectory to the true path by plootting them
		pl.plot(predictions.true_paths[k][:,0],predictions.true_paths[k][:,1],'k');
		pl.plot(predictions.plan[k][:,0],predictions.plan[k][:,1],'r');
		error= path_errors(predictions.true_paths[k], predictions.plan[k])[0:23]
		E.append(sum(error)/len(error))
		Ed.append(error[-1])
	pl.show()

	Err.append(E)
	Edt.append(Ed)
# we evaluate the mean error in two ways. Em give us the average error in meters. Edm give us the average displacement of the final position. (it depends on how many frame we want to predict on)
	Em=0;Edm=0;
	for j in range(len(Err)):
		if len(Err[j])>0:
			Em = Em+(sum(Err[j])/len(Err[j]))/len(Err);
			Edm = Edm+(sum(Edt[j])/len(Edt[j]))/len(Err);

		


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


