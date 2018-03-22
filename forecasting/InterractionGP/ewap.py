'''
Utility functions for dealing with the BIWI Walking Pedestrians dataset (or
EWAP for "ETH Walking Pedestrians"). From Pellegrini et al., ICCV 2009.

Created on Mar 1, 2014

@author: ntraft
'''

import numpy as np
from PIL import Image
import util as util

#import com.ntraft.util as util

ignored_peds = [171, 216]

def create_obstacle_map(map_png):
	rawmap = np.array(Image.open(map_png))
	return rawmap

def parse_annotations(Hinv, obsmat_txt):
	mat = np.loadtxt(obsmat_txt)
	num_frames = int(mat[-1,0] + 1)
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
		loc = np.array([row[2], row[4], 1])
		#loc = util.to_image_frame(Hinv, loc)
		loc = [time, loc[0], loc[1], loc[2]] # loc[0], loc[1] should be img coords, loc[2] always "1"
		peds[ped] = np.vstack((peds[ped], loc))
	return (frames, timeframes, timesteps, peds)

def parse_annotations_trajnet(obsmat_txt):
	
	mat = np.loadtxt(obsmat_txt)
	mat = mat[mat[:,0].argsort()]

	largest_frame = int(np.max(mat[:, 0]) + 1)
	num_times = np.unique(mat[:, 0]).size
	num_peds = int(np.max(mat[:,1])) + 1

	# frames -> maps frame to timestep. large dense array where frames[i] contains 
	# the timestep for the ith frame. most values are -1.
	frames = [-1] * largest_frame

	# list of timesteps -> first frame in timestep
	timeframes = [-1] * num_times

	timesteps = [[] for _ in range(num_times)]
	peds = [np.array([]).reshape(0,4) for _ in range(num_peds)]	

	frame = 0
	time = -1

	for row in mat:
		if row[0] != frame:
			frame = int(row[0])
			time += 1
			frames[frame] = time
			timeframes[time] = frame
		ped = int(row[1])
		timesteps[time].append(ped)
		loc = np.array([row[2], row[3], 1])
		loc = [time, loc[0], loc[1], loc[2]]
		peds[ped] = np.vstack((peds[ped], loc))

	return (frames, timeframes, timesteps, peds)
	# TODO

	return frames 

def parse_annotations_SF(obsmat_csv):
	#mat = np.loadtxt(obsmat_csv)
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
	return (frames, timeframes, timesteps, peds)























