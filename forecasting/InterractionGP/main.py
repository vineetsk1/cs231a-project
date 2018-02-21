'''
Created on Mar 1, 2014

@author: ntraft
'''
from __future__ import division
import sys
import os
import argparse
import cv2
import numpy as np
import matplotlib
# The 'MacOSX' backend appears to have some issues on Mavericks.
if sys.platform.startswith('darwin'):
	matplotlib.use('TkAgg')
import matplotlib.pyplot as pl

import ewap as ewap
import display as display
import models as models
import util as util

#import com.ntraft.ewap as ewap
#import com.ntraft.display as display
#import com.ntraft.models as models
#import com.ntraft.util as util

POS_MSEC = cv2.cv.CV_CAP_PROP_POS_MSEC
POS_FRAMES = cv2.cv.CV_CAP_PROP_POS_FRAMES
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
ESC = 27

def main():
	# Parse command-line arguments.
	args = parse_args()
	
	Hfile = os.path.join(args.datadir, "H.txt")
	mapfile = os.path.join(args.datadir, "map.png")
	obsfile = os.path.join(args.datadir, "obsmat.txt")
	destfile = os.path.join(args.datadir, "destinations.txt")

	# Parse homography matrix.
	H = np.loadtxt(Hfile)
	Hinv = np.linalg.inv(H)
	# Parse obstacle map.
	obs_map = ewap.create_obstacle_map(mapfile)
	# Parse pedestrian annotations.
	frames, timeframes, timesteps, agents = ewap.parse_annotations(Hinv, obsfile)
	# Parse destinations.
	destinations = np.loadtxt(destfile)
	
	# Play the video
	seqname = os.path.basename(args.datadir)
	cap = cv2.VideoCapture(os.path.join(args.datadir, seqname+".avi"))

	disp = display.Display(cap, Hinv, obs_map, frames, timesteps, agents, destinations)
	disp.model = models.model21

	# We can set the exact frame we want to look at:
# 	disp.set_frame(780) # ETH sequence, beginning
# 	disp.set_frame(2238) # ETH sequence, agent #48 (1 v 1)
	disp.set_frame(11301) # ETH sequence, big crowds both ways
# 	disp.set_frame(8289) # ETH sequence, agent #175 (1 v crowd)
# 	disp.set_frame(9867) # ETH sequence, agent #236 (1 v crowd)
# 	disp.set_frame(8859) # ETH sequence, agent #194 (1 v 1, choosing non-optimal avoidance)
# 	disp.set_frame(9261) # Hotel sequence, agent #175

	# We can also go by minutes & seconds:
# 	seekpos = 7.5 * 60 * 1000 # About 7 mins 30 secs
# 	endpos = 8.7 * 60 * 1000 # About 8 mins 40 secs
# 	cap.set(POS_MSEC, seekpos)

	# Alternatively, we can search for a particular agent we're interested in:
# 	agent_to_search = 194
# 	ped_path = agents[agent_to_search]
# 	frame_num = timeframes[int(ped_path[0,0])]
# 	disp.set_frame(frame_num)
# 	disp.agent_num = agent_to_search
# 	disp.do_predictions = False
	
	pl.ion()
	display.plot_prediction_metrics([], [], [])
	
	while cap.isOpened():
		
		disp.do_frame()
		
		key = cv2.waitKey(0) & 0xFF
		if key == ord('e'):
			run_experiment(cap, disp, timeframes, timesteps, agents)
		elif key == ord('q') or key == ESC:
			break
		elif key == ord('r'):
			disp.redo_prediction()
		elif key == ord(' '):
			disp.toggle_prediction()
		elif key == LEFT:
			disp.back_one_frame()
		elif key == UP:
			if disp.draw_all_agents: disp.next_sample()
			else: disp.next_agent()
		elif key == DOWN:
			if disp.draw_all_agents: disp.prev_sample()
			else: disp.prev_agent()
		elif key == ord('a'):
			disp.draw_all_agents = not disp.draw_all_agents
			disp.draw_all_samples = not disp.draw_all_samples
			disp.reset_frame()
		elif key == ord('t'):
			disp.draw_truth = not disp.draw_truth
			disp.reset_frame()
		elif key == ord('f'):
			disp.draw_plan = not disp.draw_plan
			disp.reset_frame()
		elif key == ord('p'):
			disp.draw_past = not disp.draw_past
			disp.reset_frame()
		elif key == ord('s'):
			disp.draw_samples = (disp.draw_samples + 1) % display.SAMPLE_CHOICES
			disp.reset_frame()
	
	cap.release()
	cv2.destroyAllWindows()
	pl.close('all')

def run_experiment(cap, disp, timeframes, timesteps, agents):
	# We're going to compare ourselves to the agents with the following IDs.
	print 'Running experiment...'
	util.reset_timer()
	agents_to_test = range(319, 331)
# 	agents_to_test = [48, 49]
	true_paths = []
	planned_paths = []
	one_step_paths = []
# 	IGP_scores = np.zeros((len(agents_to_test), 2))
# 	ped_scores = np.zeros((len(agents_to_test), 2))
	display.plot_prediction_metrics([], [], [])
	for i, agent in enumerate(agents_to_test):
		print 'Agent ', agent
		ped_path = agents[agent]
		path_length = ped_path.shape[0]
		start = 3 # Initializes path with start+1 points.
		final_path = np.zeros((path_length, 3))
		final_path[0:start+1, :] = ped_path[0:start+1,1:4]
		one_step_final_path = final_path.copy()
		# Run IGP through the path sequence.
		for t in range(start, path_length-1):
			frame_num = timeframes[int(ped_path[t,0])]
			print 'doing frame', frame_num
			disp.set_frame(frame_num)
			final_path[t+1], one_step_final_path[t+1] = disp.do_frame(agent, final_path[:t+1], False, True)
			if cv2.waitKey(1) != -1:
				print 'Canceled!'
				return
		# Compute the final score for both IGP and pedestrian ground truth.
		print 'Agent', agent, 'done.'
# 		start_time = int(ped_path[0,0])
# 		other_peds = [agents[a] for a in timesteps[start_time] if a != agent]
# 		other_paths = [util.get_path_at_time(start_time, fullpath)[1][:,1:4] for fullpath in other_peds]
# 		IGP_scores[i] = util.length_and_safety(final_path, other_paths)
# 		ped_scores[i] = util.length_and_safety(ped_path[:,1:4], other_paths)
		true_paths.append(ped_path[:,1:4])
		planned_paths.append(final_path)
		one_step_paths.append(one_step_final_path)
		pred_errs = util.calc_pred_scores(true_paths, planned_paths, util.prediction_errors)
		path_errs = util.calc_pred_scores(true_paths, planned_paths, util.path_errors)
		display.plot_prediction_metrics(pred_errs, path_errs, agents_to_test[0:i+1])
		one_step_errors = util.calc_pred_scores(true_paths, one_step_paths, util.prediction_errors)
		pl.figure(2)
		pl.clf()
		display.plot_prediction_error('One-Step Prediction Errors', one_step_errors, agents_to_test[0:i+1])
		pl.draw()
# 	results = np.column_stack((agents_to_test, ped_scores, IGP_scores))
	pred_results = np.column_stack((agents_to_test, pred_errs.T))
	path_results = np.column_stack((agents_to_test, path_errs.T))
# 	np.savetxt('experiment.txt', results)
	np.savetxt('prediction_errors.txt', pred_results)
	np.savetxt('path_errors.txt', path_results)
	print 'EXPERIMENT COMPLETE.'
	util.report_time()

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("datadir", help="The parent directory for the dataset to be used.")
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	sys.exit(main())
