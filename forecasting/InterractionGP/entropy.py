'''
Created on Dec 10, 2014

@author: ntraft
'''
from __future__ import division
import sys
import os
import argparse
import numpy as np

import com.ntraft.ewap as ewap
import com.ntraft.util as util

def main():
	# Parse command-line arguments.
	args = parse_args()
	
	Hfile = os.path.join(args.datadir, "H.txt")
	obsfile = os.path.join(args.datadir, "obsmat.txt")

	# Parse homography matrix.
	H = np.loadtxt(Hfile)
	Hinv = np.linalg.inv(H)
	# Parse pedestrian annotations.
	frames, timeframes, timesteps, agents = ewap.parse_annotations(Hinv, obsfile)
	
	# Agents 319-338
	timeline = range(11205, 11554)
	
	print 'Running experiment...'
	util.reset_timer()
	
	num_samples = 100
	variances = np.array([0.1, 0.3, 1, 3, 10, 30, 100])
	entropies = np.zeros_like(variances)
	for i, sigma2 in enumerate(variances):
		M = np.zeros((2,2))
		total_samples = 0
		for frame in timeline:
			t = frames[frame]
			if t == -1: continue
			
			T = len(timeline)
			print '{:.1%} complete'.format((T*i+frame-timeline[0])/(T*len(variances)))
			
			for _ in range(num_samples):
	# 			predictions = util.make_predictions(t, timesteps, agents)
				predictions = util.fake_predictions(t, timesteps, agents, sigma2)
				for a,plan in enumerate(predictions.plan):
					if plan.shape[0] > 1:
						error = predictions.true_paths[a][1,0:2] - plan[1,0:2]
						M += np.outer(error, error)
						total_samples += 1
		
		M /= total_samples
		entropies[i] = 0.5*np.log((2*np.pi*np.e)**2 * np.linalg.det(M))
# 		print 'entropy is', entropy
# 		print 'variance:'
# 		print M
	
	print 'EXPERIMENT COMPLETE.'
	util.report_time()
	results = np.column_stack((variances, entropies))
	print results
	np.savetxt('experiments/entropy_vs_variance.txt', results)

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("datadir", help="The parent directory for the dataset to be used.")
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	sys.exit(main())
