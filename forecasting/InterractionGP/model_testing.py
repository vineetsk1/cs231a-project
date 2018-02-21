'''
Created on Dec 10, 2014

@author: ntraft
'''
from __future__ import division
import sys
import os
import argparse
import numpy as np

import ewap as ewap
import models as models
import util as util

#import com.ntraft.ewap as ewap
#import com.ntraft.models as models
#import com.ntraft.util as util

def main():
	# Parse command-line arguments.
	args = parse_args()
	
	Hfile = os.path.join(args.datadir, "seq_eth/H.txt")
	obsfile = os.path.join(args.datadir, "seq_eth/obsmat.txt")

	# Parse homography matrix.
	H = np.loadtxt(Hfile)
	Hinv = np.linalg.inv(H)
	# Parse pedestrian annotations.
	frames, timeframes, timesteps, agents = ewap.parse_annotations(Hinv, obsfile)
	
	test_sequences = np.array([
		#range(10300, 10305),
		range(3648, 3769),
		range(4163, 4374),
		range(6797, 7164),
		range(7301, 7608),
		range(8373, 8620),
		range(8859, 9544),
		range(9603, 10096),
		range(10101, 10528),
		range(11205, 11500),
		range(11835, 12172),
		#range(32000, 32200),
	])
	total_t = 0
	for s in test_sequences: total_t += len(s)
	model = models.model1
	
	print 'Running experiment...'
	util.reset_timer()
	
	num_samples = 5
	iters = 0
	total_samples = 0
	entropies = np.zeros(test_sequences.shape[0])
	E=[];
	for i,timeline in enumerate(test_sequences):
		M = np.zeros((2,2))
		for frame in timeline:
		
			iters += 1
			t = frames[frame]
			if t == -1: continue
		
			for _ in range(num_samples):
				predictions = util.make_predictions(t, timesteps, agents, model)
				for a,plan in enumerate(predictions.plan):
					if plan.shape[0] > 1:
						error = predictions.true_paths[a][1,0:2] - plan[1,0:2]
						M += np.outer(error, error)
						total_samples += 1
						E.append(error)
			
			print "frame {}: {:.1%} complete" .format(frame, iters/total_t)
		
		M /= total_samples
		entropies[i] = 0.5*np.log((2*np.pi*np.e)**2 * np.linalg.det(M))
# 		print 'entropy is', entropy
# 		print 'variance:'
# 		print M
	
	print 'EXPERIMENT COMPLETE.'
	util.report_time()
	np.savetxt('hyperparam-test-194-202-joint-2.txt', entropies)
	

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("datadir", help="The parent directory for the dataset to be used.")
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	sys.exit(main())
