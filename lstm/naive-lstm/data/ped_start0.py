import os
import sys
import numpy as np
import math
import csv

root = "gt/"

for folder in os.listdir(root):
	if "." in folder:
		continue
	for fname in os.listdir(root + folder):
		if not fname.endswith(".txt"):
			continue
		print "Processing", fname

		path = root + folder + "/" + fname
		data = np.loadtxt(path)
		
		frames = np.unique(data[:, 0])
		frames.sort()

		peds = np.unique(data[:, 1])
		npeds = peds.shape[0]
		newpeds = np.arange(npeds) + 1

		frame_row = []
		ped_row = []
		y_row = []
		x_row = []

		for frame in frames:
			trajs = data[data[:, 0] == frame, :]
			for traj in trajs:
				frame_row.append(traj[0])
				ped_row.append( newpeds[np.where(peds==traj[1])[0]][0] )
				y_row.append(traj[3])
				x_row.append(traj[2])

		csvfile = open(root + folder + "/" + fname[:fname.rfind(".txt")] + "/" + fname[:fname.rfind(".txt")] + ".csv", 'wb')
		writer = csv.writer(csvfile)
		writer.writerows([frame_row, ped_row, y_row, x_row])
		csvfile.close()

