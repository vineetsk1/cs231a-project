'''
Created on Apr 10, 2014

@author: ntraft
'''
from __future__ import division

import cv2

import util as util
import models as models
#import com.ntraft.util as util
#import com.ntraft.models as models
import matplotlib.pyplot as pl
import numpy as np


POS_MSEC = cv2.cv.CV_CAP_PROP_POS_MSEC
POS_FRAMES = cv2.cv.CV_CAP_PROP_POS_FRAMES

NO_SAMPLES = 0
PRIOR_SAMPLES = 1
POSTERIOR_SAMPLES = 2
SAMPLE_CHOICES = 3


'''
TODO

- New problem: cannot assume the robot will take the same time to arrive at
	the goal! Can we make sure the robot travels at the correct velocity?
	- Probably not, this is a problem that arises from having the GP be
		time-dependent rather than previous-state-dependent.
- Re-run experiment after fixing the above bugs.
- May still need to play with the interaction potential.
	- Check files Pete sent to see if his parameters are in there.
- Can maybe think about drawing other things like future paths or goals.
'''

class Display:
	
	def __init__(self, cap, Hinv, obs_map, frames, timesteps, agents, destinations):
		self.cap = cap
		self.Hinv = Hinv
		self.obs_map = obs_map
		self.frames = frames
		self.timesteps = timesteps
		self.agents = agents
		self.destinations = destinations
		self.predictions = util.empty_predictions
		self.model = models.model1
		self.agent_num = 1
		self.sample_num = 0
		self.last_t = -1
		self.do_predictions = True
		self.draw_all_agents = False
		self.draw_all_samples = True
		self.draw_samples = NO_SAMPLES
		self.draw_truth = True
		self.draw_past = True
		self.draw_plan = True
	
	def set_frame(self, frame):
		self.cap.set(POS_FRAMES, frame)

	def back_one_frame(self):
		frame_num = int(self.cap.get(POS_FRAMES))
		self.set_frame(frame_num-2)

	def reset_frame(self):
		frame_num = int(self.cap.get(POS_FRAMES))
		self.set_frame(frame_num-1)

	def redo_prediction(self):
		self.last_t = -1
		self.reset_frame()

	def toggle_prediction(self):
		self.do_predictions = not self.do_predictions
		if self.do_predictions:
			self.redo_prediction()
		else:
			self.predictions = util.empty_predictions
			self.reset_frame()

	def next_agent(self):
		self.change_agent(lambda x: x+1)
	def prev_agent(self):
		self.change_agent(lambda x: x-1)
	def change_agent(self, fn):
		if self.last_t == -1: return
		agents_in_this_frame = self.timesteps[self.last_t]
		curr_adex = next((i for i,v in enumerate(agents_in_this_frame) if v==self.agent_num), 0)
		next_adex = util.cycle_index(curr_adex, fn, len(agents_in_this_frame))
		self.agent_num = agents_in_this_frame[next_adex]
		self.reset_frame()

	# Returns the index and ID for the desired agent if present in this frame.
	# Otherwise, returns the first agent in the frame.
	def get_agent_index(self, desired_agent):
		if self.last_t <= 0: return (0,0)
		agents_in_this_frame = self.timesteps[self.last_t]
		adex = next((i for i,v in enumerate(agents_in_this_frame) if v==desired_agent), 0)
		return (adex, agents_in_this_frame[adex])

	def next_sample(self):
		self.change_sample(lambda x: x+1)
	def prev_sample(self):
		self.change_sample(lambda x: x-1)
	def change_sample(self, fn):
		self.sample_num = util.cycle_index(self.sample_num, fn, util.NUM_SAMPLES)
		self.reset_frame()

	def do_frame(self, agent=-1, past_plan=None, with_scores=True, multi_prediction=False):
		if not self.cap.isOpened():
			raise Exception('Video stream closed.')
		if agent == -1:
			agent = self.agent_num
		adex, displayed_agent = self.get_agent_index(agent)
		
		t_plus_one = None
		t_plus_one2 = None
		frame_num = int(self.cap.get(POS_FRAMES))
		now = int(self.cap.get(POS_MSEC) / 1000)
		_, frame = self.cap.read()
	
		frame_txt = "{:0>2}:{:0>2}".format(now//60, now%60)
		agent_txt = 'Agent: {}'.format(displayed_agent)
		
		# Check for end of annotations.
		if frame_num >= len(self.frames):
			frame_txt += ' (eof)'
			agent_txt = ''
		else:
			frame_txt += ' (' + str(frame_num) + ')'
			t = self.frames[frame_num]
			# There really shouldn't be any empty timesteps but I found some so we'll have to deal with it.
			if t >= 0 and self.timesteps[t]:
				# If we've reached a new timestep, recompute the observations.
				if t != self.last_t:
					self.last_t = t
					adex, displayed_agent = self.get_agent_index(agent)
					agent_txt = 'Agent: {}'.format(displayed_agent)
					if self.do_predictions:
						self.predictions = util.make_predictions(t, self.timesteps, self.agents, self.model, agent, past_plan)
# 						self.predictions = util.fake_predictions(t, self.timesteps, self.agents, 10.0)
						if multi_prediction and past_plan is not None:
							predictions2 = util.make_predictions(t, self.timesteps, self.agents, self.model, agent, None)
							if predictions2.plan[adex].shape[0] > 1:
								t_plus_one2 = predictions2.plan[adex][1]
						if self.predictions.plan[adex].shape[0] > 1:
							t_plus_one = self.predictions.plan[adex][1]
						if with_scores:
	# 						ped_scores, IGP_scores = util.calc_nav_scores(self.predictions.true_paths, self.predictions.plan)
	# 						plot_nav_metrics(ped_scores, IGP_scores)
							pred_errs = util.calc_pred_scores(self.predictions.true_paths, self.predictions.plan, util.prediction_errors)
							path_errs = util.calc_pred_scores(self.predictions.true_paths, self.predictions.plan, util.path_errors)
							plot_prediction_metrics(pred_errs, path_errs, self.timesteps[t])
					else:
						self.predictions = util.get_past_paths(t, self.timesteps, self.agents)
		
		# Draw the obstacles.
		frame = np.maximum(frame, cv2.cvtColor(self.obs_map, cv2.COLOR_GRAY2BGR))
		
		# Draw destinations.
		for d in self.destinations:
			d = np.append(d, 1)
			cv2.circle(frame, util.to_pixels(self.Hinv, d), 5, (0,255,0), -1)
	
		# Inform of frame and agent number.
		pt = (3, frame.shape[0]-3)
		ll, ur = draw_text(frame, pt, frame_txt)
		if agent_txt:
			pt = (ll[0], ur[1])
			ll, ur = draw_text(frame, pt, agent_txt)
		
		# Draw pedestrian paths so far.
		if self.draw_past and self.predictions.past:
			for path in self.predictions.past:
				draw_path(frame, path, (192,192,192))
			draw_waypoints(frame, self.predictions.past[adex], (255,211,176))
		
		# Draw predictions, if we have them.
		if self.predictions.plan:
			# For each agent, draw...
			peds_to_draw = range(len(self.predictions.plan)) if self.draw_all_agents else [adex]
			
			# The GP samples.
			if self.draw_samples != NO_SAMPLES:
				# What kind of sample are we showing?
				preds = self.predictions.prior if self.draw_samples == PRIOR_SAMPLES else self.predictions.posterior
				# Which sample(s) are we showing?
				if self.draw_all_samples:
					if preds: samples_to_draw = range(preds[0].shape[1])
					else: samples_to_draw = []
				else:
					sdex = self.sample_num
					samples_to_draw = [sdex]
					pt = (ll[0], ur[1])
					ll, ur = draw_text(frame, pt, 'Sample: {}'.format(sdex+1))
					pt = (ll[0], ur[1])
					draw_text(frame, pt, 'Weight: {:.1e}'.format(self.predictions.weights[sdex]))
				# Now actually draw the sample.
				for ddex in peds_to_draw:
					for i in samples_to_draw:
						path = preds[ddex][:,i,:]
						path = np.column_stack((path, np.ones(path.shape[0])))
						draw_path(frame, path, (255,0,0))
						
			# The ground truth.
			if self.draw_truth:
				for ddex in peds_to_draw:
					draw_path(frame, self.predictions.true_paths[ddex], (0,255,0))
				
			# The final prediction.
			if self.draw_plan:
				for ddex in peds_to_draw:
					draw_path(frame, self.predictions.plan[ddex], (0,192,192))
					if not self.draw_all_agents and past_plan is not None:
							draw_path(frame, past_plan, (0,192,192))
							draw_waypoints(frame, past_plan, (0,192,192))
		
		cv2.imshow('frame', frame)
		return t_plus_one if t_plus_one2 is None else (t_plus_one, t_plus_one2)

def plot_prediction_metrics(prediction_errors, path_errors, agents):
	pl.figure(1, (10,10))
	pl.clf()
	if len(prediction_errors) > 0:
		pl.subplot(2,1,1)
		plot_prediction_error('Prediction Error', prediction_errors, agents)
		
		pl.subplot(2,1,2)
		plot_prediction_error('Path Error', path_errors, agents)
		
		pl.draw()

def plot_prediction_error(title, errors, agents):
	pl.title(title)
	pl.xlabel('Time (frames)'); pl.ylabel('Error (px)')
	m = np.nanmean(errors, 1)
	lines = pl.plot(errors)
	meanline = pl.plot(m, 'k--', lw=4)
	pl.legend(lines + meanline, ['{}'.format(a) for a in agents] + ['mean'])

def plot_nav_metrics(ped_scores, IGP_scores):
	pl.clf()
	if len(ped_scores) > 0:
		pl.subplot(1,2,1)
		pl.title('Path Length (px)')
		pl.xlabel('IGP'); pl.ylabel('Pedestrian')
		pl.scatter(IGP_scores[:,0], ped_scores[:,0])
		plot_diag()
		
		pl.subplot(1,2,2)
		pl.title('Minimum Safety (px)')
		pl.xlabel('IGP'); pl.ylabel('Pedestrian')
		pl.scatter(IGP_scores[:,1], ped_scores[:,1])
		plot_diag()
		
		pl.draw()

def plot_diag():
	xmin, xmax = pl.xlim()
	ymin, ymax = pl.ylim()
	lim = (min(0, min(xmin, ymin)), max(xmax, ymax))
	pl.plot((0, 1000), (0, 1000), 'k')
	pl.xlim(lim); pl.ylim(lim)

def draw_text(frame, pt, frame_txt):
	font = cv2.FONT_HERSHEY_SIMPLEX
	scale = 0.6
	thickness = 1
	sz, baseline = cv2.getTextSize(frame_txt, font, scale, thickness)
	baseline += thickness
	lower_left = (pt[0], pt[1])
	pt = (pt[0], pt[1]-baseline)
	upper_right = (pt[0]+sz[0], pt[1]-sz[1]-2)
	cv2.rectangle(frame, lower_left, upper_right, (0,0,0), -1, cv2.CV_AA)
	cv2.putText(frame, frame_txt, pt, font, scale, (0,255,0), thickness, cv2.CV_AA)
	return lower_left, upper_right

def crossline(curr, prev, length):
	diff = curr - prev
	if diff[1] == 0:
		p1 = (int(curr[1]), int(curr[0]-length/2))
		p2 = (int(curr[1]), int(curr[0]+length/2))
	else:
		slope = -diff[0]/diff[1]
		x = np.cos(np.arctan(slope)) * length / 2
		y = slope * x
		p1 = (int(curr[1]-y), int(curr[0]-x))
		p2 = (int(curr[1]+y), int(curr[0]+x))
	return p1, p2

def draw_path(frame, path, color):
	if path.shape[0] > 0:
		prev = path[0]
		for curr in path[1:]:
			loc1 = (int(prev[1]), int(prev[0])) # (y, x)
			loc2 = (int(curr[1]), int(curr[0])) # (y, x)
			p1, p2 = crossline(curr, prev, 3)
			cv2.line(frame, p1, p2, color, 1, cv2.CV_AA)
			cv2.line(frame, loc1, loc2, color, 1, cv2.CV_AA)
			prev = curr

def draw_waypoints(frame, points, color):
	for loc in ((int(y), int(x)) for x,y,z in points):
		cv2.circle(frame, loc, 3, color, -1, cv2.CV_AA)
