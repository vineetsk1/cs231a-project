#!/usr/bin/env python

import rospy
import sys
import os
import argparse
import time

import numpy as np
import tensorflow as tf
import pickle

from social_utils import SocialDataLoader
from social_model import SocialModel
from grid import getSequenceGridMask

from spencer_tracking_msgs.msg import TrackedPersons
from people_msgs.msg import PeoplePrediction
from people_msgs.msg import People
from people_msgs.msg import Person
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

# Social lstm trained on drone dataset with obs 8, no relative position
# CHK_DIR = '/cvgl2/u/junweiy/Jackrabbot/checkpoints/obs-8-2/social/08_28_00_45'

# Social lstm trained on drone dataset with obs 4, no relative position
# CHK_DIR = '/cvgl2/u/junweiy/Jackrabbot/checkpoints/obs-8-2/social/08_28_00_45'

# Naive lstm trained on simulation data with obs 8
# CHK_DIR = '/cvgl2/u/junweiy/Jackrabbot/checkpoints/naive-lstm/naive/09_06_19_47'

# Naive lstm trained on simulation + drone data with obs 8
# CHK_DIR = '/cvgl2/u/junweiy/Jackrabbot/checkpoints/naive-lstm/naive/09_07_16_18'

# Occupancy lstm trained on simulation + drone dataset with obs 8
CHK_DIR = '/cvgl2/u/junweiy/Jackrabbot/checkpoints/naive-lstm/occupancy/09_10_13_13'
class Social_Lstm_Prediction():
    def __init__(self):
        self.node_name = 'social_lstm'
        rospy.init_node(self.node_name)
        rospy.on_shutdown(self.cleanup)

        self.obs_length = 8
        self.pred_length = 8
        self.frame_interval = 2
        self.max_pedestrians = 60
        self.dimensions = [0, 0]
        self.fps = 15
        self.frame_interval_index = 0
        self.time_resolution = float(self.frame_interval * self.obs_length / self.fps)

        # Define the path for the config file for saved args
        with open(os.path.join(CHK_DIR, 'social_config.pkl'), 'rb') as f:
            self.saved_args = pickle.load(f)
            
        rospy.loginfo("Creating the model.  This can take about 10 minutes...")

        # Create a SocialModel object with the saved_args and infer set to true
        self.social_lstm_model = SocialModel(self.saved_args, True)
        
        rospy.loginfo("Model created.")

        # Initialize a TensorFlow session
        self.sess = tf.InteractiveSession()

        # Initialize a saver
        saver = tf.train.Saver()

        # Get the checkpoint state for the model
        ckpt = tf.train.get_checkpoint_state(CHK_DIR)
        print ('Loading model: ', ckpt.model_checkpoint_path)

        # Restore the model at the checkpoint
        saver.restore(self.sess, ckpt.model_checkpoint_path)

        # Dict of person_id -> [row_index in obs_seq]
        self.id_index_dict = {}
        self.vacant_rows = range(self.max_pedestrians)
        self.frame_num = 0
        self.obs_sequence = np.zeros((self.obs_length * self.frame_interval + self.frame_interval/2, self.max_pedestrians, 3))

        self.tracked_persons_sub = rospy.Subscriber("tracked_persons", TrackedPersons, self.predict, queue_size=1)
        self.pedestrian_prediction_pub = rospy.Publisher("predicted_persons", PeoplePrediction, queue_size=1)
        self.prediction_marker_pub = rospy.Publisher("predicted_persons_marker_array", MarkerArray, queue_size=1)

        rospy.loginfo("Waiting for tracked persons...")
        rospy.loginfo("Ready.")

    def predict(self, tracked_persons):
        print "********************** Processing new frame ******************************"
        start_time = time.time()

        # Initialize the markers array
        prediction_markers = MarkerArray()

        # Initialize the people predictions message
        people_predictions = PeoplePrediction()

        tracks = tracked_persons.tracks
        track_ids = [track.track_id for track in tracks]

        print "Number of people being tracked: ", len(tracks)
        print 'track ids in current frame is ', track_ids

        self.frame_num += 1
        # self.frame_interval_index += 1
        self.obs_sequence = np.delete(self.obs_sequence, 0, axis=0)
        
        existing_track_ids = self.obs_sequence[:, :, 0]
        # print 'existing track ids: ', existing_track_ids.shape, existing_track_ids
        for track_id in self.id_index_dict.keys():
            if track_id not in existing_track_ids:
                self.vacant_rows.append(self.id_index_dict[track_id])
                del self.id_index_dict[track_id]


        curr_frame = np.zeros((1, self.max_pedestrians, 3))
        for track in tracks:
            track_id = track.track_id
            if track_id in self.id_index_dict:
                row_index = self.id_index_dict[track_id]
            else:
                row_index = self.vacant_rows[0]
                # print 'vacant row is: ', self.vacant_rows
                del self.vacant_rows[0]
                self.id_index_dict[track_id] = row_index
            # print 'row_index is: ', row_index
            curr_frame[0, row_index, :] = [track_id, track.pose.pose.position.x, track.pose.pose.position.y]

        self.obs_sequence = np.concatenate((self.obs_sequence, curr_frame), axis=0)

        if len(tracks) == 0 or self.frame_num < self.fps: # or self.frame_interval_index < self.frame_interval:
            self.pedestrian_prediction_pub.publish(people_predictions)
            self.prediction_marker_pub.publish(prediction_markers)
            return

        print "This is a predicting step............................"

        interpolated_obs_sequence = np.zeros((self.obs_sequence.shape[0], self.max_pedestrians, 3))
        for i in range(self.max_pedestrians):
            f = np.nonzero(self.obs_sequence[:, i, 0])[0]
            if np.size(f) == 0 or f[np.size(f) - 1] < self.obs_sequence.shape[0] - self.frame_interval:
                continue
            
            x = self.obs_sequence[f, i, 1]
            y = self.obs_sequence[f, i, 2]

            x_interpolate = np.polyfit(f, x, 1)
            y_interpolate = np.polyfit(f, y, 1)

            fx = np.poly1d(x_interpolate)
            fy = np.poly1d(y_interpolate)

            fnew = np.linspace(0, self.obs_sequence.shape[0], num=self.obs_sequence.shape[0], endpoint=True)
            xnew = fx(fnew)
            ynew = fy(fnew)

            interpolated_obs_sequence[:, i, 1] = xnew
            interpolated_obs_sequence[:, i, 2] = ynew

            pedId = self.obs_sequence[f[np.size(f) - 1], i, 0]
            interpolated_obs_sequence[:, i, 0] = pedId

            print 'before interpolation: ', self.obs_sequence[:, i, :]
            print 'x interpolation: ', xnew
            print 'y interpolation: ', ynew
        
        mean_interpolated_obs_sequence = np.zeros((self.obs_length, self.max_pedestrians, 3))
        for step in range(self.obs_length, 0, -1):
            end_step = step * self.frame_interval + self.frame_interval/2 - 1
            start_step = end_step - self.frame_interval
            curr_seq = interpolated_obs_sequence[start_step: end_step + 1, :, :]

            mean_seq_cords = np.mean(curr_seq[:, :, :], axis=0)
            mean_interpolated_obs_sequence[step - 1, :, :] = mean_seq_cords

        # # Generate interpolated obs_sequence
        # for step in range(self.obs_length, 0, -1):
        #     end_step = step * self.frame_interval + self.frame_interval/2 - 1
        #     start_step = end_step - self.frame_interval
        #     curr_seq = self.obs_sequence[start_step: end_step + 1, :, :]

        #     # mean_seq_cords = np.mean(curr_seq[:, :, 1:], axis=0)
        #     mean_seq_cords = np.zeros((self.max_pedestrians, 2))
        #     for i in range(self.max_pedestrians):
        #         nonzeros_count = 0.0
        #         for j in range(self.frame_interval + 1):
        #             if curr_seq[j, i, 0] != 0.0:
        #                 nonzeros_count += 1.0
        #                 mean_seq_cords[i, :] += curr_seq[j, i, 1:]

        #         if nonzeros_count != 0.0:
        #             mean_seq_cords[i, :] = mean_seq_cords[i, :] / nonzeros_count


        #     all_zeros_rows = np.where(~mean_seq_cords.any(axis=1))[0]
            
        #     non_zeros_rows = np.where(mean_seq_cords.any(axis=1))[0]
        #     print non_zeros_rows
        #     nonzeros = np.nonzero(curr_seq[:, :, 0])
        #     print np.unique(nonzeros[1])
        #     if step < self.obs_length:
        #         mean_seq_cords[all_zeros_rows, :] = interpolated_obs_sequence[step, all_zeros_rows, 1:]
        #         interpolated_obs_sequence[step - 1, all_zeros_rows, 0] = interpolated_obs_sequence[step, all_zeros_rows, 0]

        #     interpolated_obs_sequence[step - 1, :, 1:] = mean_seq_cords
        #     interpolated_obs_sequence[step - 1, nonzeros[1], 0] = curr_seq[nonzeros[0], nonzeros[1], 0]

        #     # for seq_frame_index in range(self.frame_interval + 1):
        #     #     for pedIndex in non_zeros_rows:
        #     #         if curr_seq[seq_frame_index, pedIndex, 0] != 0:
        #     #             self.interpolated_obs_sequence[step - 1, pedIndex, 0] = curr_seq[seq_frame_index, pedIndex, 0]

        x_batch = np.concatenate((mean_interpolated_obs_sequence, np.zeros((self.pred_length, self.max_pedestrians, 3))), axis=0)
        grid_batch = getSequenceGridMask(x_batch, self.dimensions, self.saved_args.neighborhood_size, self.saved_args.grid_size)

        current_peds = np.nonzero(mean_interpolated_obs_sequence[self.obs_length - 1, :, 0])
        print 'interpolated obs sequence: ', mean_interpolated_obs_sequence[:, current_peds, :]
        complete_traj = self.social_lstm_model.sample(self.sess, mean_interpolated_obs_sequence, x_batch, grid_batch, self.dimensions, self.pred_length)
        print 'complete trajectory: ', complete_traj[:, current_peds, :]

        for frame_index in range(self.pred_length):
            people_one_time_step = People()
            people_one_time_step.header.stamp = tracked_persons.header.stamp + rospy.Duration(frame_index * self.time_resolution)
            people_one_time_step.header.frame_id = tracked_persons.header.frame_id
            
            predicted_frame_index = frame_index + self.obs_length
            
            for person_index in range(self.max_pedestrians):
                track_id = complete_traj[predicted_frame_index, person_index, 0]
                
                if track_id not in track_ids:
                   continue
               
                x_coord = complete_traj[predicted_frame_index, person_index, 1]
                y_coord = complete_traj[predicted_frame_index, person_index, 2]

                person_one_time_step = Person()
                person_one_time_step.name = str(track_id)

                point = Point()
                point.x = x_coord
                point.y = y_coord
                person_one_time_step.position = point
                people_one_time_step.people.append(person_one_time_step)
                
                prediction_marker = Marker()
                prediction_marker.type = Marker.SPHERE
                prediction_marker.action = Marker.MODIFY
                prediction_marker.ns = "predictor"
                prediction_marker.lifetime = rospy.Duration(0.1)
                prediction_marker.pose.orientation.w = 1
                prediction_marker.color.r = 0
                prediction_marker.color.g = 0
                prediction_marker.color.b = 0.5
                prediction_marker.scale.x = 0.2
                prediction_marker.scale.y = 0.2
                prediction_marker.scale.z = 0.2
                
                prediction_marker.header.stamp = tracked_persons.header.stamp
                prediction_marker.header.frame_id = tracked_persons.header.frame_id
                prediction_marker.id = int(frame_index + person_index * self.pred_length)
                prediction_marker.pose.position.x = person_one_time_step.position.x
                prediction_marker.pose.position.y = person_one_time_step.position.y
                #prediction_marker.color.a = 1 - (frame_index * 1.0 / (self.pred_length * 1.0))
                prediction_marker.color.a = 1.0
                prediction_markers.markers.append(prediction_marker)

            people_predictions.predicted_people.append(people_one_time_step)
     
        # print people_predictions 

        self.frame_interval_index = 0
        print 'time spent for frame: ', time.time() - start_time
        self.pedestrian_prediction_pub.publish(people_predictions)
        self.prediction_marker_pub.publish(prediction_markers)


    def cleanup(self):
        print "Shutting down social lstm node"


def main(args):
    try:
        Social_Lstm_Prediction()
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down social lstm node."

if __name__ == '__main__':
    main(sys.argv)
