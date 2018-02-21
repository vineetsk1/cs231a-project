import rospy
import sys
import os
import argparse

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

class Social_Lstm_Prediction():
    def __init__(self):
        self.node_name = 'social_lstm'

        rospy.init_node(self.node_name)
        rospy.on_shutdown(self.cleanup)

#         self.obs_length = 4
#         self.pred_length = 8
#         self.prev_length = 8
        self.obs_length = 8
        self.pred_length = 12
        self.prev_length = 12
        self.max_pedestrians = 40
        self.dimensions = [640, 480]
        self.time_resolution = 0.5

        # Define the path for the config file for saved args
        with open(os.path.join('save', 'social_config.pkl'), 'rb') as f:
            self.saved_args = pickle.load(f)

        # Create a SocialModel object with the saved_args and infer set to true
        self.social_lstm_model = SocialModel(self.saved_args, True)
        
        # Initialize a TensorFlow session
        self.sess = tf.InteractiveSession()
        
        # Initialize a saver
        saver = tf.train.Saver()

        # Get the checkpoint state for the model
        ckpt = tf.train.get_checkpoint_state('save')
        print ('loading model: ', ckpt.model_checkpoint_path)

        # Restore the model at the checkpoint
        saver.restore(self.sess, ckpt.model_checkpoint_path)

        self.tracked_persons_sub = rospy.Subscriber("/tracked_persons", TrackedPersons, self.predict)
        self.pedestrian_prediction_pub = rospy.Publisher("/predicted_persons", PeoplePrediction, queue_size=1)
        self.prediction_marker_pub = rospy.Publisher("/predicted_persons_marker_array", MarkerArray, queue_size=1)
        
        # Initialize the marker for people prediction
        self.prediction_marker = Marker()
        self.prediction_marker.type = Marker.SPHERE
        self.prediction_marker.action = Marker.MODIFY
        self.prediction_marker.ns = "people_predictions"
        self.prediction_marker.pose.orientation.w = 1
        self.prediction_marker.color.r = 0
        self.prediction_marker.color.g = 0
        self.prediction_marker.color.b = 0.5
        self.prediction_marker.scale.x = 0.2
        self.prediction_marker.scale.y = 0.2
        self.prediction_marker.scale.z = 0.2

        self.prev_frames = []
        for i in range(self.prev_length):
            self.prev_frames.append({})

        rospy.loginfo("Waiting for tracked persons...")
        rospy.wait_for_message("/predicted_persons", PeoplePrediction)
        rospy.loginfo("Ready.")

    def __interp_helper(self, y):
        """Helper to handle indices and logical indices of NaNs.

        Input:
            - y, 1d numpy array with possible NaNs
        Output:
            - nans, logical indices of NaNs
            - index, a function, with signature indices= index(logical_indices),
              to convert logical indices of NaNs to 'equivalent' indices
        Example:
            >>> # linear interpolation of NaNs
            >>> nans, x= nan_helper(y)
            >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        """

        return np.isnan(y), lambda z: z.nonzero()[0]


    def __interpolate_1d_array(self, before_interpolated):
        nparray = np.array(before_interpolated)
        
        index = -1
        for i in range(self.prev_length):
            if np.isnan(nparray[i]) == False:
                index = i
                break

        for i in range(index):
            nparray[i] = nparray[index]

        nans, x= self.__interp_helper(nparray)
        nparray[nans]= np.interp(x(nans), x(~nans), nparray[~nans])
        return nparray


    def __generate_input(self, tracks):
        num_tracks = len(tracks)
        whole_array = []
        for i in range(num_tracks):
            track = tracks[i]
            track_id = track.track_id
            
            history_positions_x = []
            history_positions_y = []
            for index in range(self.prev_length):
                history_positions_x.append(float('nan'))
                history_positions_y.append(float('nan'))
                if track_id in self.prev_frames[index]:
                    history_positions_x[index] = self.prev_frames[index][track_id][0]
                    history_positions_y[index] = self.prev_frames[index][track_id][1]

            print history_positions_x
            print history_positions_y
            
            history_positions_x = self.__interpolate_1d_array(history_positions_x)
            history_positions_y = self.__interpolate_1d_array(history_positions_y)
            tracks_array = np.zeros((self.obs_length, 3))
            tracks_array[:, 0] = track_id
            tracks_array[:, 1] = np.array(history_positions_x)[4:]
            tracks_array[:, 2] = np.array(history_positions_y)[4:]
            tracks_array = np.expand_dims(tracks_array, 1)

            print tracks_array

            if i == 0:
                whole_array = tracks_array
            else:
                whole_array = np.append(whole_array, tracks_array, axis=1)

        res_input = np.zeros((self.obs_length + self.prev_length, self.max_pedestrians, 3))
        res_input[:self.obs_length, :num_tracks, :] = whole_array
        return res_input

    def predict(self, tracked_persons):
        tracks = tracked_persons.tracks

        track_dict = {}
        for track in tracks:
            #print track
            #print track.pose.pose.position.x
            track_dict[track.track_id] = [track.pose.pose.position.x, track.pose.pose.position.y]
        
        del self.prev_frames[0]
        self.prev_frames.append(track_dict)

        if len(tracks) == 0:
            return

        input_data = self.__generate_input(tracks)
        #print input_data.shape
        #print input_data
        grid_batch = getSequenceGridMask(input_data, self.dimensions, self.saved_args.neighborhood_size, self.saved_args.grid_size)

        obs_traj = input_data[:self.obs_length]
        obs_grid = grid_batch[:self.obs_length]

        print "********************** PREDICT NEW TRAJECTORY ******************************"
        complete_traj = self.social_lstm_model.sample(self.sess, obs_traj, obs_grid, self.dimensions, input_data, self.pred_length)
        #print complete_traj
        
        # Initialize the markers array
        prediction_markers = MarkerArray()

        # Publish them
        people_predictions = PeoplePrediction()
        for frame_index in range(self.pred_length):
            people = People()
            people.header.stamp = tracked_persons.header.stamp + rospy.Duration(frame_index * self.time_resolution);
            people.header.frame_id = tracked_persons.header.frame_id
            
            predicted_frame_index = frame_index + self.obs_length
            for person_index in range(self.max_pedestrians):
                track_id = complete_traj[predicted_frame_index, person_index, 0]
                x_coord = complete_traj[predicted_frame_index, person_index, 1]
                y_coord = complete_traj[predicted_frame_index, person_index, 2]
                if track_id == 0:
                    break

                person = Person()
                person.name = str(track_id)

                point = Point()
                point.x = x_coord
                point.y = y_coord
                person.position = point
                people.people.append(person)
                
                self.prediction_marker.header.frame_id = tracked_persons.header.frame_id
                self.prediction_marker.header.stamp = tracked_persons.header.stamp
                self.prediction_marker.id = int(track_id);
                self.prediction_marker.pose.position.x = person.position.x
                self.prediction_marker.pose.position.y = person.position.y
                #self.prediction_marker.color.a = 1 - (frame_index * 1.0 / (self.pred_length * 1.0))
                self.prediction_marker.color.a = 1.0
                prediction_markers.markers.append(self.prediction_marker)

            people_predictions.predicted_people.append(people)
     
        #print people_predictions 

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
