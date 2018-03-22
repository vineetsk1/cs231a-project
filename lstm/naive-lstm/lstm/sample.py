import numpy as np
import tensorflow as tf

import os
import pickle
import argparse

from utils import DataLoaderNew
from model import Model


def get_mean_error(predicted_traj, true_traj, observed_length):
    '''
    Function that computes the mean euclidean distance error between the
    predicted and the true trajectory
    params:
    predicted_traj : numpy matrix with the points of the predicted trajectory
    true_traj : numpy matrix with the points of the true trajectory
    observed_length : The length of trajectory observed
    '''
    # The data structure to store all errors
    error = np.zeros(len(true_traj) - observed_length)
    # For each point in the predicted part of the trajectory
    for i in range(observed_length, len(true_traj)):
        # The predicted position
        pred_pos = predicted_traj[i, :]
        # The true position
        true_pos = true_traj[i, :]

        # The euclidean distance is the error
        error[i-observed_length] = np.linalg.norm(true_pos - pred_pos)

    # Return the mean error
    return np.mean(error)


def main():
    # Define the parser
    parser = argparse.ArgumentParser()
    # Observed length of the trajectory parameter
    parser.add_argument('--obs_length', type=int, default=8, # 5
                        help='Observed length of the trajectory')
    # Predicted length of the trajectory parameter
    parser.add_argument('--pred_length', type=int, default=12, # 5
                        help='Predicted length of the trajectory')
    # Test dataset
    parser.add_argument('--test_dataset', type=int, default=1, # NOT USED
                        help='Dataset to be tested on')

    # Read the arguments
    sample_args = parser.parse_args()

    # Load the saved arguments to the model from the config file
    with open(os.path.join('save_lstm', 'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)

    # Initialize with the saved args
    model = Model(saved_args, True)
    # Initialize TensorFlow session
    sess = tf.InteractiveSession()
    # Initialize TensorFlow saver
    saver = tf.train.Saver()

    # Get the checkpoint state to load the model from
    ckpt = tf.train.get_checkpoint_state('save_lstm')
    print('loading model: ', ckpt.model_checkpoint_path)

    # Restore the model at the checpoint
    saver.restore(sess, ckpt.model_checkpoint_path)

    # Dataset to get data from
    # dataset = range(4) #[sample_args.test_dataset]
    # dataset = [38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57]
    dataset = [51]
    # dataset = [0, 1, 2]
    # dataset = [1]

    # Initialize the dataloader object to
    # Get sequences of length obs_length+pred_length
    data_loader = DataLoaderNew(1, sample_args.pred_length + sample_args.obs_length, dataset, True)

    mod_folder = data_loader.data_dirs[dataset[0]]
    mod_fname = mod_folder[:-1][mod_folder[:-1].rfind("/")+1:]
    mod_orig = mod_folder[:-1] + ".txt"
    mod_new = mod_folder[mod_folder.find("/gt/")+4:]
    mod_new = mod_new[:mod_new.find("/")]
    mod_new = "predict/" + mod_new + "/" + mod_fname + ".txt"
    mod_data = np.loadtxt(mod_orig)

    # Reset the data pointers of the data loader object
    data_loader.reset_batch_pointer()

    # Maintain the total_error until now
    total_error = 0.
    counter = 0.

    for b in range(data_loader.num_batches):
        # Get the source, target data for the next batch
        x, y = data_loader.next_batch()

        # The observed part of the trajectory
        obs_traj = x[0][:sample_args.obs_length]
        # Get the complete trajectory with both the observed and the predicted part from the model
        complete_traj = model.sample(sess, obs_traj, x[0], num=sample_args.pred_length) # contains given observed and the predicted

        # print mod_data[0:30]
        mod_data[20*b:20*(b+1), 2] = complete_traj[:, 0]
        mod_data[20*b:20*(b+1), 3] = complete_traj[:, 1]
        # print mod_data[0:30]
        # import sys
        # sys.exit(0)

        # print obs_traj
        # print complete_traj
        # if b > 5:
        #     import sys
        #     sys.exit(0)

        # Compute the mean error between the predicted part and the true trajectory
        total_error += get_mean_error(complete_traj, x[0], sample_args.obs_length)
        print "Processed trajectory number : ", b, "out of ", data_loader.num_batches, " trajectories"

    # Print the mean error across all the batches
    np.savetxt(mod_new, mod_data)
    print "Total mean error of the model is ", total_error/data_loader.num_batches

if __name__ == '__main__':
    main()
