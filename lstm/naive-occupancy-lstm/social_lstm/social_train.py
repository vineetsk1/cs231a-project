import tensorflow as tf
import argparse
import os
import sys
import time
import pickle
import numpy as np
#import ipdb

from social_model import SocialModel
from social_utils import SocialDataLoader
from grid import getSequenceGridMask

# CHK_DIR = '/cvgl2/u/junweiy/Jackrabbot/checkpoints/test'
CHK_DIR = '/cvgl2/u/junweiy/Jackrabbot/checkpoints/naive-lstm'

def main():
    parser = argparse.ArgumentParser()
    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    # TODO: (improve) Number of layers not used. Only a single layer implemented
    # Number of layers parameter
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    # Model currently not used. Only LSTM implemented
    # Type of recurrent unit parameter
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, or lstm')
    # Size of each batch parameter
    parser.add_argument('--batch_size', type=int, default=10,
                        help='minibatch size')
    # Dimension of the embeddings parameter
    parser.add_argument('--embedding_size', type=int, default=64,
                        help='Embedding dimension for the spatial coordinates')
    
    
    # Length of sequence to be considered parameter
    parser.add_argument('--seq_length', type=int, default=20,
                        help='RNN sequence length')
    # Length of sequence to be considered parameter
    parser.add_argument('--obs_length', type=int, default=8,
                        help='Observed length of frames in a sequence')
    # Length of sequence to be considered parameter
    parser.add_argument('--pred_length', type=int, default=8,
                        help='Predicted length of frames in a sequence')

    # Number of splits for the dataset training and testing
    parser.add_argument('--num_splits', type=int, default=5, 
                        help='number of splits')
    # Number of epochs parameter
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')
    # Frequency at which the model should be saved parameter
    parser.add_argument('--save_every', type=int, default=50,
                        help='save frequency')
    
    
    # TODO: (resolve) Clipping gradients for now. No idea whether we should
    # Gradient value at which it should be clipped
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    # Learning rate parameter
    parser.add_argument('--learning_rate', type=float, default=0.005,
                        help='learning rate')
    # Decay rate for the learning rate parameter
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='decay rate for rmsprop')
    # Dropout not implemented. Dropout probability parameter
    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='dropout keep probability')


    # Size of neighborhood to be considered parameter
    parser.add_argument('--neighborhood_size', type=float, default=32,
                        help='Neighborhood size to be considered for social grid')   
    # Size of the social grid parameter
    parser.add_argument('--grid_size', type=int, default=4,
                        help='Grid size of the social grid')
    # Maximum number of pedestrians to be considered
    parser.add_argument('--maxNumPeds', type=int, default=60,
                        help='Maximum Number of Pedestrians')

    parser.add_argument('--dataset_path', type=str, default='./../data/',
                        help='Path training data')
    parser.add_argument('--visible',type=str,
                        required=False, default=None, help='GPU to run on')
    parser.add_argument('--mode', type=str, default='occupancy', 
                        help='social, occupancy, naive')
    parser.add_argument('--model_path', type=str)


    args = parser.parse_args()
    train(args)

def get_mean_error(predicted_traj, true_traj, observed_length, maxNumPeds):
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

    total_error = 0.0
    counter = 0.0
    for i in range(observed_length, len(true_traj)):
        # The predicted position. This will be a maxNumPeds x 3 matrix
        pred_pos = predicted_traj[i, :]
        # The true position. This will be a maxNumPeds x 3 matrix
        true_pos = true_traj[i, :]
        # timestep_error = 0
        # counter = 0
        for j in range(maxNumPeds):
            if true_pos[j, 0] == 0:
                # Non-existent ped
                continue
            elif pred_pos[j, 0] == 0:
                # Ped comes in the prediction time. Not seen in observed part
                continue
            else:
                # timestep_error += np.linalg.norm(true_pos[j, [1, 2]] - pred_pos[j, [1, 2]])
                total_error += np.linalg.norm(true_pos[j, [1, 2]] - pred_pos[j, [1, 2]])
                counter += 1

        # if counter != 0:
        #     error[i - observed_length] = timestep_error / counter

        # The euclidean distance is the error
        # error[i-observed_length] = np.linalg.norm(true_pos - pred_pos)

    # Return the mean error
    # return np.mean(error)
    return total_error, counter

def make_save_path(args):
    import datetime
    folder_name = args.mode
    now = datetime.datetime.now()
    timestamp = now.strftime("%m_%d_%H_%M")
    save_path = os.path.join(CHK_DIR, folder_name, timestamp)

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    return save_path

def train(args):
    if args.visible:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible

    if args.model_path:
        save_path = args.model_path
    else:
        save_path = make_save_path(args)
    
    dataset_path = args.dataset_path
    log_path = os.path.join(save_path, 'log')
    if not os.path.isdir(log_path):
        os.makedirs(log_path)

    with open(os.path.join(save_path, 'social_config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    # Create a SocialModel object with the arguments
    model = SocialModel(args)
    
    # Variables to store losses
    all_train_loss = []
    all_test_loss = []
    best_epoch = -1
    best_test_loss = 0.0
    
    # Initialize a TensorFlow session
    with tf.Session() as sess:
        # Get the checkpoint state for the model
        ckpt = tf.train.get_checkpoint_state(save_path)

        if ckpt:
            # Restore the model at the checkpoint
            print 'Loading model: ', ckpt.model_checkpoint_path
            saver = tf.train.Saver(max_to_keep=50)
            summary_writer = tf.summary.FileWriter(log_path, sess.graph)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print 'Initializing variables....'
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=50)
            summary_writer = tf.summary.FileWriter(log_path, sess.graph)

        # For each epoch
        for e in range(args.num_epochs):
            # Assign the learning rate value for this epoch
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))

            # Create the training and testing SocialDataLoader objects
            data_files = [os.path.join(dataset_path, _file) for _file in os.listdir(dataset_path) if _file.endswith('csv')]
            # print data_files
            test_index = e % args.num_splits
            num_files = len(data_files) / args.num_splits
            
            split_data_files = [data_files[i: i + num_files] for i in range(0, len(data_files), num_files)]
            # print split_data_files
            train_data_files = [datafile for i in range(args.num_splits) if i != test_index 
                                for datafile in split_data_files[i]]
            test_data_files = split_data_files[test_index]
            if len(split_data_files) > args.num_splits:
                test_data_files.extend(split_data_files[args.num_splits])

            train_data_loader = SocialDataLoader(args.batch_size, args.seq_length,
                args.maxNumPeds, dataset_path, train_data_files, forcePreProcess=True)
            test_data_loader = SocialDataLoader(1, args.obs_length + args.pred_length, 
                args.maxNumPeds, dataset_path, test_data_files, forcePreProcess=True)

            print 'Training files: ', train_data_files
            print 'Testing files: ', test_data_files
            print 'Number of training batches: ', train_data_loader.num_batches
            print 'Number of testing batches: ', test_data_loader.num_batches

            # Training
            for b in range(train_data_loader.num_batches):
                # Tic
                start = time.time()

                # Get the source, target and dataset data for the next batch
                # s_batch, t_batch are input and target data which are lists containing numpy arrays of size seq_length x maxNumPeds x 3
                # d is the list of dataset indices from which each batch is generated (used to differentiate between datasets)
                s_batch, t_batch, d = train_data_loader.next_batch()

                # variable to store the loss for this batch
                loss_batch = 0
                counter = 0

                # For each sequence in the batch
                for seq_num in range(train_data_loader.batch_size):
                    # s_seq, t_seq and d_batch contains the source, target and dataset index data for
                    # seq_length long consecutive frames in the dataset
                    # s_seq, t_seq would be numpy arrays of size seq_length x maxNumPeds x 3
                    # d_batch would be a scalar identifying the dataset from which this sequence is extracted
                    s_seq, t_seq, d_seq = s_batch[seq_num], t_batch[seq_num], d[seq_num]
                    
                    print 'Processing frame sequence ' + str(seq_num) + '.....................'
                    for starting_frame_index in range(args.seq_length - args.obs_length - args.pred_length):
                        sub_s_seq = s_seq[starting_frame_index:starting_frame_index + args.obs_length + args.pred_length, :, :]
                        sub_t_seq = t_seq[starting_frame_index:starting_frame_index + args.obs_length + args.pred_length, :, :]

                        grid_batch = getSequenceGridMask(sub_s_seq, [0, 0], args.neighborhood_size, args.grid_size)

                        # Feed the source, target data
                        feed = {model.input_data: sub_s_seq, model.target_data: sub_t_seq, model.grid_data: grid_batch}

                        train_loss, train_counter, _ = sess.run([model.cost, model.counter, model.train_op], feed)

                        loss_batch += train_loss

                        counter += train_counter

                end = time.time()
                loss_batch = loss_batch / counter
                all_train_loss.append(loss_batch)
                print(
                    "{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
                    .format(
                        e * train_data_loader.num_batches + b,
                        args.num_epochs * train_data_loader.num_batches,
                        e,
                        loss_batch, end - start))

                # Save the model for each epoch
                # if (e * data_loader.num_batches + b) % args.save_every == 0 and ((e * data_loader.num_batches + b) > 0):
                #     checkpoint_path = os.path.join(save_path, 'social_model.ckpt')
                #     saver.save(sess, checkpoint_path, global_step=e * train_data_loader.num_batches + b)
                #     print("model saved to {}".format(checkpoint_path))
                np.savetxt(os.path.join(log_path, 'train_loss.txt'), np.asarray(all_train_loss))


            # Testing
            print '************************** Testing for current epoch *******************************'
            test_error = 0.0
            test_counter = 0.0
            
            # For each batch
            for b in range(test_data_loader.num_batches):
                # Get the source, target and dataset data for the next batch
                x, y, d = test_data_loader.next_batch(randomUpdate=False)

                # Batch size is 1
                x_batch, y_batch, d_batch = x[0], y[0], d[0]

                grid_batch = getSequenceGridMask(x_batch, [0,0], args.neighborhood_size, args.grid_size)

                # obs_traj is an array of shape obs_length x maxNumPeds x 3
                obs_traj = x_batch[:args.obs_length]

                print "Processed trajectory number : ", b, "out of ", test_data_loader.num_batches, " trajectories"
                complete_traj = model.sample(sess, obs_traj, x_batch, grid_batch, [0,0], args.pred_length)

                # complete_traj is an array of shape (obs_length+pred_length) x maxNumPeds x 3
                error, counter = get_mean_error(complete_traj, x[0], args.obs_length, args.maxNumPeds)
                test_error += error
                test_counter += counter

            # Print the mean error across all the batches
            test_epoch_loss = test_error/test_counter
            all_test_loss.append(test_epoch_loss)
            print "Average test error of the model is ", test_epoch_loss

            if e == 0:
                best_epoch = e
                best_test_loss = test_epoch_loss
            else:
                if test_epoch_loss < best_test_loss:
                    best_epoch = e
                    best_test_loss = test_epoch_loss

            # Save checkpoints for current epoch
            checkpoint_path = os.path.join(save_path, 'social_model.ckpt')
            saver.save(sess, checkpoint_path, global_step=e * train_data_loader.num_batches)
            print("model saved to {}".format(checkpoint_path))
            
            np.savetxt(os.path.join(log_path, 'loss.txt'), np.asarray(all_train_loss))
            np.savetxt(os.path.join(log_path, 'test_loss.txt'), np.asarray(all_test_loss))
            np.savetxt(os.path.join(log_path, 'best_epoch.txt'), np.asarray([best_epoch, best_test_loss]))

if __name__ == '__main__':
    main()
