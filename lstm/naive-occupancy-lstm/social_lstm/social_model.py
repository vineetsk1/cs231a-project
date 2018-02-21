'''
Social LSTM model implementation using Tensorflow
Social LSTM Paper: http://vision.stanford.edu/pdf/CVPR16_N_LSTM.pdf

Author : Anirudh Vemula
Date: 17th October 2016
'''

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn_cell
from grid import getSequenceGridMask
#import ipdb
import time

class SocialModel():

    def __init__(self, args, infer=False):
        '''
        Initialisation function for the class SocialModel
        params:
        args : Contains arguments required for the model creation
        '''

        # If sampling new trajectories, then infer mode
        if infer:
            # Sample one position at a time
            args.batch_size = 1

        # Store the arguments
        self.args = args
        self.infer = infer
        self.mode = args.mode

        # Store rnn size and grid_size
        self.rnn_size = args.rnn_size
        self.grid_size = args.grid_size

        # Maximum number of peds
        self.maxNumPeds = args.maxNumPeds

        # NOTE : 
        # For now assuming, batch_size is always 1. 
        # That is the input to the model is always a sequence of frames

        # Construct the basicLSTMCell recurrent unit with a dimension given by args.rnn_size
        with tf.name_scope("LSTM_cell"):
            cell = rnn_cell.BasicLSTMCell(args.rnn_size, state_is_tuple=False)

        # placeholders for the input data and the target data
        # A sequence contains an ordered set of consecutive frames
        # Each frame can contain a maximum of 'args.maxNumPeds' number of peds
        # For each ped we have their (pedID, x, y) positions as input
        self.input_data = tf.placeholder(tf.float32, [args.obs_length + args.pred_length, args.maxNumPeds, 3], name="input_data")

        # target data would be the same format as input_data except with
        # one time-step ahead
        self.target_data = tf.placeholder(tf.float32, [args.obs_length + args.pred_length, args.maxNumPeds, 3], name="target_data")

        # Grid data would be a binary matrix which encodes whether a pedestrian is present in
        # A grid cell of other pedestrian
        self.grid_data = tf.placeholder(tf.float32, [args.obs_length + args.pred_length, args.maxNumPeds, args.maxNumPeds, args.grid_size*args.grid_size], name="grid_data")

        # Variable to hold the value of the learning rate
        self.lr = tf.Variable(args.learning_rate, trainable=False, name="learning_rate")

        # Output dimension of the model
        self.output_size = 2

        # Define embedding and output layers
        self.define_embedding_and_output_layers(args)

        # Define LSTM cell states for each pedestrian
        with tf.variable_scope("LSTM_states"):
            self.LSTM_states = tf.zeros([args.maxNumPeds, cell.state_size], name="LSTM_states")
            self.cell_states = tf.split(self.LSTM_states, args.maxNumPeds, 0)

        # Define hidden states for each pedestrian
        with tf.variable_scope("Output_states"):
            self.hidden_states = tf.split(tf.zeros([args.maxNumPeds, cell.output_size]), args.maxNumPeds, 0)

        # List of tensors each of shape args.maxNumPedsx3 corresponding to each frame in the sequence
        with tf.name_scope("frame_data_tensors"):
            seq_data = [tf.squeeze(input_, [0]) for input_ in tf.split(self.input_data, args.obs_length + args.pred_length, 0)]

        with tf.name_scope("frame_target_data_tensors"):
            frame_target_data = [tf.squeeze(target_, [0]) for target_ in tf.split(self.target_data, args.obs_length + args.pred_length, 0)]

        with tf.name_scope("grid_frame_data_tensors"):
            # This would contain a list of tensors each of shape MNP x MNP x (GS**2) encoding the mask
            grid_frame_data = [tf.squeeze(input_, [0]) for input_ in tf.split(self.grid_data, args.obs_length + args.pred_length, 0)]

        # Cost
        with tf.name_scope("Cost_related_stuff"):
            self.cost = tf.constant(0.0, name="cost")
            self.counter = tf.constant(0.0, name="counter")
            self.increment = tf.constant(1.0, name="increment")

        # Containers to store output distribution parameters
        with tf.name_scope("Distribution_parameters_stuff"):
            self.initial_output = tf.split(tf.zeros([args.maxNumPeds, self.output_size]), args.maxNumPeds, 0)

        # Tensor to represent non-existent ped
        with tf.name_scope("Non_existent_ped_stuff"):
            nonexistent_ped = tf.constant(0.0, name="zero_ped")

        with tf.name_scope("Predicted_traj"):
            self.pred_traj = tf.zeros([args.pred_length, args.maxNumPeds, 3])

        # Fit the obs_length and predict pred_length steps
        newpos = tf.zeros([1, args.maxNumPeds, 3])
        current_frame_data = seq_data[1]
        prev_frame_data = seq_data[0]
        
        # Iterate over each frame in the sequence
        for frame_num in range(args.obs_length + args.pred_length - 1):
            if frame_num == 0:
                continue

            if frame_num < args.obs_length - 1:
                is_pred = False
            else:
                is_pred = True

            if frame_num > 1:
                prev_frame_data = current_frame_data

            if frame_num >= args.obs_length:
                current_frame_data = newpos
                current_grid_frame_data = grid_frame_data[args.obs_length - 1]
            else:
                current_frame_data = seq_data[frame_num] # MNP x 3 tensor
                current_grid_frame_data = grid_frame_data[frame_num]  # MNP x MNP x (GS**2) tensor
        
            if self.mode != 'naive':
                social_tensor = self.getSocialTensor(current_grid_frame_data, self.hidden_states)  

            for ped in range(args.maxNumPeds):
                # print "Pedestrian Number", ped

                # pedID of the current pedestrian
                pedID = current_frame_data[ped, 0]

                with tf.name_scope("extract_input_ped"):
                    # Extract relative x and y positions of the current ped
                    self.spatial_input = tf.slice(current_frame_data - prev_frame_data, [ped, 1], [1, 2])  # Tensor of shape (1,2)
                    
                    # Extract the social tensor of the current ped
                    if self.mode == 'social':
                        self.tensor_input = tf.slice(social_tensor, [ped, 0], [1, args.grid_size*args.grid_size*args.rnn_size])  
                    elif self.mode == 'occupancy':
                        self.tensor_input = tf.expand_dims(social_tensor[ped], 0)

                with tf.name_scope("embeddings_operations"):
                    # Embed the spatial input
                    embedded_spatial_input = tf.nn.relu(tf.nn.xw_plus_b(self.spatial_input, self.embedding_w, self.embedding_b))
                    
                    # Embed the tensor input
                    if self.mode != 'naive':
                        embedded_tensor_input = tf.nn.relu(tf.nn.xw_plus_b(self.tensor_input, self.embedding_t_w, self.embedding_t_b))

                with tf.name_scope("concatenate_embeddings"):
                    complete_input = embedded_spatial_input

                    # Concatenate the embeddings
                    if self.mode != 'naive':
                        complete_input = tf.concat([embedded_spatial_input, embedded_tensor_input], 1)

                # One step of LSTM
                with tf.variable_scope("LSTM") as scope:
                    if frame_num > 1 or ped > 0:
                        scope.reuse_variables()
                    self.hidden_states[ped], self.cell_states[ped] = cell(complete_input, self.cell_states[ped])

                # Apply the linear layer. Output would be a tensor of shape 1 x output_size
                with tf.name_scope("output_linear_layer"):
                    self.initial_output[ped] = tf.nn.xw_plus_b(self.hidden_states[ped], self.output_w, self.output_b)
                    next_x = self.initial_output[ped][0][0] + current_frame_data[ped, 1]
                    next_y = self.initial_output[ped][0][1] + current_frame_data[ped, 2]

                    ped_position_tensor = tf.reshape([pedID, next_x, next_y], [1, -1])
                    
                    if ped == 0:
                        newpos = ped_position_tensor
                    else:
                        newpos = tf.concat([newpos, ped_position_tensor], axis=0)
                
                with tf.name_scope("extract_target_ped"):
                    # Extract x and y coordinates of the target data
                    # x_data and y_data would be tensors of shape 1 x 1
                    [x_data, y_data] = tf.split(tf.slice(frame_target_data[frame_num], [ped, 1], [1, 2]), 2, 1)
                    target_pedID = frame_target_data[frame_num][ped, 0]

                with tf.name_scope("calculate_loss"):
                    # Calculate loss for the current ped
                    lossfunc = self.simple_lossfunc(next_x, next_y, x_data, y_data)

                with tf.name_scope("increment_cost"):
                    # If it is a non-existent ped, it should not contribute to cost
                    # If the ped doesn't exist in the next frame, he/she should not contribute to cost as well
                    if is_pred:
                        self.cost = tf.where(tf.logical_or(tf.equal(pedID, nonexistent_ped), tf.equal(target_pedID, nonexistent_ped)), self.cost, tf.add(self.cost, lossfunc))
                        self.counter = tf.where(tf.logical_or(tf.equal(pedID, nonexistent_ped), tf.equal(target_pedID, nonexistent_ped)), self.counter, tf.add(self.counter, self.increment))

            if is_pred:
                if frame_num == args.obs_length - 1:
                    self.pred_traj = tf.reshape(newpos, [1, args.maxNumPeds, 3])
                elif frame_num > args.obs_length - 1:
                    self.pred_traj = tf.concat([self.pred_traj, tf.reshape(newpos, [1, args.maxNumPeds, 3])], axis=0)

        with tf.name_scope("mean_cost"):
            # Mean of the cost
            self.cost = tf.div(self.cost, 1.0)

        # Get all trainable variables
        tvars = tf.trainable_variables()

        # Compute gradients
        self.gradients = tf.gradients(self.cost, tvars)

        # Clip the gradients
        grads, _ = tf.clip_by_global_norm(self.gradients, args.grad_clip)

        # Define the optimizer
        optimizer = tf.train.RMSPropOptimizer(self.lr)

        # The train operator
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        # Merge all summmaries
        # merged_summary_op = tf.merge_all_summaries()

    def define_embedding_and_output_layers(self, args):
        # Define variables for the spatial coordinates embedding layer
        with tf.variable_scope("coordinate_embedding"):
            self.embedding_w = tf.get_variable("embedding_w", [2, args.embedding_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
            self.embedding_b = tf.get_variable("embedding_b", [args.embedding_size], initializer=tf.constant_initializer(0.1))

        # Define variables for the social tensor embedding layer
        size_factor = args.rnn_size
        if self.mode == 'occupancy':
            size_factor = 1

        with tf.variable_scope("tensor_embedding"):
            self.embedding_t_w = tf.get_variable("embedding_t_w", [args.grid_size*args.grid_size*size_factor, args.embedding_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
            self.embedding_t_b = tf.get_variable("embedding_t_b", [args.embedding_size], initializer=tf.constant_initializer(0.1))

        # Define variables for the output linear layer
        with tf.variable_scope("output_layer"):
            self.output_w = tf.get_variable("output_w", [args.rnn_size, self.output_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
            self.output_b = tf.get_variable("output_b", [self.output_size], initializer=tf.constant_initializer(0.1))

    def simple_lossfunc(self, predicted_x, predicted_y, x, y):
        cost = tf.sqrt(tf.square((predicted_x - x)[0][0]) + tf.square((predicted_y - y)[0][0]))
        return cost

    def getSocialTensor(self, grid_frame_data, hidden_state_list):
        '''
        Computes the social tensor for all the maxNumPeds in the frame
        params:
        grid_frame_data : A tensor of shape MNP x MNP x (GS**2)
        hidden_state_list : A list of tensors each of shape 1 x RNN_size of length MNP
        '''
        # Create a zero tensor of shape MNP x (GS**2) x RNN_size
        social_tensor = tf.zeros([self.args.maxNumPeds, self.grid_size*self.grid_size, self.rnn_size], name="social_tensor")
        
        # Create a list of zero tensors each of shape 1 x (GS**2) x RNN_size of length MNP
        social_tensor = tf.split(social_tensor, self.args.maxNumPeds, 0)
        
        # Concatenate list of hidden states to form a tensor of shape MNP x RNN_size * 2
        hidden_states = tf.concat(hidden_state_list, 0)
        
        # Split the grid_frame_data into grid_data for each pedestrians
        # Consists of a list of tensors each of shape 1 x MNP x (GS**2) of length MNP
        grid_frame_ped_data = tf.split(grid_frame_data, self.args.maxNumPeds, 0)
        
        # Squeeze tensors to form MNP x (GS**2) matrices
        grid_frame_ped_data = [tf.squeeze(input_, [0]) for input_ in grid_frame_ped_data]

        # For each pedestrian
        for ped in range(self.args.maxNumPeds):
            # Compute social tensor for the current pedestrian
            with tf.name_scope("tensor_calculation"):
                if self.mode == 'social':
                    social_tensor_ped = tf.matmul(tf.transpose(grid_frame_ped_data[ped]), hidden_states)
                    social_tensor[ped] = tf.reshape(social_tensor_ped, [1, self.grid_size*self.grid_size, self.rnn_size])
                elif self.mode == 'occupancy':
                    social_tensor_ped = tf.reduce_sum(grid_frame_ped_data[ped], 0)
                    social_tensor[ped] = tf.expand_dims(social_tensor_ped, 0)
                else:
                    exit("Function getSocialTensor should not be called in naive mode")
        
        # Concatenate the social tensor from a list to a tensor of shape
        # social : MNP x (GS**2) x RNN_size
        # occupancy : MNP x (GS**2)
        social_tensor = tf.concat(social_tensor, 0)
        
        # Reshape the tensor to match the dimensions MNP x (GS**2 * RNN_size)
        if self.mode == 'social':
            social_tensor = tf.reshape(social_tensor, [self.args.maxNumPeds, self.grid_size*self.grid_size*self.rnn_size])
        return social_tensor

    def sample(self, sess, obs_traj, traj, grid, dimensions, num=10):
        ret = obs_traj
        feed = {self.input_data: traj, self.grid_data: grid}
        [pred_traj] = sess.run([self.pred_traj], feed)
        ret = np.vstack((ret, np.array(pred_traj)))
        return ret
