'''
Utils script for the social LSTM implementation
Handles processing the input and target data in batches and sequences

Author : Anirudh Vemula
Date : 17th October 2016
'''

import os
import pickle
import numpy as np
import random

# The data loader class that loads data from the datasets considering
# each frame as a datapoint and a sequence of consecutive frames as the
# sequence.
class SocialDataLoader():

    def __init__(self, batch_size=50, seq_length=5, maxNumPeds=40,
            dataset_path='../data', forcePreProcess=False):
        '''
        Initialiser function for the SocialDataLoader class
        params:
        batch_size : Size of the mini-batch
        grid_size : Size of the social grid constructed
        forcePreProcess : Flag to forcefully preprocess the data again from csv files
        '''
        # List of data directories where raw data resides
        '''
        self.data_dirs = ['../data/eth/univ', '../data/eth/hotel',
                          '../data/ucy/zara/zara01', '../data/ucy/zara/zara02',
                          '../data/ucy/univ']
        '''
        self.data_dir = dataset_path
        self.used_data_dirs = [os.path.join(dataset_path, _file) for _file in
                os.listdir(self.data_dir)]
        # Number of datasets
        self.numDatasets = len(self.used_data_dirs)

        # Maximum number of peds in a single frame (Number obtained by checking the datasets)
        self.maxNumPeds = maxNumPeds

        # Store the arguments
        self.batch_size = batch_size
        self.seq_length = seq_length

        # Define the path in which the process data would be stored
        data_file = os.path.join(self.data_dir, "social-trajectories.cpkl")

        # If the file doesn't exist or forcePreProcess is true
        if not(os.path.exists(data_file)) or forcePreProcess:
            print("Creating pre-processed data from raw data")
            # Preprocess the data from the csv files of the datasets
            # Note that this data is processed in frames
            self.frame_preprocess(self.used_data_dirs, data_file)

        # Load the processed data from the pickle file
        self.load_preprocessed(data_file)
        # Reset all the data pointers of the dataloader object
        self.reset_batch_pointer()

    def frame_preprocess(self, data_dirs, data_file):
        '''
        Function that will pre-process the pixel_pos.csv files of each dataset
        into data with occupancy grid that can be used
        params:
        data_dirs : List of directories where raw data resides
        data_file : The file into which all the pre-processed data needs to be stored
        '''

        # all_frame_data would be a list of numpy arrays corresponding to each dataset
        # Each numpy array would be of size (numFrames, maxNumPeds, 3) where each pedestrian's
        # pedId, x, y , in each frame is stored
        all_frame_data = []
        # frameList_data would be a list of lists corresponding to each dataset
        # Each list would contain the frameIds of all the frames in the dataset
        frameList_data = []
        # numPeds_data would be a list of lists corresponding to each dataset
        # Ech list would contain the number of pedestrians in each frame in the dataset
        numPeds_data = []
        # Index of the current dataset
        dataset_index = 0

        # For each dataset
        for file_path in data_dirs:

            if os.path.splitext(file_path)[1] != '.csv':
                print file_path, os.path.splitext(file_path)[1]
                continue
            # Load the data from the csv file
            data = np.genfromtxt(file_path, delimiter=',')

            # Frame IDs of the frames in the current dataset
            frameList = np.unique(data[0, :]).tolist()
            # Number of frames
            numFrames = len(frameList)

            # Add the list of frameIDs to the frameList_data
            frameList_data.append(frameList)
            # Initialize the list of numPeds for the current dataset
            numPeds_data.append([])
            # Initialize the numpy array for the current dataset
            all_frame_data.append(np.zeros((numFrames, self.maxNumPeds, 3)))

            # index to maintain the current frame
            curr_frame = 0
            for frame in frameList:
                # Extract all pedestrians in current frame
                pedsInFrame = data[:, data[0, :] == frame]

                # Extract peds list
                pedsList = pedsInFrame[1, :].tolist()

                # Helper print statement to figure out the maximum number of peds in any frame in any dataset
                # if len(pedsList) > 1:
                # print len(pedsList)
                # DEBUG
                #    continue

                # Add number of peds in the current frame to the stored data
                numPeds_data[dataset_index].append(len(pedsList))

                # Initialize the row of the numpy array
                pedsWithPos = []

                # For each ped in the current frame
                for ped in pedsList:
                    # Extract their x and y positions
                    current_x = pedsInFrame[3, pedsInFrame[1, :] == ped][0]
                    current_y = pedsInFrame[2, pedsInFrame[1, :] == ped][0]

                    # Add their pedID, x, y to the row of the numpy array
                    pedsWithPos.append([ped, current_x, current_y])

                # Add the details of all the peds in the current frame to all_frame_data
                all_frame_data[dataset_index][curr_frame, 0:len(pedsList), :] = np.array(pedsWithPos)
                # Increment the frame index
                curr_frame += 1
            # Increment the dataset index
            dataset_index += 1

        # Save the tuple (all_frame_data, frameList_data, numPeds_data) in the pickle file
        f = open(data_file, "wb")
        pickle.dump((all_frame_data, frameList_data, numPeds_data), f, protocol=2)
        f.close()

    def load_preprocessed(self, data_file):
        '''
        Function to load the pre-processed data into the DataLoader object
        params:
        data_file : the path to the pickled data file
        '''
        # Load data from the pickled file
        f = open(data_file, 'rb')
        self.raw_data = pickle.load(f)
        f.close()

        # Get all the data from the pickle file
        self.data = self.raw_data[0]
        self.frameList = self.raw_data[1]
        self.numPedsList = self.raw_data[2]
        counter = 0

        # For each dataset
        for dataset in range(len(self.data)):
            # get the frame data for the current dataset
            all_frame_data = self.data[dataset]
            # Increment the counter with the number of sequences in the current dataset
            # DOUBT
            counter += int(len(all_frame_data) / (self.seq_length+2))

        # Calculate the number of batches
        self.num_batches = int(counter/self.batch_size)
        # On an average, we need twice the number of batches to cover the data
        # due to randomization introduced
        self.num_batches = self.num_batches * 2

    def next_batch(self, randomUpdate=True):
        '''
        Function to get the next batch of points
        '''
        # Source data
        source_batch = []
        # Target data
        target_batch = []
        # Dataset data
        d = []
        # Iteration index
        i = 0
        while i < self.batch_size:
            # Shape : numFrames, maxNumPeds, 3 : pedId, x, y
            curr_dataset = self.data[self.dataset_pointer]
            # Get the frame pointer for the current dataset
            idx = self.frame_pointer
            # While there is still seq_length number of frames left in the current dataset
            if idx + self.seq_length < curr_dataset.shape[0]:
                # All the data in this sequence
                seq_frame_data = curr_dataset[idx:idx+self.seq_length+1, :]
                source_seq = curr_dataset[idx:idx+self.seq_length, :]
                target_seq = curr_dataset[idx+1:idx+self.seq_length+1, :]
                # Number of unique peds in this sequence of frames
                pedID_list = np.unique(seq_frame_data[:, :, 0])
                numUniquePeds = pedID_list.shape[0]

                sourceData = np.zeros((self.seq_length, self.maxNumPeds, 3))
                targetData = np.zeros((self.seq_length, self.maxNumPeds, 3))

                # Align ped in all frames. After this a ped within the
                # seq will be at the same index in all frames
                for frame_num in range(self.seq_length):
                    curr_source_frame = source_seq[frame_num, :]
                    curr_target_frame = target_seq[frame_num, :]
                    for ped in range(numUniquePeds):
                        pedID = pedID_list[ped]
                        if pedID == 0:
                            continue
                        else:
                            curr_source_ped = curr_source_frame[curr_source_frame[:, 0] == pedID, :]
                            curr_target_ped = np.squeeze(curr_target_frame[curr_target_frame[:, 0] == pedID, :])
                            if curr_source_ped.size != 0:
                                sourceData[frame_num, ped, :] = curr_source_ped
                            if curr_target_ped.size != 0:
                                targetData[frame_num, ped, :] = curr_target_ped

                source_batch.append(sourceData)
                target_batch.append(targetData)

                # Advance the frame pointer to a random point
                if randomUpdate:
                    self.frame_pointer += random.randint(1, self.seq_length)
                else:
                    self.frame_pointer += self.seq_length

                d.append(self.dataset_pointer)
                i += 1
            else:
                # Not enough frames left
                # Increment the dataset pointer and set the frame_pointer to zero
                self.tick_batch_pointer()

        return source_batch, target_batch, d

    def tick_batch_pointer(self):
        '''
        Advance the dataset pointer
        '''
        # Go to the next dataset
        self.dataset_pointer += 1
        # Set the frame pointer to zero for the current dataset
        self.frame_pointer = 0
        # If all datasets are done, then go to the first one again
        if self.dataset_pointer >= len(self.data):
            self.dataset_pointer = 0

    def reset_batch_pointer(self):
        '''
        Reset all pointers
        '''
        # Go to the first frame of the first dataset
        self.dataset_pointer = 0
        self.frame_pointer = 0
