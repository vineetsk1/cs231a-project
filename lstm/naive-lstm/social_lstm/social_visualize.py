'''
Script to help visualize the results of the trained model

Author : Anirudh Vemula
Date : 10th November 2016
'''

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os

CHK_DIR = '/cvgl2/u/junweiy/Jackrabbot/checkpoints/naive-lstm/naive/09_06_19_47'


def plot_trajectories(true_trajs, pred_trajs, obs_length, name):
    '''
    Function that plots the true trajectories and the
    trajectories predicted by the model alongside
    params:
    true_trajs : numpy matrix with points of the true trajectories
    pred_trajs : numpy matrix with points of the predicted trajectories
    Both parameters are of shape traj_length x maxNumPeds x 3
    obs_length : Length of observed trajectory
    name: Name of the plot
    '''
    traj_length, maxNumPeds, _ = true_trajs.shape

    # Initialize figure
    plt.figure()

    # Load the background
    # im = plt.imread('plot/background.png')
    # implot = plt.imshow(im)
    width = 1
    height = 1

    traj_data = {}
    # For each frame/each point in all trajectories
    for i in range(traj_length):
        pred_pos = pred_trajs[i, :]
        true_pos = true_trajs[i, :]

        # For each pedestrian
        for j in range(maxNumPeds):
            if true_pos[j, 0] == 0:
                # Not a ped
                continue
            elif pred_pos[j, 0] == 0:
                # Not a ped
                continue
            else:
                # If he is a ped
                # if true_pos[j, 1] > 1 or true_pos[j, 1] < 0:
                #     continue
                # elif true_pos[j, 2] > 1 or true_pos[j, 2] < 0:
                #     continue

                if (j not in traj_data) and i < obs_length:
                    traj_data[j] = [[], []]

                if j in traj_data:
                    traj_data[j][0].append(true_pos[j, 1:3])
                    traj_data[j][1].append(pred_pos[j, 1:3])

    plt.axis([-30, 30, -30, 30])  
    for j in traj_data:
        c = np.random.rand(3, )
        true_traj_ped = traj_data[j][0]  # List of [x,y] elements
        pred_traj_ped = traj_data[j][1]

        true_x = [p[0]*height for p in true_traj_ped]
        true_y = [p[1]*width for p in true_traj_ped]
        pred_x = [p[0]*height for p in pred_traj_ped]
        pred_y = [p[1]*width for p in pred_traj_ped]
        print true_x
        s = [2 for n in range(16)]
        s2 = [4 for n in range(16)]
        plt.scatter(true_x, true_y, color=c, marker='o',  s=s )
        plt.scatter(pred_x, pred_y, color=c, marker='x', s=s2)
        # plt.show()

    # plt.ylim((0, 1))
    # plt.xlim((0, 1))
    if traj_data:
        plt.show()
        plt.savefig('plot/'+name+'.svg')
        plt.gcf().clear()
        plt.close()


def main():
    '''
    Main function
    '''
    f = open(os.path.join(CHK_DIR, 'social_results.pkl'), 'rb')
    results = pickle.load(f)

    for i in range(len(results)):
        print i
        name = 'sequence' + str(i)
        plot_trajectories(results[i][0], results[i][1], results[i][2], name)



if __name__ == '__main__':
    main()