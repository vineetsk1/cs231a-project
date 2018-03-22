# CS231A Project

CS231A Project about Trajectory Prediction.

## Data

Data is stored inside the `data` directory. Within this directory are the datasets for each of the trajectory forecasting "challenges". Challenge 1 is in 3D world coordinates (world human-human), challenge 2 is in 2D image coordinates (image human-human), and challenge 3 is in 2D image coordinates (image human-human-space). For the sakes of our project, challenges 2 and 3 are identical. Normally, challenge 3 also includes a reference image frame for each entry in the dataset but none of the trajectory prediction techniques we implement will use that.

Within each challenge, the data is split into train and test. The ground truth is also provided in the folder gt. Within each of these folders are subfolders, containing data from various sources. For example, in train the "stanford" folder refers to data from the stanford drone dataset. Each file inside these folders then contains trajectories from one video (eg: `data/challenges/1/train/stanford/coupa_3.txt` refers to the third coupa video from the stanford drone dataset, included as the training set for the first challenge).

Within each file is a list of trajectories. Each trajectory is 20 lines long. Each line's format is four numbers, seperated by spaces. The first number is the frameid (each trajectory consists of 20 frames). The second number is the person id (each trajectory corresponds to one person, and this id tracks them). The next two numbers are the x and the y positions. In train and gt all these values are filled in.

In test, these values are given in full for the first 8 frames of every trajectory. Our task is then to predict the (x,y) for the next 12 frames of the trajectory. 

Note that this is an initial version of the data and we might expand it in the future. Specifically, the dataset currently was chosen as follows:

- we collated 30 different common datasets used in trajectory prediction

- from those, found ones that fit common requirements we were looking for

- from those, found the ones that we could calculate 2D and 3D coordinates for (they provided the homography matrix)

- from there, we processed the datasets to split them into trajectories that are 20 frames long

- and we also processed by removing overly-linear entries in the dataset

It's possible we could make the dataset larger in the future by adding more common datasets used in the field, or use datasets that don't provide the homography matrix but can be applied in either 2D or 3D (not necessarily both), or preprocess the datasets in different ways.

## Source Code

The code for the `lstm` models is within `lstm`. The code for the naive lstm is in the `naive-lstm` folder. Inside of here, I believe all the code that will need to be changed is within `lstm`. Similarily, to work on the `naive-occupancy-lstm` (for the final project), which is the O-LSTM referenced in the "Social Forces CVGL" paper, that's within `naive-occupancy-lstm/lstm` I think. For the actual social LSTM code, that's either inside of the `social_lstm` folder of those folders, or within the `social-lstm` seperate directory, I will double check this.

The code for the social forces and for interaction GP is within the `forecasting` folder.

## Visualization

Check out the visualization at [vineetsk1.github.io/cs231a-project-results/](http://vineetsk1.github.io/cs231a-project-results/). The code for the visualization is in website.

## License

Please do not share code and/or data outside of the Stanford Vision and Learning Group (previously the Computational Vision and Geometry Lab), or outside of the course staff for CS231A for licensing reasons.