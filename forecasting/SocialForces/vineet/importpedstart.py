
# Import data from text file

# Script for importing data from a text file
# To extend the code to different selected data or
# a different text file, generate a function instead
# of a script

def importpedstart(filename):
	dataArray = np.loadtxt(filename, delimiter=",", skiprows=1)
	return dataArray