import csv
import argparse
import os

def main():
	parser = argparse.ArgumentParser()

	# Input folder path
	parser.add_argument('--input_data_folder', type=str,
		help='input data folder with generated traj files')

	parser.add_argument('--output_data_folder', type=str,
		help='output data folder with converted traj data')

	args = parser.parse_args()
	convert_files(args)

def convert_files(args):
	input_folder = args.input_data_folder
	output_folder = args.output_data_folder

	print input_folder
	print output_folder
	
	for filename in os.listdir(input_folder):
		file_path = os.path.join(input_folder, filename)
		output_file_path = os.path.join(output_folder, filename)
		
		with open(output_file_path, 'wb') as outfile:
			writerCSV = csv.writer(outfile, quoting=csv.QUOTE_NONE)

			with open(file_path) as traj_data:
				readCSV = csv.reader(traj_data, delimiter=',')
				print readCSV
				frames, pedIds, y_cords, x_cords = [], [], [], []
				for row in readCSV:
					frames.append(row[3])
					pedIds.append(row[0])
					y_cords.append(row[2])
					x_cords.append(row[1])

			writerCSV.writerow(frames)
			writerCSV.writerow(pedIds)
			writerCSV.writerow(y_cords)
			writerCSV.writerow(x_cords)

if __name__ == '__main__':
    main()


