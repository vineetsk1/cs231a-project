from os import listdir
from os.path import isfile, join
import numpy as np
import csv

dirs = ['./1/train/biwi/', './1/train/crowds/', './1/train/mot/', './1/train/stanford/']
#csvfile = open('xx', 'wb')
files = [f for i in range(3) for f in listdir(dirs[i])if isfile(join(dirs[i], f))]
for i in range(len(dirs)):
    for f in listdir(dirs[i]):
        if not f.endswith(".txt"):
            continue
        print f
        if isfile(join(dirs[i], f)) and f[-3:] != 'csv':
            counter = 0
            # f is now the filename ending in .txt
            fil = open(dirs[i] + f, 'r')
            rows = [[], [], [], []]
            for line in fil:
                vals = map(float, line.split())
                a = [(k >= vals[0]) * 1 for k in rows[0]]
                if 1 not in a:
                    for j in range(2):
                        rows[j].append(vals[j])
                    rows[2].append(vals[3])
                    rows[3].append(vals[2])
                else:
                    ind = a.index(1)
                    if vals[0] != rows[0][ind]:
                        for j in range(2):
                            rows[j].insert(ind, vals[j])
                        rows[2].insert(ind, vals[3])
                        rows[3].insert(ind, vals[2])
                    else:
                        row1copy = []
                        for l in range(ind, len(rows[1])):
                            if vals[0] == rows[0][l]:
                                row1copy.append((rows[1][l] > vals[1]) * 1)
                            else:
                                break
                        indm = ind + len(row1copy)
                        if 1 in row1copy:
                            indm = ind + row1copy.index(1)
                        for j in range(2):
                            rows[j].insert(indm, vals[j])
                        rows[2].insert(indm, vals[3])
                        rows[3].insert(indm, vals[2])
            fil.close()
            csvfile = open(dirs[i] + f[:-3] + 'csv', 'wb')
            writer = csv.writer(csvfile)
            writer.writerows(rows)
            csvfile.close()
    
            
            
