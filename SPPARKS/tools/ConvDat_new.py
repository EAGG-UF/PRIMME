#!/Users/lin.yang/miniconda3/bin/python python
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 16:18:52 2020

@author: Lin.Yang
"""

import sys
import numpy as np

"""
sys arguments: PythonFileName

PythonFileName = Name of python executable, ConvDat.py in this case

"""
filename = 'cluster.dat'
savename = 'potts.csv'

# keep the data we will write
head = 'Time' # head line in csv file
data = []     # data body in csv file
grainlist = []# all the grain we have in model
timestep = 0  # tiemstep

# Open filename
with open(filename, 'r') as file:
    line = file.readline()

    # loop of each line
    while line:
        eachline = line.split()

        # jump over the empty line
        if eachline == []:
            line = file.readline()
            continue

        # set the first data 'Time' in each csv_line
        if eachline[0] == 'Time':
            nowtime = eachline[2]
            line = file.readline()
            continue

        # jump over all line without 7 data
        if len(eachline) != 7:
            line = file.readline()
            continue
        # when meet 'id ...', it will
        # 1.add csv_line of last timestep to data
        # 2.creat empty need_datafor new timestep
        elif eachline[0] == 'id':
            if timestep > 0:
                str_data = ' '.join(need_data)
                data.append(str_data+'\n')

            timestep += 1
            need_data = ['0' for index in range(len(grainlist)+1)]
            need_data[0] = nowtime
            line = file.readline()
            continue
        # for each grain type
        elif eachline[1] in grainlist:
            j = grainlist.index(eachline[1])+1
            need_data[j] = str(int(need_data[j]) + int(eachline[3]))
        else:
            grainlist.append(eachline[1])
            head = head + ' gs' + eachline[1]
            need_data.append(eachline[3])


        line = file.readline()

# add the last row of timestep data
str_data = ' '.join(need_data)
data.append(str_data+'\n')

with open(savename, 'w') as file:
    # str_head = ' '.join(head)
    file.writelines( head+'\n' )
    file.writelines( data )
