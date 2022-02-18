#!/Users/lin.yang/miniconda3/bin/python python
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 16:18:52 2020

@author: Lin.Yang
"""

import sys
import numpy as np

"""
sys arguments: PythonFileName Mode NumberOfGrains GrainID(s) Misorientation(s)

PythonFileName = Name of python executable, EditMiso.py in this case
Mode = The type of operation to perform, see below for options and format
NumberOfGrains = The number of grains in the system
GrainID(s) = The grainIDs of the grains to be changed, see below for details
Misorientation(s) = Misorientation that the grain boundry should be set to

Possible Modes: A R E S
    A = All, Sets all misorientations to specified energy value
        Needs 1 misorientation
    R = Random, Sets all misorientations to random values between two energies
        Needs 2 misorientations
    E = Everyother, Edits all GB misorientations between grain GrainID(s) and all other grains
        Needs 1 GrainID and 1 misorientation for each grain to be changed
    S = Specified, Edits list of GB pairs from GrainID(s)
        Needs 2 GrainIDs per pair and 1 misorientation per pair

"""
filename = 'cluster.dat'
savename = 'potts.csv'
graintype = '201'

# keep the data we will write
data = ['id ivalue dvalue size cx cy cz\n']
# Open filename
with open(filename, 'r') as file:
    line = file.readline()
    while line:
        eachline = line.split()
        # print(len(eachline))
        if len(eachline)<2:
            line = file.readline()
            continue
        if eachline[1]==graintype:
            data.append(line)
        line = file.readline()

with open(savename, 'w') as file:
    file.writelines( data )
