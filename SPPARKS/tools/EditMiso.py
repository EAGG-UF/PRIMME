#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 11:13:52 2020

@author: fhilty
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

filename = 'Miso.txt'
savename = 'Miso.txt'

def GetLine(i,j):
    return i-1+(j-2)*(j-1)/2

if sys.argv[1] == 'A':
    data = [sys.argv[3]+'\n']*GetLine(int(sys.argv[2]),int(sys.argv[2]))
elif sys.argv[1] == 'R':
    data = [0]*GetLine(int(sys.argv[2]),int(sys.argv[2]))
    for i in range(0,GetLine(int(sys.argv[2]),int(sys.argv[2]))):
        data[i] = str(int(sys.argv[3])+np.random.uniform(0,1)*(int(sys.argv[4])-int(sys.argv[3]))) + '\n'
else:    
    # Open filename
    with open(filename, 'r') as file:
        # read a list of lines into data
        data = file.readlines()
    
    if sys.argv[1] == 'E':
        if ((len(sys.argv)-3) % 2) != 0:
            print('Incorrect number of GrainID(s) and/or Misorientation(s)')
        pairs = (len(sys.argv)-3)/2
        for j in range(0,pairs):
            for i in range(0,int(sys.argv[3+j])):
                data[GetLine(i,int(sys.argv[3+j]))] = sys.argv[3+pairs+j] + '\n'
            for i in range(int(sys.argv[3+j])+1,int(sys.argv[2])):
                data[GetLine(int(sys.argv[3+j]),i)] = sys.argv[3+pairs+j] + '\n'
    elif sys.argv[1] == 'S':
        if (len(sys.argv) % 3) != 0:
            print('Incorrect number of GrainID(s) and/or Misorientation(s)')
        pairs = (len(sys.argv)-3)/3
        for i in range(0,pairs):
            Min = min(int(sys.argv[3+(2*i)]),int(sys.argv[4+(2*i)]))
            Max = max(int(sys.argv[3+(2*i)]),int(sys.argv[4+(2*i)]))
            data[GetLine(Min,Max)] = sys.argv[(3+2*pairs)+i] + '\n'
    else:
        print('Unknown mode (Options are: A R E S)')

# and write everything back
with open(savename, 'w') as file:
    file.writelines( data )