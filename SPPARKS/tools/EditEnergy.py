#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 11:13:52 2020

@author: fhilty
"""

import sys

"""
sys arguments: PythonFileName Mode NumberOfGrains GrainID(s) Energy(s)

PythonFileName = Name of python executable, EditEnergy.py in this case
Mode = The type of operation to perform, see below for options and format
NumberOfGrains = The number of grains in the system
GrainID(s) = The grainIDs of the grains to be changed, see below for details
Energy(s) = Energy value that the grain boundry should be set to

Possible Modes: A    S
    A = All, Sets all energy values to specified energy value
        Needs 1 Energy
    E = Everyother, Edits all GB energies between grain GrainID(s) and all other grains
        Needs 1 GrainID and 1 Energy for each grain to be changed
    S = Specified, Edits list of GB pairs from GrainID(s)
        Needs 2 GrainIDs per pair and 1 Energy per pair

"""

filename = 'Energy.txt'
savename = 'Energy.txt'

def GetLine(i,j):
    return i-1+(j-2)*(j-1)/2

if sys.argv[1] == 'A':
    data = [sys.argv[3]+'\n']*GetLine(int(sys.argv[2]),int(sys.argv[2]))
else:    
    # Open filename
    with open(filename, 'r') as file:
        # read a list of lines into data
        data = file.readlines()
    
    if sys.argv[1] == 'E':
        if ((len(sys.argv)-3) % 2) != 0:
            print('Incorrect number of GrainID(s) and/or Energy(s)')
        pairs = (len(sys.argv)-3)/2
        for j in range(0,pairs):
            for i in range(0,int(sys.argv[3+j])):
                data[GetLine(i,int(sys.argv[3+j]))] = sys.argv[3+pairs+j] + '\n'
            for i in range(int(sys.argv[3+j])+1,int(sys.argv[2])):
                data[GetLine(int(sys.argv[3+j]),i)] = sys.argv[3+pairs+j] + '\n'
    elif sys.argv[1] == 'S':
        if (len(sys.argv) % 3) != 0:
            print('Incorrect number of GrainID(s) and/or Energy(s)')
        pairs = (len(sys.argv)-3)/3
        for i in range(0,pairs):
            Min = min(int(sys.argv[3+(2*i)]),int(sys.argv[4+(2*i)]))
            Max = max(int(sys.argv[3+(2*i)]),int(sys.argv[4+(2*i)]))
            data[GetLine(Min,Max)] = sys.argv[(3+2*pairs)+i] + '\n'
    else:
        print('Unknown mode (Options are: A E S)')

# and write everything back
with open(savename, 'w') as file:
    file.writelines( data )
