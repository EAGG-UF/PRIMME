=============================================================================================
Physics-Regulated Interpretable Machine Learning Microstructure Evolution (PRIMME)
=============================================================================================

DESCRIPTION:
	Physics-Regularized Interpretable Machine Learning Microstructure Evolution (PRIMME)
	This code can be used to train and validate PRIMME neural network models for simulating isotropic microstructural grain growth
	
CONTRIBUTORS: 
	Weishi Yan (1), Joel Harley (1), Joseph Melville (1), Kristien Everett (1), Lin Yang (2)

AFFILIATIONS:
	1. University of Florida, SmartDATA Lab, Department of Electrical and Computer Engineering
	2. University of Florida, Tonks Research Group, Department of Material Science and Engineering

FUNDING SPONSORS:
	U.S. Department of Energy, Office of Science, Basic Energy Sciences under Award \#DE-SC0020384
	U.S. Department of Defence through a Science, Mathematics, and Research for Transformation (SMART) scholarship

REQUIRMENTS:
	numpy
	scipy
	keras
	tensorflow
	torch
	tqdm
	h5py
	unfoldNd
	pynvml
	matplotlib
	imageio

FOLDER/FILE DESCRIPTIONS:

Top level folders:
SPPARKS - 	Reference files to run SPPARKS simulations
PRIMME - 	Actual PRIMME code

"PRIMME" folder:
cfg - 		Keras reference files
spparks_files -	See "Getting_Started.txt" for help getting SPPARKS functioning on the lambda server
functions - 	All of the functions used to create initial conditions, run SPPARKS and PRIMME, and calculate statistics
PRIMME - 	A class that contains the PRIMME model and some helper functions
run - 		References 'functions' to run and evaluate SPPARKS and PRIMME simulations

"functions" file (sections):
Script - Set up folders and GPU
General - 			File management functions
Create initial conditions - 	See "voronoi2image" first
Run and read SPPARKS - 		See "run_spparks" first
Find misorientations - 		See "find_misorientation" first
Statistical functions - 	See "compute_grain_stats" first 
Run PRIMME - 			See "run_primme" first 

Other notes:
-The use of GPU 0 (or CPU is GPU 0 is not available) is hard coded in two places, at the beginning of both the "PRIMME" and "functions" files
-The output of 'run.py' is the images of a circle grain PRIMME simulation.