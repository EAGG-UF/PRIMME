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

FOLDERS:
	cfg - PRIMME network architechture parameters (do not need to change)
	data_validation - grain centers for validation 
	results_training - plots and data from the "train_model_script" (created by script)
	results_validation - plots and data from the "validation_model_script" (created by script)
	saved_models - trained PRIMME models
	spparks_files - files used to run SPPARKS simulations
	spparks_simulations - SPPARKS simulation data and supporting files (created by script)

SCRIPTS:
	train_model_script - trains a PRIMME model
	PRIMME - contains PRIMME object class for training and running PRIMME simulations
	SPPARKS - contains SPPARKS object class for running SPPARKS simulations
	functions - referenced by both PRIMME and SPPARKS classes
	validate_model_script - used to validate a specific PRIMME model
