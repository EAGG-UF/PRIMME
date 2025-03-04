# Physics-Regulated Interpretable Machine Learning Microstructure Evolution (PRIMME)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gabo0802/PRIMME-Readable/blob/main/PRIMME/run.ipynb)

## Description:

Physics-Regularized Interpretable Machine Learning Microstructure Evolution (PRIMME)
This code can be used to train and validate PRIMME neural network models for simulating isotropic microstructural grain growth
	
## Contributors: 

Weishi Yan (1), Joel Harley (1), Joseph Melville (1), Kristien Everett (1), Lin Yang (2)

## Folder / File Descriptions:

### Top level folders:

SPPARKS - 	Reference files to run SPPARKS simulations

PRIMME - 	Actual PRIMME code

### "PRIMME" folder:

cfg - 		Keras reference files

spparks_files -	See "Getting_Started.txt" for help getting SPPARKS functioning on the lambda server

functions - 	All of the functions used to create initial conditions, run SPPARKS and PRIMME, and calculate statistics

PRIMME - 	A class that contains the PRIMME model and some helper functions

run - 		References 'functions' to run and evaluate SPPARKS and PRIMME simulations

#### "functions" file (sections):

Script - Set up folders and GPU

General - 			File management functions

Create initial conditions - 	See "voronoi2image" first

Run and read SPPARKS - 		See "run_spparks" first

Find misorientations - 		See "find_misorientation" first

Statistical functions - 	See "compute_grain_stats" first 

Run PRIMME - 			See "run_primme" first 

### Other notes:
-The use of GPU 0 (or CPU is GPU 0 is not available) is hard coded in two places, at the beginning of both the "PRIMME" and "functions" files
-The output of 'run.py' is the images of a circle grain PRIMME simulation.


## Usage
Use the following command to install pre-requirement packages
```bash
pip install -r requirements.txt
```

Running the PRIMME
```python
python run.py
```

## Demo
### Isotropic Case
<div style="display: flex; justify-content: center; align-items: center;">
  <img src="materials/mf.gif" width="260" />&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="materials/mcp.gif" width="260" />&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="materials/phase_field.gif" width="260" />
</div>
<p align="middle">
    <em >Training on mode filter(left), Training on MCP(mid) and Training on phase field (right).</em>
</p>
<be>

### Anisotropic Case
<div style="display: flex; justify-content: center; align-items: center;">
  <img src="materials/mf_incl.gif" width="390" />&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="materials/mcp_incl.gif" width="390" />
</div>
<p align="middle">
    <em >Training on mode filter(left) and Training on phase field (right).</em>
</p>
<be>

## Affiliation:

1. University of Florida, SmartDATA Lab, Department of Electrical and Computer Engineering
2. University of Florida, Tonks Research Group, Department of Materials Science and Engineering

## Funding Sponsors:

U.S. Department of Energy, Office of Science, Basic Energy Sciences under Award \#DE-SC0020384
U.S. Department of Defence through a Science, Mathematics, and Research for Transformation (SMART) scholarship

