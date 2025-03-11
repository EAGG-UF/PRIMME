# Physics-Regulated Interpretable Machine Learning Microstructure Evolution (PRIMME)

## Description:

Physics-Regularized Interpretable Machine Learning Microstructure Evolution (PRIMME): This code can be used to train and validate PRIMME neural network models for simulating isotropic microstructural grain growth.
	
## Contributors: 

Weishi Yan (1), Joel Harley (1), Joseph Melville (1), Kristien Everett (1), Lin Yang (2)

## Folder / File Descriptions:

### Top level folders:

PRIMME - 	Actual PRIMME code

### "PRIMME" folder:

functions - 	All of the functions used to create initial conditions, run SPPARKS and PRIMME, and calculate statistics

PRIMME - 	A class that contains the PRIMME model and some helper functions

run_script - 	References 'functions' to run and evaluate SPPARKS and PRIMME simulations, with parameters specified through program arguments.

test_run -	Test run file for user to manually run PRIMME with desired parameters, outside of the ones provided by 'run_script'.

run.ipynb -	Copy of test_run made for Jupyter Notebook and Google colab.

gui_appplication -	Provides a Graphic User Interface (GUI) for the user to run and train PRIMME, leverages 'run_script'.

plots -		Output plots from run_script are stored here, PRIMME will automatically make this folder if it does not exist.

data -		Output models from run_script are stored here, PRIMME will automatically make this folder if it does not exist.

#### "functions" file (sections):

Script - Set up folders and GPU

General - 			File management functions

Create initial conditions - 	See "voronoi2image" first

Run and read SPPARKS - 		See "run_spparks" first **NOT USED FOR THIS VERSION OF PRIMME*

Find misorientations - 		See "find_misorientation" first

Statistical functions - 	See "compute_grain_stats" first 

Run PRIMME - 			See "run_primme" first 

### Other notes:

- GPU Usage (GPU 0,  MPS (for MAC), or CPU if neither is available) is hard coded in two places, at the beginning of both the "PRIMME" and "functions" files.

- This model is often trained of SPPARKS data, see its [GitHub](https://github.com/spparks/spparks) and [Documentation](https://spparks.github.io/) for more information.

## Usage

There are two ways to run the program:

### Google Colab

See the following Colab link to run PRIMME remotely 

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gabo0802/PRIMME-Readable/blob/main/PRIMME/run.ipynb)

### Local GUI

Use the following command to install pre-requirement packages (ideally in a [virtual environment](https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/))
```bash
pip install -r requirements.txt
```

Run the GUI Application for Training and Running PRIMME
```python
python gui_application.py
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

