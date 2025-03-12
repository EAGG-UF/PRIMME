# Physics-Regulated Interpretable Machine Learning Microstructure Evolution (PRIMME)

## Description:

Physics-Regularized Interpretable Machine Learning Microstructure Evolution (PRIMME): This code can be used to train and validate PRIMME neural network models for simulating isotropic microstructural grain growth.
	
## Contributors: 

Weishi Yan (1), Joel Harley (1), Joseph Melville (1), Kristien Everett (1), Lin Yang (2)

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

## Notes:

- GPU Usage (GPU 0,  MPS (for MAC), or CPU if neither is available) is hard coded in two places, at the beginning of both the "PRIMME" and "functions" files.

- This model is often trained of SPPARKS data, see its [GitHub](https://github.com/spparks/spparks) and [Documentation](https://spparks.github.io/) for more information.

## Affiliation:

1. University of Florida, SmartDATA Lab, Department of Electrical and Computer Engineering
2. University of Florida, Tonks Research Group, Department of Materials Science and Engineering

## Funding Sponsors:

U.S. Department of Energy, Office of Science, Basic Energy Sciences under Award \#DE-SC0020384
U.S. Department of Defence through a Science, Mathematics, and Research for Transformation (SMART) scholarship


