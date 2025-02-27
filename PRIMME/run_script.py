#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:48:30 2025

@author: gabriel.castejon
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 3 14:10:01 2025

@author: gabriel.castejon
"""

import PRIMME as fsp
import functions as fs
import h5py 
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle
from tqdm import tqdm
import argparse
from pathlib import Path
import sys

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = "cpu"

'''

How to run the program:

1. Ensure that plots and data directories are created in this directory.
2. Run the following command in the terminal:
    python run_primme.py

'''
def main(args):

    if args.input_args_enabled:
        trainset = args.trainset
        num_eps = args.num_eps
        if not args.modelname:
            modelname = fsp.train_primme(trainset, num_eps=num_eps, obs_dim=args.obs_dim, act_dim=args.act_dim, lr=args.lr, reg=args.reg, if_miso=args.if_miso, plot_freq=args.plot_freq)
        else:
            modelname = args.modelname

        if not args.voroni_loaded:
            ic, ea, _ = fs.voronoi2image(size=[args.size, args.size, args.size], ngrain=args.ngrain)
            ma = fs.find_misorientation(ea, mem_max=1) 
            np.save("./data/ic.npy", ic), np.save("./data/ea.npy", ea), np.save("./data/ma.npy", ma)
        else:
            ic, ea, ma = np.load(args.ic), np.load(args.ea), np.load(args.ma)

        fp = fsp.run_primme(ic, ea, nsteps=args.nsteps, modelname=modelname, miso_array=ma, pad_mode=args.pad_mode, if_miso=args.if_miso, plot_freq=args.plot_freq)

        fs.compute_grain_stats(fp)
        fs.make_videos(fp)
        fs.make_time_plots(fp)
    else:
        ## Choose initial conditions
        ic, ea = fs.generate_circleIC(size=[257,257], r=64) #nsteps=200, pad_mode='circular'
        # ic, ea = fs.generate_circleIC(size=[512,512], r=200) #nsteps=200, pad_mode='circular'
        # ic, ea = fs.generate_3grainIC(size=[512,512], h=350) #nsteps=300, pad_mode=['reflect', 'circular']
        # ic, ea = fs.generate_hexIC() #nsteps=500, pad_mode='circular'
        # ic, ea = fs.generate_SquareIC(size=[512,512], r=64) 
        # ic, ea, _ = fs.voronoi2image(size=[512, 512], ngrain=512) #nsteps=500, pad_mode='circular'
        # ic, ea, _ = fs.voronoi2image(size=[1024, 1024], ngrain=2**12) #nsteps=1000, pad_mode='circular'
        # ic, ea, _ = fs.voronoi2image(size=[2048, 2048], ngrain=2**14) #nsteps=1500, pad_mode='circular'
        # ic, ea, _ = fs.voronoi2image(size=[2400, 2400], ngrain=24000) #nsteps=1500, pad_mode='circular'


        trainset = "./data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(200)_future(4)_max(100)_kt(0.66)_cut(0).h5"
        ## Run PRIMME model
        model_location = "./data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(0).h5"
        ic_shape = "grain(512-512)"

        nsteps = 800
        test_case_dict = {"case6": ["grain", [[512, 512], 512]]}
        for key in test_case_dict.keys():
            grain_shape, grain_sizes = test_case_dict[key]
            if grain_shape == "hex":
                ic_shape = grain_shape
            else:
                ic_shape = grain_shape + "(" + ("_").join([str(grain_sizes[0][0]), str(grain_sizes[0][1]), str(grain_sizes[1])]) + ")"
            filename_test = ic_shape + ".pickle"
            path_load = Path('./data').joinpath(filename_test)
            if os.path.isfile(str(path_load)):
                data_dict = fs.load_picke_files(load_dir = Path('./data'), filename_save = filename_test)
                ic, ea, miso_array, miso_matrix = data_dict["ic"], data_dict["ea"], data_dict["miso_array"], data_dict["miso_matrix"]
            else:
                ic, ea, miso_array, miso_matrix = fs.generate_train_init(filename_test, grain_shape, grain_sizes, device)

        # def train_primme(trainset, n_step, n_samples, test_case_dict, mode = "Single_Step", num_eps=25, dims=2, obs_dim=17, act_dim=17, lr=5e-5, reg=1, pad_mode="circular", if_plot=False):
        # model_location = PRIMME.train_primme(trainset, n_step=1000, n_samples=200, test_case_dict=test_case_dict, mode="Single_Step", num_eps=100, dims=2, obs_dim=17, act_dim=17, lr=5e-5, reg=1, pad_mode="circular", if_plot=False)

        # def run_primme(ic, ea, miso_array, miso_matrix, nsteps, ic_shape, modelname, pad_mode='circular',  mode = "Single_Step", if_plot=False):
        ims_id, fp_primme = fsp.run_primme(ic, ea, miso_array, miso_matrix, nsteps=1800, ic_shape=ic_shape, modelname=model_location, pad_mode='circular', if_plot=True)
        # run_primme(ic, ea, miso_array, miso_matrix, nsteps, ic_shape, modelname, pad_mode='circular',  mode = "Single_Step", if_plot=False):
        fs.compute_grain_stats(fp_primme)
        fs.make_videos(fp_primme) #saves to 'plots'
        fs.make_time_plots(fp_primme) #saves to 'plots'
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRIMME Program Arguments.")

    # Instructions:
    print("Physics-Regularized Interpretable Machine Learning Microstructure Evolution (PRIMME) This code can be used to" 
          "train and validate PRIMME neural network models for simulating isotropic microstructural grain growth.")
    print("Before running this program, please ensure that the following directories are created"
          "in this directory: 'plots' and 'data'. Also ensure that there is a trainset provided in the 'data' directory.")
    
    print("This program can be run with program arguments to skip the set-up process, see documentation for available args.")
    # Input Arguments
    parser.add_argument("--input_args_enabled", type=bool, default="False", help="Input arguments enabled.")

    # Check if "--input_args_enabled" is provided and its value
    input_args_enabled = False
    if "--input_args_enabled" in sys.argv:
        index = sys.argv.index("--input_args_enabled") + 1
        if index < len(sys.argv) and sys.argv[index].lower() in ["true", "1", "yes"]:
            input_args_enabled = True
    if input_args_enabled:        
        # SPPARKS Related Arguments
        parser.add_argument("--trainset", type=str, default='./data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(200)_future(4)_max(100)_kt(0.66)_cut(0).h5', help="Trainset provided.")
        # TODO: Include SPPARKS Running Instructions, we can see if venv solves the issue of spparks as the concern was it requiring a different python version as Spyder
        # PRIMME Model Related Arguments
        parser.add_argument("--modelname", action="store_true", help="Model provided.")
        # if not provided, these are the default values
        parser.add_argument("--num_eps", type=int, default=200, help="Number of epochs.")
        parser.add_argument("--obs_dim", type=int, default=17, help="Observation dimension.")
        parser.add_argument("--act_dim", type=int, default=17, help="Action dimension.")
        parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
        parser.add_argument("--reg", type=int, default=1, help="Regularization.")
        parser.add_argument("--if_miso", action="store_true", help="If miso, used for both PRIMME model and run.")
        parser.add_argument("--plot_freq", type=int, default=1, help="Plot frequency, , used for both PRIMME model and run.")

        # voroni2image and miso Related Arguments
        parser.add_argument("--voroni_loaded", action="store_true", help="Voroni loaded.")
        parser.add_argument("--ic", type=str, default="./data/ic.npy", help="Initial condition.")
        parser.add_argument("--ea", type=str, default="./data/ea.npy", help="Euler angles.")
        parser.add_argument("--ma", type=str, default="./data/ma.npy", help="Misorientation angles.")
        # if not provided, these are the default values
        parser.add_argument("--size", type=list, default=93, help="Size of the image generated by voroni2image.")
        parser.add_argument("--dimension", type=int, default=3, help="Dimension of the image generated by voroni2image.")
        parser.add_argument("--ngrain", type=int, default=2**14, help="Number of grains generated by voroni2image.")

        # PRIMME Run Related Arguments
        parser.add_argument("--primme", type=str, default="./data/primme_sz(93x93x93)_ng(16384)_nsteps(1000)_freq(1)_kt(0.66)_freq(0.1)_cut(0).h5", help="PRIMME File was provided.")
        # if not provided, these are the default values
        parser.add_argument("--nsteps", type=int, default=200, help="Number of steps.")
        parser.add_argument("--pad_mode", type=str, default="circular", help="Padding mode.")
    args = parser.parse_args()
    main(args)


