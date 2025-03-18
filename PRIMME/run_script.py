#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 4 02:05:54 2025

@author: gabriel.castejon
"""

import PRIMME as fsp
import functions as fs
import torch
import argparse
from pathlib import Path

def main(args):
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    # Define grain sizes based on provided value
    grain_sizes = [[args.grain_size, args.grain_size], args.grain_size]
    
    # Create IC shape string
    ic_shape = f"{args.grain_shape}({grain_sizes[0][0]}_{grain_sizes[0][1]}_{grain_sizes[1]})" if args.grain_shape != "hex" else "hex"
    
    # Define filename for potential saved data
    filename_test = f"{ic_shape}.pickle"
    path_load = Path('./data') / filename_test
    
    # Load or generate initial conditions and misorientation data
    if path_load.is_file():
        print(f"Loading existing initial conditions from {path_load}")
        data_dict = fs.load_picke_files(load_dir=Path('./data'), filename_save=filename_test)
        ic, ea, miso_array, miso_matrix = data_dict["ic"], data_dict["ea"], data_dict["miso_array"], data_dict["miso_matrix"]
    else:
        print(f"Generating new initial conditions with shape {ic_shape}")
        ic, ea, miso_array, miso_matrix = fs.generate_train_init(filename_test, args.grain_shape, grain_sizes, device)
    
    # Train or load PRIMME model
    if not args.primme and not args.modelname:
        print(f"Training new PRIMME model with {args.num_eps} epochs")
        modelname = fsp.train_primme(
            args.trainset, 
            n_step=args.nsteps, 
            n_samples=args.n_samples, 
            mode=args.mode, 
            num_eps=args.num_eps, 
            dims=args.dims, 
            obs_dim=args.obs_dim, 
            act_dim=args.act_dim, 
            lr=args.lr, 
            reg=args.reg, 
            pad_mode=args.pad_mode, 
            if_plot=args.if_plot
        )
    else:
        if not args.primme:
            print(f"Using existing model: {args.modelname}")
            modelname = args.modelname
        else:
            print(f"Will use existing PRIMME file, skipping model.")
    
    # Run PRIMME model
    if not args.primme:
        print(f"Running PRIMME model for {args.nsteps} steps")
        ims_id, fp = fsp.run_primme(
            ic, ea, 
            nsteps=args.nsteps, 
            modelname=modelname, 
            miso_array=miso_array, 
            miso_matrix=miso_matrix,
            pad_mode=args.pad_mode, 
            ic_shape=ic_shape
        )
    else:
        print(f"Using existing PRIMME file: {args.primme}")
        fp = args.primme
    
    # Generate plots and statistics
    print("Computing grain statistics and generating plots")
    fs.compute_grain_stats(fp)
    fs.make_videos(fp, ic_shape=ic_shape)
    fs.make_time_plots(fp, ic_shape=ic_shape, if_plot=args.if_output_plot)
    
    print(f"PRIMME simulation complete. Results saved to: {fp}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRIMME: Physics-Regularized Interpretable Machine Learning Microstructure Evolution")

    parser.add_argument("--trainset", type=str, default="./data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(200)_future(4)_max(100)_kt(0.66)_cut(0).h5", help="Training set.")

    parser.add_argument("--modelname", type=str, default=None, help="Model name.")
    parser.add_argument("--dims", type=int, default=2, help="Dimensions for training.")
    parser.add_argument("--if_plot", action="store_true", help="If plot, used for training.")
    parser.add_argument("--num_eps", type=int, default=1000, help="Number of epochs.")
    parser.add_argument("--obs_dim", type=int, default=17, help="Observation dimension.")
    parser.add_argument("--act_dim", type=int, default=17, help="Action dimension.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--reg", type=float, default=1, help="Regularization.")
    parser.add_argument("--nsteps", type=int, default=1000, help="Number of steps.")
    parser.add_argument("--n_samples", type=int, default=200, help="Number of samples.")
    parser.add_argument("--mode", type=str, default="Single_Step", help="Mode.")

    # voroni2image and miso Related Arguments

    parser.add_argument("--grain_shape", type=str, default="grain", help="Grain shape.") # Alternatives include "circle", "hex", "square"
    parser.add_argument("--grain_size", type=int, default=512, help="Grain sizes.")

    parser.add_argument("--voroni_loaded", action="store_true", help="Voroni loaded.")
    parser.add_argument("--ic", type=str, default="./data/ic.npy", help="Initial condition.")
    parser.add_argument("--ea", type=str, default="./data/ea.npy", help="Euler angles.")
    parser.add_argument("--ma", type=str, default="./data/ma.npy", help="Misorientation angles.")
    parser.add_argument("--ic_shape", type=str, default="grain(512_512_512)", help="Initial condition shape.")
    
    # if not provided, these are the default values
    parser.add_argument("--size", type=list, default=93, help="Size of the image generated by voroni2image.")
    parser.add_argument("--dimension", type=int, default=3, help="Dimension of the image generated by voroni2image.")
    parser.add_argument("--ngrain", type=int, default=2**14, help="Number of grains generated by voroni2image.")

    # PRIMME Run Related Arguments
    parser.add_argument("--primme", type=str, default=None, help="PRIMME File was provided.")

    parser.add_argument("--pad_mode", type=str, default="circular", help="Padding mode.")
    parser.add_argument("--if_output_plot", action="store_true", help="If output plot.")

    # Show plots:
    args = parser.parse_args()
    print(f"Running PRIMME with arguments: {args}")
    main(args)


# Example run: python run_script.py --modelname="./data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(0).h5" --primme="./data/primme_shape(grain(512_512_512))_model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(0).h5"

