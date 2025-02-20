from tqdm import tqdm
import os.path
import torch
import h5py
import numpy as np
from pathlib import Path
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import functions as fs
import PRIMME

paras_dict = {"num_eps": 25,
              "mode": "Single_Step",
              "dims": 2,
              "obs_dim":17,
              "act_dim":17,
              "lr": 5e-5,
              "reg": 1,
              "pad_mode": "circular",
              "if_plot": True}

trainset_dict = {"case1": [r"./data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(200)_future(4)_max(100)_kt(0.66)_cut(0).h5"]}
n_samples_dict = {"case1": [[200]]}
n_step_dict = {"case1": [[5]]}

test_case_dict = {"case6": ["grain", [[512, 512], 512]]}
for key_data in trainset_dict.keys():
    trainset = trainset_dict[key_data][0]
    for n_samples in n_samples_dict[key_data]:
        for n_step in n_step_dict[key_data]:
            modelname = PRIMME.train_primme(trainset, n_step, n_samples, test_case_dict)
nsteps = 800
for key in test_case_dict.keys():
    grain_shape, grain_sizes = test_case_dict[key]
    if grain_shape == "hex":
        ic_shape = grain_shape
    else:
        ic_shape = grain_shape + "(" + ("_").join([str(grain_sizes[0][0]), str(grain_sizes[0][1]), str(grain_sizes[1])]) + ")"
    filename_test = ic_shape + ".pickle"
    path_load = Path('/Users/gabecastejon/Coding/PRIMME_Work/PRIMME-Readable/data').joinpath(filename_test)
    if os.path.isfile(str(path_load)):
        data_dict = fs.load_picke_files(load_dir = Path('/Users/gabecastejon/Coding/PRIMME_Work/PRIMME-Readable/data'), filename_save = filename_test)
        ic, ea, miso_array, miso_matrix = data_dict["ic"], data_dict["ea"], data_dict["miso_array"], data_dict["miso_matrix"]
    else:
        ic, ea, miso_array, miso_matrix = fs.generate_train_init(filename_test, grain_shape, grain_sizes, device)
modelname = r"/Users/gabecastejon/Coding/PRIMME_Work/PRIMME-Readable/data/pred(Single_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(4)_max(100)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"
ims_id, fp_primme = PRIMME.run_primme(ic, ea, miso_array, miso_matrix, nsteps, ic_shape, modelname)

fp_primme = r"/Users/gabecastejon/Coding/PRIMME_Work/PRIMME-Readable/data/primme_shape(grain(512_512_512))"
sub_folder = "video"
for key in test_case_dict.keys():
    grain_shape, grain_sizes = test_case_dict[key]
    if grain_shape == "hex":
        ic_shape = grain_shape
    else:
        ic_shape = grain_shape + "(" + ("_").join([str(grain_sizes[0][0]), str(grain_sizes[0][1]), str(grain_sizes[1])]) + ")"
fs.make_videos(fp_primme, sub_folder, ic_shape)
