#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 09:56:02 2023

@author: joseph.melville
"""



import functions as fs



trainset = './data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(200)_future(4)_max(100)_kt(0.66)_cut(0).h5'
fs.train_primme(trainset, 10, batch_size=3)



    