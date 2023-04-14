#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 14:07:40 2023

@author: joseph.melville
"""


import functions as fs


trainset_location = fs.create_SPPARKS_dataset_circles(size=[512,512], radius_rng=[64, 200], kt=0.66, cutoff=0.0, nsets=200, max_steps=10, offset_steps=1, future_steps=4)
