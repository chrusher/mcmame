#! /usr/bin/env python

import datetime
import logging
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from astropy import table
from astropy.io import fits

import mcmame_lib
import mcmame

# import matplotlib
# matplotlib.rcParams['figure.dpi'] = 50

LOG_FORMAT = "[%(asctime)s] %(levelname)8s %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

plot = False


metal = -0.6822533361523742
metal_e = 0.1595040018637024
mags = [['u', -6.6404734, 0.029137253712118956],
        ['g', -8.340472, 0.01],
        ['r', -9.140472, 0.01],
        ['i', -9.5604725, 0.01],
        ['z', -9.840472, 0.01]]
A_V = 3.1 * 0.08
A_V_e = 3.1 * 0.03

with open(os.path.expanduser('~') + '/sluggs/sps_models/fsps_mist_inter_mags.pickle', 'rb') as f:
    grids = pickle.load(f)

with open(os.path.expanduser('~') + '/sluggs/sps_models/fsps_reddening_mist_inter_mags.pickle', 'rb') as f:
    reddening_grids = pickle.load(f)    

age = 12.6
metal = -1
metal_e = 0.2
mass = 5
A_V = 0.3
A_V_e = 0.03
A_V2 = 0.5
A_V2_e = 0.25

print('True values', metal, age, mass, A_V2)

mags = []

for band in ['u', 'g', 'r', 'i', 'z']:

    mag = grids[band].ev(metal, age) - 2.5 * mass + A_V2 * reddening_grids[band].ev(metal, age)
    mags.append([band, mag, 0.02])

start = datetime.datetime.now()
mcmame_lib.calc_age_mass(mags, None, None, None, None, plot=plot, threads=1, nwalkers=100, steps=500, nburn=200)
print(datetime.datetime.now() - start)
start = datetime.datetime.now()
mcmame.calc_age_mass(mags, None, None, None, None, grids, reddening_grids, plot=plot, threads=1, nwalkers=100, steps=500, nburn=200)
print(datetime.datetime.now() - start)
#     calc_age_mass(mags, metal, metal_e, A_V2, A_V2_e, plot=plot, threads=1, nwalkers=100, steps=200, nburn=200)    
#     calc_age_mass(mags, metal, metal_e, A_V, A_V_e, A_V2=A_V2, A_V2_e=A_V2_e, plot=plot, threads=1, nwalkers=100, steps=200, nburn=200)    

print()

if plot:
    plt.show()