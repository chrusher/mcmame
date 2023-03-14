#! /usr/bin/env python

import datetime
import logging
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats

from astropy import table
from astropy.io import fits

import mcmame_lib

mpl.rcParams['figure.dpi'] = 50



LOG_FORMAT = "[%(asctime)s] %(levelname)8s %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

plot = True

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
mcmame_lib.calc_age_mass(mags, None, None, None, None, plot=plot)
print('Runtime: {} s\n'.format((datetime.datetime.now() - start).total_seconds()))
# start = datetime.datetime.now()
# mcmame_lib.calc_age_mass(mags, None, None, None, None, plot=plot, nwalkers=16, steps=10000, nburn=200)
# print('Runtime: {} s\n'.format((datetime.datetime.now() - start).total_seconds()))    
# start = datetime.datetime.now()
# mcmame_lib.calc_age_mass(mags, None, None, None, None, plot=plot, nwalkers=64, steps=50000, nburn=2000, thin=200, sampler='em')
# print('Runtime: {} s\n'.format((datetime.datetime.now() - start).total_seconds()))  
# start = datetime.datetime.now()
# mcmame_lib.calc_age_mass(mags, None, None, None, None, plot=plot, nwalkers=32, steps=50 * 2000, nburn=1000, sampler='em')
# print('Runtime: {} s\n'.format((datetime.datetime.now() - start).total_seconds())) 

# start = datetime.datetime.now()
# mcmame_lib.calc_age_mass(mags, None, None, None, None, plot=plot, nwalkers=16, steps=100, nburn=200)
# print('Runtime: {} s\n'.format((datetime.datetime.now() - start).total_seconds()))
# start = datetime.datetime.now()
# mcmame_lib.calc_age_mass(mags, None, None, None, None, plot=plot, nwalkers=16, steps=1000, nburn=200)
# print('Runtime: {} s\n'.format((datetime.datetime.now() - start).total_seconds()))
# start = datetime.datetime.now()
# mcmame_lib.calc_age_mass(mags, None, None, None, None, plot=plot, nwalkers=16, steps=10000, nburn=200)
# print('Runtime: {} s\n'.format((datetime.datetime.now() - start).total_seconds()))
# start = datetime.datetime.now()
# mcmame_lib.calc_age_mass(mags, None, None, None, None, plot=plot, nwalkers=16, steps=100000, nburn=200)
# print('Runtime: {} s\n'.format((datetime.datetime.now() - start).total_seconds()))

# start = datetime.datetime.now()
# mcmame_lib.calc_age_mass(mags, None, None, None, None, plot=plot, threads=1, nwalkers=100, steps=500, nburn=200)
# print(datetime.datetime.now() - start)
# start = datetime.datetime.now()
# mcmame_lib.calc_age_mass(mags, None, None, None, None, plot=plot, threads=1, nwalkers=200, steps=500, nburn=200)
# print(datetime.datetime.now() - start)
# start = datetime.datetime.now()
# mcmame_lib.calc_age_mass(mags, None, None, None, None, plot=plot, threads=1, nwalkers=100, steps=1000, nburn=200)
# print(datetime.datetime.now() - start)
# start = datetime.datetime.now()
# mcmame_lib.calc_age_mass(mags, None, None, None, None, plot=plot, threads=1, nwalkers=1000, steps=500, nburn=500)
# print(datetime.datetime.now() - start)
# start = datetime.datetime.now()
# mcmame.calc_age_mass(mags, None, None, None, None, grids, reddening_grids, plot=plot, threads=1, nwalkers=100, steps=500, nburn=200)
# print(datetime.datetime.now() - start)
#     calc_age_mass(mags, metal, metal_e, A_V2, A_V2_e, plot=plot, threads=1, nwalkers=100, steps=200, nburn=200)    
#     calc_age_mass(mags, metal, metal_e, A_V, A_V_e, A_V2=A_V2, A_V2_e=A_V2_e, plot=plot, threads=1, nwalkers=100, steps=200, nburn=200)    

print()

if plot:
    plt.show()