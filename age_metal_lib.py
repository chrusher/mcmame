#! /usr/bin/env python

from __future__ import print_function


import logging
import os
import pickle

import numpy as np
from scipy import stats

from astropy import table
from astropy.io import fits

import emcee
import ptemcee

mag_sun = table.Table.read(os.path.expanduser('~') + '/sluggs/sps_models/mag_sun.fits')

#Schlafly & Finkbeiner (2011)
reddenings = {'u':4.239,
              'g':3.303,
              'r':2.285,
              'i':1.698,
              'z':1.263,
              'U':4.107,
              'B':3.641,
              'V':2.682,
              'R':2.119,
              'I':1.516,
              'J':0.709,
              'H':0.449,
              'K':0.302,
              'WFC_ACS_F435W':3.610,
              'WFC_ACS_F475W':3.268,
              'WFC_ACS_F555W':2.792,
              'WFC_ACS_F606W':2.471,
              'WFC_ACS_F625W':2.219,
              'WFC_ACS_F775W':1.629,
              'WFC_ACS_F814W':1.526,
              'WFC_ACS_F850LP':1.243,
              'WFC3_UVIS_F218W':7.760,
              'WFC3_UVIS_F275W':5.487,
              'WFC3_UVIS_F336W':4.453,
              'WFC3_UVIS_F438W':3.623,
              'WFC3_UVIS_F475W':3.248,
              'WFC3_UVIS_F555W':2.855,
              'WFC3_UVIS_F606W':2.488,
              'WFC3_UVIS_F814W':1.536,
              'WFC3_UVIS_F850LP':1.208,
              'WFC3_IR_F105W':0.969,
              'WFC3_IR_F110W':0.881,
              'WFC3_IR_F125W':0.726,
              'WFC3_IR_F140W':0.613,
              'WFC3_IR_F160W':0.512,
              'PS1_g':3.172,
              'PS1_r':2.271,
              'PS1_i':1.682,
              'PS1_z':1.322,
              'PS1_y':1.087,
              'JWST_F070W':1.692, # calculated using wave**-1.5
              'JWST_F090W':1.167,
              'JWST_F115W':0.807,
              'JWST_F150W':0.544,
              'JWST_F200W':0.356,
              'JWST_F277W':0.218,
              'JWST_F356W':0.148,
              'JWST_F444W':0.108,
              'LSST_u':4.145,
              'LSST_g':3.237,
              'LSST_r':2.273,
              'LSST_i':1.684,
              'LSST_z':1.323,
              'LSST_y':1.088,
              'Euclid_VIS':1.671, # calculated using wave**-1.5
              'Euclid_Y':0.887,
              'Euclid_J':0.628,
              'Euclid_H':0.425,
              'Roman_F062':2.000, # calculated using wave**-1.5
              'Roman_F087':1.237,
              'Roman_F106':0.922,
              'Roman_F129':0.683,
              'Roman_F158':0.506,
              'Roman_F184':0.401,
              'Castor_uv':6.278, # from FSPS with different dust attenuations
              'Castor_u':4.174,
              'Castor_g':3.091,
             }

for band in ['u', 'g', 'r', 'i', 'z']:
    reddenings['SDSS_' + band] = reddenings[band]
    reddenings['MegaCam_' + band] = reddenings[band]
    
for band in ['R', 'I']:
    reddenings['Cousins_' + band] = reddenings[band]
    

reddenings['2MASS_J'] = reddenings['J']
reddenings['2MASS_H'] = reddenings['H']
reddenings['2MASS_Ks'] = reddenings['K']

lumin_reddenings = {}
for band in reddenings.keys():
    lumin_reddenings[band] = reddenings[band] / -2.5
    
    
def get_mag_sun(name, AB=True):
    
    row = mag_sun[mag_sun['filter'] == name][0]
        
    if AB:
        return row['AB']
    else:
        return row['Vega']
    

def ln_prob_red(theta, lumins, metal, metal_e, ebv, ebv_e):                             
    lnprob = ((metal - theta[0]) / metal_e)**2 / 2
    lnprob += ((ebv - theta[3]) / ebv_e)**2 / 2
    for lumin, lumin_e, reddening, grid in lumins:
        lnprob += ((lumin - theta[3] * reddening - (grid(theta[0], theta[1])[0][0] + theta[2])) / lumin_e)**2 / 2
    return -lnprob

def ln_prob_bi_red_fast(theta, lumins, metal, metal_ivar2, ebv, ebv_ivar2, ebv2, ebv2_ivar2):
    lnprob = (metal - theta[0])**2 * metal_ivar2
    red = -(ebv - theta[3])**2 * ebv_ivar2
    red_2 = -(ebv2 - theta[3])**2 * ebv2_ivar2
    lnprob += -np.logaddexp(red, red_2)
    for lumin, lumin_ivars2, reddening, grid in lumins:
        lnprob += (lumin - theta[3] * reddening - grid(theta[0], theta[1])[0][0] + theta[2])**2 * lumin_ivars2
    return -lnprob

def ln_prop_bi_varred_fast(theta, lumins, metal, metal_ivar2, AV, AV_ivar2, AV2, AV2_ivar2):
    lnprob = (metal - theta[0])**2 * metal_ivar2
    red = -(AV - theta[3])**2 * AV_ivar2
    red_2 = -(AV2 - theta[3])**2 * AV2_ivar2
    lnprob += -np.logaddexp(red, red_2)
    for lumin, lumin_ivars2, reddening_grid, grid in lumins:
        lnprob += (lumin - theta[3] * reddening_grid(theta[0], theta[1])[0][0] - grid(theta[0], theta[1])[0][0] + theta[2])**2 * lumin_ivars2
    return -lnprob

def ln_prob_bi_red(theta, lumins, metal, metal_e, ebv, ebv_e, ebv2, ebv2_e): 
    lnprob = ((metal - theta[0]) / metal_e)**2 / 2
    red = -((ebv - theta[3]) / ebv_e)**2 / 2
    red_2 = -((ebv2 - theta[3]) / ebv2_e)**2 / 2    
    lnprob += -np.logaddexp(red, red_2)
    for lumin, lumin_e, reddening, grid in lumins:
        lnprob += ((lumin - theta[3] * reddening - (grid(theta[0], theta[1])[0][0] + theta[2])) / lumin_e)**2 / 2
    return -lnprob
    
def ln_prob(theta, lumins, metal, metal_e):
    lnprob = ((metal - theta[0]) / metal_e)**2 / 2
    for lumin, lumin_e, reddening, grid in lumins:
        lnprob += ((lumin - (grid(theta[0], theta[1])[0][0] + theta[2])) / lumin_e)**2 / 2
    return -lnprob  

def ln_prior(theta):
    
    if theta[0] > 0.7 or theta[0] < -3.0:
        return -np.inf
    if theta[1] > 15.84 or theta[1] < 0.1:
        return -np.inf
    else:
        return 0.  

def ln_prior_red(theta):
    
    if theta[0] > 0.7 or theta[0] < -3.0:
        return -np.inf
    if theta[1] > 15.84 or theta[1] < 0.1:
        return -np.inf
    if theta[3] < 0.:
        return -np.inf    
    else:
        return 0.
        
def prior_red(theta, metal_lower, metal_upper, age_lower, age_upper, ebv_lower, ebv_upper):
    if theta[0] > metal_upper or theta[0] < metal_lower:
        return -np.inf
    if theta[1] > age_upper or theta[1] < age_lower:
        return -np.inf
    if theta[3] > ebv_upper or theta[3] < ebv_lower:
        return -np.inf    
    else:
        return 0.
        
class pt_ln_prior_red:
    
    def __init__(self, metal_lower=-3.0, metal_upper=0.7, age_lower=0.1,
                 age_upper=15.84, ebv_lower=0., ebv_upper=np.inf):
        self.metal_lower = metal_lower
        self.metal_upper = metal_upper
        self.age_lower = age_lower
        self.age_upper = age_upper
        self.ebv_lower = ebv_lower
        self.ebv_upper = ebv_upper
        
    def __call__(self, x):
        return prior_red(x, self.metal_lower, self.metal_upper, self.age_lower,
                         self.age_upper, self.ebv_lower, self.ebv_upper)
        
def em_ln_prob_red(theta, lumins, metal, metal_e, ebv, ebv_e):
    return ln_prob_red(theta, lumins, metal, metal_e, ebv, ebv_e) + ln_prior_red(theta)

def em_ln_prob_bi_red(theta, lumins, metal, metal_e, ebv, ebv_e, ebv2, ebv2_e):
    return ln_prob_bi_red(theta, lumins, metal, metal_e, ebv, ebv_e, ebv2, ebv2_e) + ln_prior_red(theta)

def em_ln_prob(theta, lumins, metal, metal_e):
    return ln_prob(theta, lumins, metal, metal_e) + ln_prior(theta)
    
class pt_ln_prob:
    
    def __init__(self, lumins, metal, metal_e):
        self.lumins = lumins
        self.metal = metal    
        self.metal_e = metal_e
        
    def __call__(self, x):
        return ln_prob(x, self.lumins, self.metal, self.metal_e)
    
class pt_ln_prob_red:
    
    def __init__(self, lumins, metal, metal_e, ebv, ebv_e):
        self.lumins = lumins
        self.metal = metal    
        self.metal_e = metal_e
        self.ebv = ebv
        self.ebv_e = ebv_e
        
    def __call__(self, x):
        return ln_prob_red(x, self.lumins, self.metal, self.metal_e, self.ebv, self.ebv_e)
    
class pt_ln_prob_bi_red:
    
    def __init__(self, lumins, metal, metal_e, ebv, ebv_e, ebv2, ebv2_e):
        self.lumins = lumins
        self.metal = metal    
        self.metal_e = metal_e
        self.ebv = ebv
        self.ebv_e = ebv_e
        self.ebv2 = ebv2
        self.ebv2_e = ebv2_e        
        
    def __call__(self, x):
        return ln_prob_bi_red(x, self.lumins, self.metal, self.metal_e,
                              self.ebv, self.ebv_e, self.ebv2, self.ebv2_e)    
    

def calc_age_mass(luminosities, metal, metal_e, ebv, ebv_e, grids=None,
                  plot=False, age_guess=8, mass_guess=5.3, sampler_type='PT',
                  nwalkers=1000, steps=500, thin=10, keep_chain=False,
                  threads=4, metal_lower=-3.0, metal_upper=0.7, age_lower=0.1,
                  age_upper=15.84, ebv_lower=0., ebv_upper=np.inf, ebv2=0,
                  ebv2_e=0, ntemps=8, nburn=500, logger=None, verbose=True):

    if logger is None:
        logger = logging.getLogger()
    
    input_str = 'Input:\n'
    if metal is not None and metal_e is not None:
        input_str += '[Z/H] {:.3f} {:.3f}'.format(metal, metal_e)
    if ebv_e:
        input_str += ' E(B-V) {:.3f} {:.3f}'.format(ebv, ebv_e)
        if ebv2_e:
            input_str += ' {:.3f} {:.3f}'.format(ebv2, ebv2_e)
    input_str += '\n'
    input_str += luminosity_str(luminosities)
    input_str += '\nPriors:\n'
    input_str += '{:.3f} < [Z/H] < {:.3f}\n'.format(metal_lower, metal_upper)
    input_str += '{:.3f} < Age < {:.3f}\n'.format(age_lower, age_upper)
    input_str += '{:.3f} < E(B-V) < {:.3f}\n'.format(ebv_lower, ebv_upper)
    logger.info(input_str)
    if verbose:
        print(input_str)

    if grids is None:
        with open(os.path.expanduser('~') + '/sluggs/sps_models/fsps_mist_inter.pickle', 'rb') as f:
            grids = pickle.load(f)
            
        with open(os.path.expanduser('~') + '/sluggs/sps_models/fsps_reddening_mist_inter.pickle', 'rb') as f:
            reddening_grids = pickle.load(f)

    if metal is None or metal_e is None:
        metal = -1
        metal_e = 10
    
    lumins = get_lumins(luminosities, grids)
    metal_guess, age_guess, mass_guess, ebv_guess = grid_search(lumins, metal,
        metal_e, ebv, ebv_e, ebv2, ebv2_e, age_lower, age_upper, metal_lower,
        metal_upper, ebv_lower, ebv_upper, logger)
    start_str = 'Starting at:\n{:.3f} {:.3f} {:.3f} {:.3f}\n'.format(metal_guess,
                    age_guess, mass_guess, ebv_guess)          
    logger.info(start_str)
    if verbose:
        print(start_str)

    def start_array(guess, nwalkers, lower, upper):
        start = guess + 1e-2 * np.random.randn(nwalkers)
        start[start < lower] = lower
        start[start > upper] = upper
        return start 

    start = [start_array(metal_guess, nwalkers, metal_lower, metal_upper),
             start_array(age_guess, nwalkers, age_lower, age_upper),
             start_array(mass_guess, nwalkers, -np.inf, np.inf)]
    if ebv_e:
            start.append(start_array(ebv_guess, nwalkers, ebv_lower, ebv_upper))
    
    start = np.asarray(start).T

    if sampler_type == 'PT':
        
        temp_start = []
        for i in range(ntemps):
            temp_start.append(start)
        temp_start = np.array(temp_start)

        if ebv_e == 0:
            logl = pt_ln_prob(lumins, metal, metal_e) 
            sampler = ptemcee.Sampler(nwalkers, start.shape[-1], logl,
                                      ln_prior, threads=threads, ntemps=ntemps)
        elif ebv2_e == 0:
            logl = pt_ln_prob_red(lumins, metal, metal_e, ebv, ebv_e) 
            logprior = pt_ln_prior_red(metal_lower, metal_upper, age_lower,
                                       age_upper, ebv_lower, ebv_upper)
            sampler = ptemcee.Sampler(nwalkers, start.shape[-1], logl, logprior,
                                      threads=threads, ntemps=ntemps)
        else:
            logl = pt_ln_prob_bi_red(lumins, metal, metal_e, ebv, ebv_e, ebv2, ebv2_e) 
            logprior = pt_ln_prior_red(metal_lower, metal_upper, age_lower,
                                       age_upper, ebv_lower, ebv_upper)
            sampler = ptemcee.Sampler(nwalkers, start.shape[-1], logl, logprior,
                                      threads=threads, ntemps=ntemps)
            
        sampler.run_mcmc(temp_start, (nburn + steps))
        samples = sampler.chain[0, :, nburn:, :].reshape((-1, start.shape[-1]))
        
    else:
        if ebv_e == 0:
            sampler = emcee.EnsembleSampler(nwalkers, start.shape[-1],
                    em_ln_prob, args=(lumins, metal, metal_e), threads=threads) 
        elif ebv2_e == 0:
            sampler = emcee.EnsembleSampler(nwalkers, start.shape[-1],
                    em_ln_prob_red, threads=threads,
                    args=(lumins, metal, metal_e, ebv, ebv_e))
        else:
            sampler = emcee.EnsembleSampler(nwalkers, start.shape[-1],
                        em_ln_prob_bi_red, threads=threads,
                        args=(lumins, metal, metal_e, ebv, ebv_e, ebv2, ebv2_e))
            
        sampler.run_mcmc(start, (nburn + steps))
        samples = sampler.chain[:, nburn:, :].reshape((-1, start.shape[-1]))
    
    samples = samples[::thin]
    
    if threads > 1:
        sampler.pool.close()
        
    if keep_chain:
        np.save(open(str(keep_chain) + '_chain.npy', 'w'), np.asarray(sampler.chain))
         
    norm_percentiles = stats.norm.cdf([-2, -1, 0, 1, 2]) * 100
    Z_precentiles = np.percentile(samples[:,0], norm_percentiles)
    age_precentiles = np.percentile(samples[:,1], norm_percentiles)
    mass_precentiles = np.percentile(samples[:,2], norm_percentiles)
    if ebv_e:
        ebv_precentiles = np.percentile(samples[:,3], norm_percentiles)
    
    
    output_str = 'Output:\n' 
    output_str += '[Z/H] ' + ' '.join(['{:.3f}'.format(Z) for Z in Z_precentiles]) + '\n'
    output_str += '       {:.3f} \u00B1{:.3f}'.format(np.mean(samples[:,0]), np.std(samples[:,0])) + '\n'

    output_str += 'age   ' + ' '.join(['{:.3f}'.format(age) for age in age_precentiles]) + '\n'
    output_str += '       {:.3f} \u00B1{:.3f}'.format(np.mean(samples[:,1]), np.std(samples[:,1])) + '\n'
    output_str += 'mass  ' + ' '.join(['{:.3f}'.format(mass) for mass in mass_precentiles]) + '\n'
    output_str += '       {:.3f} \u00B1{:.3f}'.format(np.mean(samples[:,2]), np.std(samples[:,2])) + '\n'
    if ebv_e:
        output_str += 'E(B-V) ' + ' '.join(['{:.3f}'.format(red) for red in ebv_precentiles]) + '\n'
        output_str += '       {:.3f} \u00B1{:.3f}'.format(np.mean(samples[:,3]), np.std(samples[:,3])) + '\n'        
    logger.info(output_str)
    if verbose:
        print(output_str)    
        
    if plot:
        import corner
        if ebv_e == 0:
            corner.corner(samples, labels=['[Z/H]', 'Age', 'Log Mass'], quantiles=[0.16, 0.50, 0.84], show_titles=True)
        else:
            corner.corner(samples, labels=['[Z/H]', 'Age', 'Log Mass', 'E(B-V)'], quantiles=[0.16, 0.50, 0.84], show_titles=True)

    if ebv_e:
        return samples, Z_precentiles[1:-1], age_precentiles[1:-1], mass_precentiles, ebv_precentiles[1:-1]
    else:
        return samples, Z_precentiles[1:-1], age_precentiles[1:-1], mass_precentiles

def get_lumins(luminosities, grids):
    
    lumins = []
    for name, log_L, log_L_e in luminosities:
        lumins.append((log_L, log_L_e, lumin_reddenings[name], grids[name]))
    return lumins

def calc_luminosities(magnitudes, m_M=0, AB=True):

    luminosities = []
    for magnitiude in magnitudes:
        name, mag, mag_e = magnitiude
        log_L = (mag - m_M - get_mag_sun(name, AB)) / -2.5
        log_L_e = mag_e / 2.5

        luminosities.append((name, log_L, log_L_e))

    return luminosities


def calc_magnitudes(luminosities, m_M=0, AB=True):
    magnitudes = []
    for luminosity in luminosities:
        name, log_L, log_L_e = luminosity
        mag = -2.5 * log_L + get_mag_sun(name, AB) + m_M
        mag_e = 2.5 * log_L_e

        magnitudes.append((name, mag, mag_e))

    return magnitudes

def luminosity_str(luminosities):

    output = ''
    for luminosity in luminosities:
        output += '{} {:.3f} {:.3f}  '.format(*luminosity)

    return output


def load_samples(catalogue, directory='.', verbose=False):
    
    samples = []
    
    for entry in catalogue:
        path = os.path.join(directory, entry['output'] + '.fits')
        with fits.open(path) as hdul:
            sample = hdul[2].data
        if verbose:
            print(entry['output'], sample.shape)
        samples.append(sample)
            
    samples = np.vstack(samples)
    return samples

def corner_array(array, names):

    return np.asarray([array[name] for name in names]).T

def plot_posterior(filename, verbose=False):

    import corner

    with fits.open(filename) as hdul:
        entry = table.Table.read(hdul[1])
        sample = hdul[2].data

    if verbose:
        print(entry[0])
    corner.corner(corner_array(sample, ['Z', 'age', 'mass', 'ebv']),
                  labels=['[Z/H]', 'age', 'mass', 'E(B-V)'],
                  quantiles=[0.02, 0.16, 0.5, 0.84, 0.98],
                  show_titles=True)

def plot_samples(samples, catalogue, name, metal_range=(-2.4, 0.7), age_range=(0, 15.8), bins=100):
    
    import matplotlib.pyplot as plt
    from matplotlib import colors
    
    h = np.histogram2d(samples['age'], samples['Z'], bins=bins, range=(age_range, metal_range))
    
    percentages = np.percentile(h[0], [5, 95])
    percentages = np.percentile(h[0], [0, 99])

    plt.figure(figsize=(8, 8))
    plt.subplots_adjust(top=0.93, bottom=0.07, left=0.1, right=0.9, hspace=0.02, wspace=0.02)
    
    plt.subplot2grid((3, 3), (0, 0), colspan=2)
#     plt.hist(samples['age'], bins=bins, range=age_range, density=True, histtype='stepfilled')
    plt.hist(catalogue['age'], bins=bins // 5, range=age_range, density=True, histtype='stepfilled')
    plt.xlabel('(Pseudo-)Age (Gyr)')
    plt.xlim(*age_range)
    axes = plt.gca()
    axes.xaxis.tick_top()
    axes.xaxis.set_label_position('top')
    axes.yaxis.set_ticklabels([])
    
    plt.subplot2grid((3, 3), (0, 2))
    plt.text(0.5, 0.6, name, size='xx-large', horizontalalignment='center', verticalalignment='center')
    plt.text(0.5, 0.4, '{} GCs'.format(len(catalogue)), size='x-large', horizontalalignment='center', verticalalignment='center')
    plt.axis('off')
    
    plt.subplot2grid((3, 3), (1, 2), rowspan=2)
#     plt.hist(samples['Z_H'], bins=bins, range=metal_range, orientation='horizontal', density=True, histtype='stepfilled')
    plt.hist(catalogue['Z'], bins=bins // 5, range=metal_range, orientation='horizontal', density=True, histtype='stepfilled')
    
    plt.ylim(*metal_range)
    axes = plt.gca()
    axes.yaxis.tick_right()
    axes.yaxis.set_label_position('right')
    axes.xaxis.set_ticklabels([])
    plt.ylabel('[Z/H]') 
    
    plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
    plt.hist2d(samples['age'], samples['Z'], bins=bins, range=(age_range, metal_range), norm=colors.Normalize(percentages[0], percentages[1]))
    plt.plot(catalogue['age'], catalogue['Z'], 'wo')
    plt.xlabel('Age (Gyr)')
    plt.ylabel('[Z/H]') 

    
def find_mass(lumins, metal_guess, age_guess, ebv_guess):

    observed = []
    model = []
    for lumin, lumin_e, reddening, grid in lumins:
        observed.append((lumin - ebv_guess * reddening - grid(metal_guess, age_guess)[0][0]) / lumin_e)
        model.append([1 / lumin_e])
    observed = np.array(observed)
    model = np.array(model)
    mass = np.linalg.lstsq(model, observed, rcond=None)[0][0]
    return mass


def grid_search(lumins, metal, metal_e, ebv, ebv_e, ebv2, ebv2_e, age_lower,
                age_upper, metal_lower, metal_upper, ebv_lower, ebv_upper,
                logger):
    best_likely = -np.inf
    best_guess = None

    metal_range = np.arange(-2.5, 0.3, 0.1)
    metal_range = metal_range[(metal_range >= metal_lower) & (metal_range <= metal_upper)]
    age_range = 10**np.hstack([np.arange(-0.9, 1.0, 0.1), np.arange(0.95, 1.2, 0.05)])
    age_range = age_range[(age_range >= age_lower) & (age_range <= age_upper)]
    if ebv_e == 0:
        ebv_range = [ebv]
    else:
        ebv_range = np.arange(0., ebv + ebv_e + 0.02, 0.02)
        ebv_range = ebv_range[(ebv_range >= ebv_lower) & (ebv_range <= ebv_upper)]
            
    for metal_guess in metal_range:
        for age_guess in age_range:
            for ebv_guess in ebv_range:
                mass_guess = find_mass(lumins, metal_guess, age_guess, ebv_guess)
                if ebv_e == 0:
                    ln_likely = ln_prob((metal_guess, age_guess, mass_guess, ebv_guess), lumins, metal, metal_e)
                elif ebv2_e == 0:
                    ln_likely = ln_prob_red((metal_guess, age_guess, mass_guess, ebv_guess), lumins, metal, metal_e, ebv, ebv_e)
                else:
                    ln_likely = ln_prob_bi_red((metal_guess, age_guess, mass_guess, ebv_guess), lumins, metal, metal_e, ebv, ebv_e, ebv2, ebv2_e)
                if ln_likely > best_likely:
                    best_guess = (metal_guess, age_guess, mass_guess, ebv_guess)
                    best_likely = ln_likely
                    logger.debug(best_guess, best_likely)
                
    return best_guess
    

if __name__ == '__main__':
    LOG_FORMAT = "[%(asctime)s] %(levelname)8s %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    plot = True
    if plot:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams['figure.dpi'] = 50
        
    metal = -0.6822533361523742
    metal_e = 0.1595040018637024
    mags = [['u', -6.6404734, 0.029137253712118956],
            ['g', -8.340472, 0.01],
            ['r', -9.140472, 0.01],
            ['i', -9.5604725, 0.01],
            ['z', -9.840472, 0.01]]
    
    lumins = calc_luminosities(mags)
    
    logger = logging.getLogger('emcee') 
    calc_age_mass(lumins, metal, metal_e, 0.08, 0.03, plot=plot, threads=1, sampler_type='ES', nwalkers=100, steps=200, nburn=200, logger=logger, verbose=False) 
    print()
   
    logger = logging.getLogger('ptemcee') 
    calc_age_mass(lumins, metal, metal_e, 0.08, 0.03, plot=plot, threads=1, sampler_type='PT', nwalkers=100, steps=200, nburn=200, logger=logger, verbose=False) 
    print()
    
    if plot:
        plt.show()

