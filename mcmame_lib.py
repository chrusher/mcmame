#! /usr/bin/env python

import logging
import os
import pickle

import numpy as np
from scipy import stats

from astropy import table
from astropy.io import fits

import ptemcee

mag_sun = table.Table.read(os.path.expanduser('~') + '/sluggs/sps_models/mag_sun.fits')
    
    
def get_mag_sun(name, AB=True):
    
    row = mag_sun[mag_sun['filter'] == name][0]
        
    if AB:
        return row['AB']
    else:
        return row['Vega']
    
def prob_bi(theta, mags, metal, metal_ivar2, A_V, A_V_ivar2, A_V2, A_V2_ivar2):
    lnprob = (metal - theta[0])**2 * metal_ivar2
    red = -(A_V - theta[3])**2 * A_V_ivar2
    red_2 = -(A_V2 - theta[3])**2 * A_V2_ivar2
    lnprob += -np.logaddexp(red, red_2)
    for mag, mag_ivars2, reddening_grid, grid in mags:
        lnprob += (mag - theta[3] * reddening_grid(theta[0], theta[1])[0][0] - grid(theta[0], theta[1])[0][0] + 2.5 * theta[2])**2 * mag_ivars2
    return -lnprob

def prob(theta, mags, metal, metal_ivar2, A_V, A_V_ivar2):
    lnprob = (metal - theta[0])**2 * metal_ivar2
    lnprob += (A_V - theta[3])**2 * A_V_ivar2
    for mag, mag_ivars2, reddening_grid, grid in mags:
        lnprob += (mag - theta[3] * reddening_grid(theta[0], theta[1])[0][0] - grid(theta[0], theta[1])[0][0] + 2.5 * theta[2])**2 * mag_ivars2
    return -lnprob
        
def prior(theta, metal_lower, metal_upper, age_lower, age_upper, A_V_lower, A_V_upper):
    if theta[0] > metal_upper or theta[0] < metal_lower:
        return -np.inf
    if theta[1] > age_upper or theta[1] < age_lower:
        return -np.inf
    if theta[3] > A_V_upper or theta[3] < A_V_lower:
        return -np.inf    
    else:
        return 0.
        
class ln_prior:
    
    def __init__(self, metal_lower=-3.0, metal_upper=0.7, age_lower=0.1,
                 age_upper=15.84, A_V_lower=0., A_V_upper=np.inf):
        self.metal_lower = metal_lower
        self.metal_upper = metal_upper
        self.age_lower = age_lower
        self.age_upper = age_upper
        self.A_V_lower = A_V_lower
        self.A_V_upper = A_V_upper
        
    def __call__(self, x):
        return prior(x, self.metal_lower, self.metal_upper, self.age_lower,
                     self.age_upper, self.A_V_lower, self.A_V_upper)
            
class ln_prob_bi:
    
    def __init__(self, mags, metal, metal_e, A_V, A_V_e, A_V2, A_V2_e):
        self.mags = mags
        self.metal = metal    
        self.metal_ivar2 = metal_e**-2
        self.A_V = A_V
        self.A_V_ivar2 = A_V_e**-2
        self.A_V2 = A_V2
        self.A_V2_ivar2 = A_V2_e**-2       
        
    def __call__(self, x):
        return prob_bi(x, self.mags, self.metal, self.metal_ivar2,
                       self.A_V, self.A_V_ivar2, self.A_V2, self.A_V2_ivar2)
    
class ln_prob:
    
    def __init__(self, mags, metal, metal_e, A_V, A_V_e):
        self.mags = mags
        self.metal = metal    
        self.metal_ivar2 = metal_e-2
        self.A_V = A_V
        self.A_V_ivar2 = A_V_e**-2
        
    def __call__(self, x):
        return prob(x, self.mags, self.metal, self.metal_ivar2,
                    self.A_V, self.A_V_ivar2)       
    

def calc_age_mass(magnitudes, metal, metal_e, A_V, A_V_e, grids=None,
                  reddening_grids=None, plot=False, age_guess=8, mass_guess=5.3,
                  nwalkers=1000, steps=500, thin=10, keep_chain=False,
                  threads=4, metal_lower=-3.0, metal_upper=0.7, age_lower=0.1,
                  age_upper=15.84, A_V_lower=0., A_V_upper=np.inf, A_V2=0,
                  A_V2_e=0, ntemps=8, nburn=500, logger=None, verbose=True):

    if logger is None:
        logger = logging.getLogger()
    
    input_str = 'Input:\n'
    if metal is not None and metal_e is not None:
        input_str += '[Z/H] {:.3f} {:.3f}'.format(metal, metal_e)
    input_str += '  A_V  {:.3f} {:.3f}'.format(A_V, A_V_e)
    if A_V2_e:
        input_str += ' {:.3f} {:.3f}'.format(A_V2, A_V2_e)
    input_str += '\n'
    input_str += magnitude_str(magnitudes)
    input_str += '\nPriors:\n'
    input_str += '{:.3f} < [Z/H] < {:.3f}\n'.format(metal_lower, metal_upper)
    input_str += '{:.3f} < Age < {:.3f}\n'.format(age_lower, age_upper)
    input_str += '{:.3f} < A_V < {:.3f}\n'.format(A_V_lower, A_V_upper)
    logger.info(input_str)
    if verbose:
        print(input_str)

    #need to update these with magnitude specific grids
    if grids is None:
        with open(os.path.expanduser('~') + '/sluggs/sps_models/fsps_mist_inter_mags.pickle', 'rb') as f:
            grids = pickle.load(f)
    
    if reddening_grids is None:
        with open(os.path.expanduser('~') + '/sluggs/sps_models/fsps_reddening_mist_inter_mags.pickle', 'rb') as f:
            reddening_grids = pickle.load(f)

    if metal is None or metal_e is None:
        metal = -1
        metal_e = 10
    
    mags = get_mags(magnitudes, reddening_grids, grids)
    metal_guess, age_guess, mass_guess, A_V_guess = grid_search(mags, metal,
        metal_e, A_V, A_V_e, A_V2, A_V2_e, age_lower, age_upper, metal_lower,
        metal_upper, A_V_lower, A_V_upper, logger)


    def start_array(guess, nwalkers, lower, upper):
        start = guess + 1e-2 * np.random.randn(nwalkers)
        start[start < lower] = lower
        start[start > upper] = upper
        return start 

    start = [start_array(metal_guess, nwalkers, metal_lower, metal_upper),
             start_array(age_guess, nwalkers, age_lower, age_upper),
             start_array(mass_guess, nwalkers, -np.inf, np.inf),
             start_array(A_V_guess, nwalkers, A_V_lower, A_V_upper)]
    
    start = np.asarray(start).T


    if A_V2_e:
        logl = ln_prob_bi(mags, metal, metal_e, A_V, A_V_e, A_V2, A_V2_e)
    else:
        logl = ln_prob(mags, metal, metal_e, A_V, A_V_e)
    logprior = ln_prior(metal_lower, metal_upper, age_lower,
                        age_upper, A_V_lower, A_V_upper)

    log_likely = logl([metal_guess, age_guess, mass_guess, A_V_guess])
    start_str = 'Starting at:\n{:.3f} {:.3f} {:.3f} {:.3f}\nStarting log likelihood {:.3f}\n'.format(metal_guess,
                    age_guess, mass_guess, A_V_guess, log_likely)          
    logger.info(start_str)
    if verbose:
        print(start_str)    
    
    sampler = ptemcee.Sampler(nwalkers, start.shape[-1], logl, logprior,
                                      threads=threads, ntemps=ntemps)

    temp_start = []
    for i in range(ntemps):
        temp_start.append(start)
    temp_start = np.array(temp_start)    
    
    sampler.run_mcmc(temp_start, (nburn + steps))
    samples = sampler.chain[0, :, nburn:, :].reshape((-1, start.shape[-1]))
            
    samples = samples[::thin]
    
    if threads > 1:
        sampler.pool.close()
        
    if keep_chain:
        np.save(open(str(keep_chain) + '_chain.npy', 'w'), np.asarray(sampler.chain))
         
    norm_percentiles = stats.norm.cdf([-2, -1, 0, 1, 2]) * 100
    Z_precentiles = np.percentile(samples[:,0], norm_percentiles)
    age_precentiles = np.percentile(samples[:,1], norm_percentiles)
    mass_precentiles = np.percentile(samples[:,2], norm_percentiles)
    A_V_precentiles = np.percentile(samples[:,3], norm_percentiles)
    
    output_str = 'Output:\n' 
    output_str += '[Z/H] ' + ' '.join(['{:.3f}\n'.format(Z) for Z in Z_precentiles]) + '\n'
    output_str += '       {:.3f} \u00B1{:.3f}\n'.format(np.mean(samples[:,0]), np.std(samples[:,0])) + '\n'
    output_str += 'age   ' + ' '.join(['{:.3f}\n'.format(age) for age in age_precentiles]) + '\n'
    output_str += '       {:.3f} \u00B1{:.3f}\n'.format(np.mean(samples[:,1]), np.std(samples[:,1])) + '\n'
    output_str += 'mass  ' + ' '.join(['{:.3f}\n'.format(mass) for mass in mass_precentiles]) + '\n'
    output_str += '       {:.3f} \u00B1{:.3f}\n'.format(np.mean(samples[:,2]), np.std(samples[:,2])) + '\n'
    output_str += 'A_V   ' + ' '.join(['{:.3f}\n'.format(red) for red in A_V_precentiles]) + '\n'
    output_str += '       {:.3f} \u00B1{:.3f}\n'.format(np.mean(samples[:,3]), np.std(samples[:,3])) + '\n'
    log_likely = logl([Z_precentiles[2], age_precentiles[2], mass_precentiles[2], A_V_precentiles[2]])
    output_str += 'Log likelihood: {:.3f}\n'.format(log_likely)
    logger.info(output_str)
    if verbose:
        print(output_str)    
        
    if plot:
        import corner
        corner.corner(samples, labels=['[Z/H]', 'Age', 'Log Mass', 'A_V'], quantiles=[0.16, 0.50, 0.84], show_titles=True)

    return samples, Z_precentiles[1:-1], age_precentiles[1:-1], mass_precentiles, A_V_precentiles[1:-1]


def get_mags(magnitudes, reddening_grids, grids):
    mags = []
    for name, mag, mag_e in magnitudes:
        mags.append((mag, mag_e**-2, reddening_grids[name], grids[name]))
    return mags

def magnitude_str(magnitudes):

    output = ''
    for magnitude in magnitudes:
        output += '{} {:.3f} {:.3f}  '.format(*magnitude)

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
    corner.corner(corner_array(sample, ['Z', 'age', 'mass', 'A_V']),
                  labels=['[Z/H]', 'age', 'mass', 'A_V'],
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

    
def find_mass(mags, metal_guess, age_guess, A_V_guess):

    observed = []
    model = []
    for mag, mag_ivar2, reddening_grid, grid in mags:
        observed.append((-mag + A_V_guess * reddening_grid(metal_guess, age_guess)[0][0] + grid(metal_guess, age_guess)[0][0]) * mag_ivar2**0.5 / 2.5)
        model.append([mag_ivar2**0.5])
    observed = np.array(observed)
    model = np.array(model)
    mass = np.linalg.lstsq(model, observed, rcond=None)[0][0]
    return mass


def grid_search(mags, metal, metal_e, A_V, A_V_e, A_V2, A_V2_e, age_lower,
                age_upper, metal_lower, metal_upper, A_V_lower, A_V_upper,
                logger):
    best_likely = -np.inf
    best_guess = None
    
    metal_range = np.arange(-2.5, 0.3, 0.25)
    metal_range = metal_range[(metal_range >= metal_lower) & (metal_range <= metal_upper)]
    age_range = 10**np.arange(-1, 1.2, 0.1)
    age_range = age_range[(age_range >= age_lower) & (age_range <= age_upper)]
    A_V_range = np.arange(0., max(A_V, A_V2) + max(A_V_e, A_V2_e) + 0.1, 0.1)
    A_V_range = A_V_range[(A_V_range >= A_V_lower) & (A_V_range <= A_V_upper)]
        
    metal_ivar2 = metal_e**-2
    A_V_ivar2 = A_V_e**-2
        
        
    if A_V2_e:
        A_V2_ivar2 = A_V2_e**-2
            
    for metal_guess in metal_range:
        for age_guess in age_range:
            for A_V_guess in A_V_range:
                mass_guess = find_mass(mags, metal_guess, age_guess, A_V_guess)
                if A_V2_e:
                    ln_likely = prob_bi((metal_guess, age_guess, mass_guess, A_V_guess), mags, metal, metal_ivar2, A_V, A_V_ivar2, A_V2, A_V2_ivar2)
                else:
                    ln_likely = prob((metal_guess, age_guess, mass_guess, A_V_guess), mags, metal, metal_ivar2, A_V, A_V_ivar2)
                
                if ln_likely > best_likely:
                    best_guess = (metal_guess, age_guess, mass_guess, A_V_guess)
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
    A_V = 3.1 * 0.08
    A_V_e = 3.1 * 0.03
    
    with open(os.path.expanduser('~') + '/sluggs/sps_models/fsps_mist_inter_mags.pickle', 'rb') as f:
        grids = pickle.load(f)

    with open(os.path.expanduser('~') + '/sluggs/sps_models/fsps_reddening_mist_inter_mags.pickle', 'rb') as f:
        reddening_grids = pickle.load(f)    
    
    age = 11
    metal = -1
    metal_e = 0.2
    mass = 5
    A_V = 0.3
    A_V_e = 0.1
    A_V2 = 1
    A_V2_e = 0.5
    
    mags = []
    
    for band in ['u', 'g', 'r', 'i', 'z']:

        mag = grids[band].ev(metal, age) - 2.5 * mass + 0.75 * reddening_grids[band].ev(metal, age)
        mags.append([band, mag, 0.02])
#         print(band, mag, grids[band].ev(metal, age), - 2.5 * mass, A_V2 * reddening_grids[band].ev(metal, age))
        
#     calc_age_mass(mags, metal, metal_e, A_V, A_V_e, plot=plot, threads=1, nwalkers=100, steps=200, nburn=200, verbose=False)
    
    calc_age_mass(mags, metal, metal_e, A_V, A_V_e, A_V2=A_V2, A_V2_e=A_V2_e, plot=plot, threads=1, nwalkers=100, steps=200, nburn=200, verbose=False)
    
    print()
    
    if plot:
        plt.show()

