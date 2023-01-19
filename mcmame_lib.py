import logging
import os
import pickle

import numpy as np
from scipy import stats

from astropy import table
from astropy.io import fits

import ptemcee
import ptemcee.util
import emcee

# todo: don't hard code paths
mag_sun = table.Table.read(os.path.expanduser('~') + '/sluggs/sps_models/mag_sun.fits')
    
# how bright is the Sun in different filters and magnitide systems    
def get_mag_sun(name, AB=True):
    
    row = mag_sun[mag_sun['filter'] == name][0]
        
    if AB:
        return row['AB']
    else:
        return row['Vega']
    
def prob_bi(theta, mags, metal, metal_ivar2, A_V, A_V_ivar2, A_V2, A_V2_ivar2):
    delta = metal - theta[0]
    lnprob = delta * delta * metal_ivar2
    delta = A_V - theta[3]
    red = -delta * delta * A_V_ivar2
    delta = A_V2 - theta[3]
    red_2 = -delta * delta * A_V2_ivar2
    lnprob += -np.logaddexp(red, red_2)
    for mag, mag_ivars2, reddening_grid, grid in mags:
        delta = mag - theta[3] * reddening_grid(theta[0], theta[1])[0][0] - grid(theta[0], theta[1])[0][0] + 2.5 * theta[2]
        lnprob += delta * delta * mag_ivars2
    return -lnprob

def prob(theta, mags, metal, metal_ivar2, A_V, A_V_ivar2):
    delta = theta[0]
    delta = metal - theta[0]
    lnprob = delta * delta * metal_ivar2
    delta = A_V - theta[3]
    lnprob += delta * delta * A_V_ivar2
    for mag, mag_ivars2, reddening_grid, grid in mags:
        delta = mag - theta[3] * reddening_grid(theta[0], theta[1])[0][0] - grid(theta[0], theta[1])[0][0] + 2.5 * theta[2]
        lnprob += delta * delta * mag_ivars2
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
    
    
def em_prop(theta, mags, metal, metal_ivar2, A_V, A_V_ivar2, metal_lower, metal_upper, age_lower, age_upper, A_V_lower, A_V_upper):
    p = prior(theta, metal_lower, metal_upper, age_lower, age_upper, A_V_lower, A_V_upper)
    if p < 0:
        return -np.inf
    else:
        return p + prob(theta, mags, metal, metal_ivar2, A_V, A_V_ivar2)
    
# takes a list of magnitudes, and samples the posterior distributions of metallicity,
# age, mass and extinction subject to the Gaussian priors on metallicity and extinction
def calc_age_mass(magnitudes, metal, metal_e, A_V, A_V_e, grids=None,
                  reddening_grids=None, plot=False, nwalkers=1000, steps=500, thin=10,
                  keep_chain=False, threads=4, metal_lower=-3.0, metal_upper=0.5,
                  age_lower=0.001, age_upper=15., A_V_lower=0., A_V_upper=np.inf, A_V2=None,
                  A_V2_e=None, ntemps=8, nburn=500, logger=None, sampler='pt'):

    if logger is None:
        logger = logging.getLogger()
        
    if metal is None:
        metal = -1.
    if not metal_e:
        metal_e = np.inf
    if A_V is None:
        A_V = 0
    if not A_V_e:
        A_V_e = np.inf
    
    input_str = 'Input:\n'
    if np.isfinite(metal_e):
        input_str += '[Z/H] {:.3f} {:.3f}'.format(metal, metal_e)
    if np.isfinite(A_V_e):
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

    # I should not hard code paths like this
    if grids is None:
        with open(os.path.expanduser('~') + '/sluggs/sps_models/fsps_mist_inter_mags.pickle', 'rb') as f:
            grids = pickle.load(f)
    
    if reddening_grids is None:
        with open(os.path.expanduser('~') + '/sluggs/sps_models/fsps_reddening_mist_inter_mags.pickle', 'rb') as f:
            reddening_grids = pickle.load(f)

    mags = get_mags(magnitudes, reddening_grids, grids)
            
    metal_ivar2 = metal_e**-2. / 2
    A_V_ivar2 = A_V_e**-2. / 2
    if A_V2_e:
        A_V2_ivars2 = A_V2_e**-2. / 2
        logl = prob_bi
        loglargs = (mags, metal, metal_ivar2, A_V, A_V_ivar2, A_V2, A_V2_ivars2)
    else:
        logl = prob
        loglargs = (mags, metal, metal_ivar2, A_V, A_V_ivar2)            
     
    metal_guess, age_guess, mass_guess, A_V_guess = grid_search(mags, logl, loglargs, age_lower, age_upper, metal_lower,
        metal_upper, A_V_lower, A_V_upper, logger)

    def start_array(guess, nwalkers, scatter, lower, upper):
        start = guess + scatter * np.random.randn(nwalkers)
        start[start < lower] = lower
        start[start > upper] = upper
        return start 

    start = [start_array(metal_guess, nwalkers, 0.1, metal_lower, metal_upper),
             start_array(age_guess, nwalkers, age_guess * 0.1, age_lower, age_upper),
             start_array(mass_guess, nwalkers, 0.05, -np.inf, np.inf),
             start_array(A_V_guess, nwalkers, 0.05, A_V_lower, A_V_upper)]
    start = np.asarray(start).T
        
    log_likely = logl(np.array([metal_guess, age_guess, mass_guess, A_V_guess]), *loglargs)
    start_str = 'Starting at:\n{:.3f} {:.3f} {:.3f} {:.3f}\n'.format(metal_guess,
                    age_guess, mass_guess, A_V_guess)
    start_str += 'Starting log likelihood {:.3f}\n'.format(log_likely)
    
    if sampler == 'pt':
        start_str += 'Using {} walkers and {} tempratures for {}+{} steps'.format(nwalkers, ntemps, nburn, steps)
        logger.info(start_str)
        sampler = ptemcee.Sampler(nwalkers, start.shape[-1], logl, prior,
                                  loglargs=loglargs,
                                  logpargs=(metal_lower, metal_upper, age_lower, age_upper, A_V_lower, A_V_upper),
                                  threads=threads, ntemps=ntemps)
        temp_start = []
        for i in range(ntemps):
            temp_start.append(start)
        temp_start = np.array(temp_start)    
        sampler.run_mcmc(temp_start, (nburn + steps))
        


        s = sampler.chain[0]
        import matplotlib.pyplot as plt        
        fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
        labels = ['[Fe/H]', 'age', 'mass', 'A_V']
        for j in range(s.shape[2]):
            print(np.mean([ptemcee.util.autocorr_integrated_time(s[i,:,j]) for i in range(s.shape[0])]))
        
            ax = axes[j]
            for i in range(s.shape[0])[:1]:
                ax.plot(s[i, :, j], "k", alpha=0.3)
#             ax.set_xlim(0, len(s))
            ax.set_ylabel(labels[j])
            ax.yaxis.set_label_coords(-0.1, 0.5)            
           
            
        samples = sampler.chain[0, :, nburn:, :].reshape((-1, start.shape[-1]))
        samples = samples[::thin]
 
        if threads > 1:
            sampler.pool.close()
           
    else:
        start_str += 'Using {} walkers for {}+{} steps'.format(nwalkers, nburn, steps)
        logger.info(start_str)
        sampler = emcee.EnsembleSampler(nwalkers, start.shape[-1], em_prop, threads=threads, 
                                        args=(mags, metal, metal_ivar2, A_V, A_V_ivar2, metal_lower, metal_upper, age_lower, age_upper, A_V_lower, A_V_upper),
                                        moves=[(emcee.moves.DEMove(), 0.8),
                                               (emcee.moves.DESnookerMove(), 0.2)])
        sampler.run_mcmc(start, (nburn + steps))
        print(sampler.get_autocorr_time())
        
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
        samples = sampler.get_chain()
        labels = ['[Fe/H]', 'age', 'mass', 'A_V']
        for i in range(4):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number");
        
        samples = sampler.sampler.get_chain(discard=nburn, thin=thin, flat=True)

    if keep_chain:
        np.save(open(str(keep_chain) + '_chain.npy', 'w'), np.asarray(sampler.chain))

    norm_percentiles = stats.norm.cdf([-2, -1, 0, 1, 2]) * 100
    Z_precentiles = np.percentile(samples[:,0], norm_percentiles)
    age_precentiles = np.percentile(samples[:,1], norm_percentiles)
    mass_precentiles = np.percentile(samples[:,2], norm_percentiles)
    A_V_precentiles = np.percentile(samples[:,3], norm_percentiles)
    
    output_str = 'Output:\n' 
    output_str += '[Z/H] ' + ' '.join(['{:.3f}'.format(Z) for Z in Z_precentiles]) + '\n'
    output_str += '       {:.3f} \u00B1{:.3f}'.format(np.mean(samples[:,0]), np.std(samples[:,0])) + '\n'
    output_str += 'age   ' + ' '.join(['{:.3f}'.format(age) for age in age_precentiles]) + '\n'
    output_str += '       {:.3f} \u00B1{:.3f}'.format(np.mean(samples[:,1]), np.std(samples[:,1])) + '\n'
    output_str += 'mass  ' + ' '.join(['{:.3f}'.format(mass) for mass in mass_precentiles]) + '\n'
    output_str += '       {:.3f} \u00B1{:.3f}'.format(np.mean(samples[:,2]), np.std(samples[:,2])) + '\n'
    output_str += 'A_V   ' + ' '.join(['{:.3f}'.format(red) for red in A_V_precentiles]) + '\n'
    output_str += '       {:.3f} \u00B1{:.3f}'.format(np.mean(samples[:,3]), np.std(samples[:,3])) + '\n'
    log_likely = logl(np.array([Z_precentiles[2], age_precentiles[2], mass_precentiles[2], A_V_precentiles[2]]), *loglargs)
    output_str += 'Log likelihood: {:.3f}\n'.format(log_likely)
    logger.info(output_str) 
        
    if plot:
        import corner
        corner.corner(samples, labels=['[Z/H]', 'Age', 'Log Mass', 'A_V'], quantiles=[0.16, 0.50, 0.84], show_titles=True)

    return samples, Z_precentiles[1:-1], age_precentiles[1:-1], mass_precentiles[1:-1], A_V_precentiles[1:-1]


def get_mags(magnitudes, reddening_grids, grids):
    mags = []
    for name, mag, mag_e in magnitudes:
        mags.append((mag, mag_e**-2. / 2, reddening_grids[name], grids[name]))
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

def find_a_mass(mags, metal_guess, age_guess):
    observed = []
    model = []
    for mag, mag_ivar2, reddening_grid, grid in mags:
        observed.append((mag - grid(metal_guess, age_guess)[0][0]) * mag_ivar2**0.5 / 2)
        model.append([-2.5 * mag_ivar2**0.5 / 2, reddening_grid(metal_guess, age_guess)[0][0] * mag_ivar2**0.5 / 2])
    observed = np.array(observed)
    model = np.array(model)
    mass, a_v = np.linalg.lstsq(model, observed, rcond=None)[0]
    return mass, a_v


def grid_search(mags, logl, loglargs, age_lower,
                age_upper, metal_lower, metal_upper, A_V_lower, A_V_upper,
                logger):
    best_likely = -np.inf
    best_guess = None
    
    metal_range = np.arange(-2.7, 0.35, 0.1)
    metal_range = metal_range[(metal_range >= metal_lower) & (metal_range <= metal_upper)]
    age_range = 10**np.arange(-2.9, 1.16, 0.05)
    age_range = age_range[(age_range >= age_lower) & (age_range <= age_upper)]
        
    for metal_guess in metal_range:
        for age_guess in age_range:    
            mass_guess, A_V_guess = find_a_mass(mags, metal_guess, age_guess)
            ln_likely = logl((metal_guess, age_guess, mass_guess, A_V_guess), *loglargs)
            if ln_likely > best_likely:
                best_guess = (metal_guess, age_guess, mass_guess, A_V_guess)
                best_likely = ln_likely
                logger.debug('{} {}'.format(best_guess, best_likely))
                
    return best_guess
    


