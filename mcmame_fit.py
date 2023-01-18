#! /usr/bin/env python

import argparse
import glob
import logging
import multiprocessing
import os
import pickle

import numpy as np

from astropy import table
from astropy.io import fits

import mcmame_lib


def calc_age_metal(arguments):
    entry, grids, reddening_grids, priors = arguments
    
    logger = logging.getLogger(entry['output'])

    magnitudes = []
    for name in entry.dtype.names:
        if '_e' == name[-2:]:
             continue
        if name in grids.keys():
            mag = entry[name]
            mag_e = entry[name + '_e']
            if (hasattr(mag, 'mask') and mag.mask) or (hasattr(mag_e, 'mask') and mag_e.mask) or ~np.isfinite(mag) or not (mag_e > 0):
                continue
            magnitudes += [[name, entry[name], entry[name + '_e']]]

            
    if len(magnitudes) <= 1:
        logger.warning('Need at least two bands for {}'.format(entry['ident']))
        return None
    
    if 'Z_H' in entry.dtype.names and 'Z_H_e' in entry.dtype.names:
        Z_H = entry['Z_H']
        Z_H_e = entry['Z_H_e']
    else:
        Z_H = None
        Z_H_e = None
    
    if 'A_V' in entry.dtype.names and 'A_V_e' in entry.dtype.names:
        A_V = entry['A_V']
        A_V_e = entry['A_V_e']
    else:
        A_V = 0
        A_V_e = 0.01
        
    if 'A_V2' in entry.dtype.names and 'A_V2_e' in entry.dtype.names:
        A_V2 = entry['A_V2']
        A_V2_e = entry['A_V2_e']
    else:
        A_V2 = 0
        A_V2_e = 0

    samples, Z_limits, age_limits, mass_limits, A_V_limits = mcmame_lib.calc_age_mass(magnitudes, Z_H, Z_H_e, A_V, A_V_e, grids=grids, reddening_grids=reddening_grids, threads=1, logger=logger, A_V2=A_V2, A_V2_e=A_V2_e, **priors)

    entry['Z'] = Z_limits[1]
    entry['Z_lower'] = Z_limits[1] - Z_limits[0]
    entry['Z_upper'] = Z_limits[2] - Z_limits[1]    
    entry['age'] = age_limits[1]
    entry['age_lower'] = age_limits[1] - age_limits[0]
    entry['age_upper'] = age_limits[2] - age_limits[1]
    entry['mass'] = mass_limits[1]
    entry['mass_lower'] = mass_limits[1] - mass_limits[0]
    entry['mass_upper'] = mass_limits[2] - mass_limits[1]
    entry['a_v'] = A_V_limits[1]
    entry['a_v_lower'] = A_V_limits[1] - A_V_limits[0]
    entry['a_v_upper'] = A_V_limits[2] - A_V_limits[1]
    
    new_samples = np.empty(samples.shape[0], dtype=[('Z', 'f'),
                            ('age', 'f'), ('mass', 'f'), ('A_V', 'f')])
    new_samples['Z'] = samples[:,0]
    new_samples['age'] = samples[:,1]
    new_samples['mass'] = samples[:,2]
    new_samples['A_V'] = samples[:,3]

    primary_hdu = fits.PrimaryHDU()
    entry_hdu = fits.table_to_hdu(table.Table(entry))
    samples_hdu = fits.BinTableHDU(new_samples)

    hdul = fits.HDUList([primary_hdu, entry_hdu, samples_hdu])
    hdul.writeto(entry['output'] + '.fits')

    logger.info('sampled')
    return entry


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input file')
    parser.add_argument('--grid', help='Model grid')
    parser.add_argument('--red-grid', help='Reddening grid')    
    parser.add_argument('--output', help='output_name')
    parser.add_argument('-N', type=int, help='Number of parallel processes')
    parser.add_argument('--age-lower', type=float, default=0.001, help='Lower limit of age prior')
    parser.add_argument('--age-upper', type=float, default=15.0, help='Upper limit of age prior')
    parser.add_argument('--metal-lower', type=float, default=-3., help='Lower limit of [Z/H] prior')
    parser.add_argument('--metal-upper', type=float, default=0.5, help='Upper limit of [Z/H] prior')
    parser.add_argument('--A_V-lower', type=float, default=0., help='Lower limit of A_V prior')
    parser.add_argument('--A_V-upper', type=float, default=np.inf, help='Upper limit of A_V prior')      
    
    os.nice(10)
    
    args = parser.parse_args()
    
    LOG_FORMAT = "[%(asctime)s] %(levelname)8s %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    if args.N:
        cpus = args.N
    else:
        cpus = max(multiprocessing.cpu_count() // 2, 1)    
    logging.info('Using {} cores'.format(cpus))

    filename = os.path.splitext(args.input)[0]
    if args.output:
        filename = args.output
        
    if args.grid is None:
        args.grid = os.path.expanduser('~') + '/sluggs/sps_models/fsps_mist_inter_mags.pickle'
    if args.red_grid is None:
        args.red_grid = os.path.expanduser('~') + '/sluggs/sps_models/fsps_reddening_mist_inter_mags.pickle'        

    with open(args.grid, 'rb') as f:
        grids = pickle.load(f)
    with open(args.red_grid, 'rb') as f:
        reddening_grids = pickle.load(f)        

    logging.info('Using grid: ' + args.grid.split('/')[-1])
    logging.info('Using reddening grid: ' + args.red_grid.split('/')[-1])
    
    catalogue = table.Table.read(args.input)
    catalogue['ident'] = [str(ident) for ident in catalogue['ident']]
    catalogue.add_column(table.Column(np.array(catalogue['ident'], dtype='a80'), name='output'))
    catalogue['Z'] = 0.
    catalogue['Z_lower'] = 0.
    catalogue['Z_upper'] = 0.
    catalogue['age'] = 0.
    catalogue['age_lower'] = 0.
    catalogue['age_upper'] = 0.   
    catalogue['mass'] = 0.
    catalogue['mass_lower'] = 0.
    catalogue['mass_upper'] = 0. 
    catalogue['a_v'] = 0.
    catalogue['a_v_lower'] = 0.
    catalogue['a_v_upper'] = 0.     
    
    
    existing_outputs = glob.glob(filename + '_*.fits') + glob.glob(filename + '_*.pickle')
    priors = {'age_lower':args.age_lower,
              'age_upper':args.age_upper,
              'metal_lower':args.metal_lower,
              'metal_upper':args.metal_upper,
              'A_V_lower':args.A_V_lower,
              'A_V_upper':args.A_V_upper}
    
    inputs = []
    skipped = []
    for entry in catalogue:
        output = filename + '_' + entry['ident']
        entry['output'] = output 
        
        if output + '.fits' not in existing_outputs:

            if output + '.pickle' in existing_outputs:
                with open(output + '.pickle', 'rb') as f:
                    row, samples = pickle.load(f, encoding='latin1')
    
                row['output'] = output
                row_table = table.Table(row)
                new_samples = np.empty(samples.shape[0], dtype=[('Z', 'f'),
                                 ('age', 'f'), ('mass', 'f'), ('A_V', 'f')])
                new_samples['Z'] = samples[:,0]
                new_samples['age'] = samples[:,1]
                new_samples['mass'] = samples[:,2]
                new_samples['a_v'] = samples[:,3]
    
                primary_hdu = fits.PrimaryHDU()
                entry_hdu = fits.table_to_hdu(row_table)
                samples_hdu = fits.BinTableHDU(new_samples)
                hdul = fits.HDUList([primary_hdu, entry_hdu, samples_hdu])
                hdul.writeto(output + '.fits')
    
                skipped.append(row_table[0])
                logging.info('Converting ' + entry['ident'])

            else:
                inputs.append([entry, grids, reddening_grids, priors])
                logging.info('Sampling ' + entry['ident'])
            
        else:
            with fits.open(output + '.fits') as hdul:
                row_table = table.Table.read(hdul[1])
            skipped.append(row_table[0])
            logging.info('Skipping ' + entry['ident'])
        

    pool = multiprocessing.Pool(processes=cpus)
    results = pool.map(calc_age_metal, inputs)
    pool.close()
    
    output_catalogue = catalogue[:0].copy()
 
    for entry in results:
        if entry is None:
            continue        
        output_catalogue.add_row(entry)

    logging.info('{} objects fitted'.format(len(output_catalogue)))

    for entry in skipped:
        
        if len(entry) == len(output_catalogue.colnames):
            output_catalogue.add_row(entry)
        else:
            input_row = catalogue[catalogue['ident'] == entry['ident']]
            new_row = []
            for field in output_catalogue.colnames:
                if field in entry.colnames:
                    new_row.append(entry[field])
                else:
                    new_row.append(input_row[field])
            output_catalogue.add_row(new_row)
                    
    logging.info('{} existing fits'.format(len(skipped)))
    output_catalogue.write(filename + '_output.ecsv', overwrite=True)



