### -- additional functions

## -- IMPORTS
import glob
import os
import sys

from astropy.io import ascii
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

## -- FUNCTIONS

def plot_light_curve(data_table, name):
    """ Plots the light curve.

    Parameters
    ----------
    data_dir : str
        The path to the data.
    name : str  
        The output plot name.
    """

    data_dict = dict(ascii.read(data_table))

    data_tups = []
    for key in data_dict:
        if 'sum' in key:
            filename = key.split('_sum')[0] + '_ima.fits'
            mjd = fits.getval(filename, 'expstart', ext=0)
            file_sum = np.nansum(data_dict[key])
            y_err = np.sqrt(file_sum)
            data_tups.append((mjd, file_sum, y_err))

    plt.errorbar([tup[0] for tup in data_tups], [tup[1] for tup in data_tups],
            yerr=[tup[2] for tup in data_tups], fmt='.', color='black',
            alpha=.5)
    plt.xlabel('MJD')
    plt.ylabel('Sum')
    plt.savefig(name)
    plt.clf()
            

def plot_cr_comparisons(data_path):
    """ Plots CR flagging comparisons for the directory
    at hand.

    Parameters
    ----------
    data_path : str
        Path to the data.
    """

    if not os.path.exists(os.path.join(data_path, 'cr_flagging')):
        os.makedirs(os.path.join(data_path, 'cr_flagging'))

    files = glob.glob(os.path.join(data_path, '*crcorr*'))

    for cr_file in files:
        root = cr_file.split('_crcorr')[0].split('/')[-1]
        new_file = os.path.join(data_path, 'cr_flagging/{}_crompare.png'.format(root))
        pre_flag = ''.join(cr_file.split('_crcorr2'))
        diff = fits.getdata(pre_flag, ext=1) - fits.getdata(cr_file, ext=1)
        plt.imshow(diff)
        plt.title(root)
        plt.tight_layout()
        plt.savefig(new_file)
        plt.clf()
        print('CR comparison plot written to : {}.'.format(new_file))


def create_binned_light_curve(data_table, bin_range):
    """Match a wavelength bin to the nearest pixel and sum
    over that bin.

    Parameters
    ----------
    data_table : str
        The path to the data.
    bin_range : tuple 
        Tuple with (start, stop) of the wavelength range.

    Returns
    -------
    mid_bin : float
        The middle of the bin.
    bin_flux : np.array
        A sum of the flux through each bin.
    mjd : np.array
        The mjd for each point.
    """
    mjd, bin_flux = [], []
    data = dict(ascii.read(data_table))
    keys = list(set([key.split('_')[-2] for key in data.keys()]))
    for key in keys:
        filename = '{}_ima.fits'.format(key)
        mjd.append(fits.getval(filename, 'expstart', ext=0))
        
        wv = data['{}_wv'.format(key)]
        flux = data['{}_sum'.format(key)]
        min_wv = min(wv, key=lambda x:abs(x-bin_range[0]))
        max_wv = min(wv, key=lambda x:abs(x-bin_range[1]))
        mindex = np.where(wv == min_wv)[0][0]
        maxdex = np.where(wv == max_wv)[0][0]
        bin_flux.append(np.sum(flux[mindex:maxdex+1]))
    
    mid_bin = np.mean([bin_range[0], bin_range[1]])
    
    return mid_bin, bin_flux, mjd, (mindex, maxdex)


def plot_spectrum_overlay(data_table, name):
    """ Plots the overlaid spectrum.

    Parameters
    ----------
    data_table : str
        The name of the data table.
    name : str  
        What to call the plot.
    """

    data = ascii.read(data_table)
    wvs = sorted([key for key in data.keys() if 'wv' in key])
    sums = sorted([key for key in data.keys() if 'sum' in key])
    for wv_key, sum_key in zip(wvs, sums):
        plt.plot(data[wv_key], data[sum_key])
    plt.savefig('{}.png'.format(name))
    plt.clf()

    
def drop_first_of_orbit(data_table, n_orbits, remove=False):
    """ Drops the first exposure in each orbit.

    Parameters
    ----------
    data_table : str
        The data file to read in.
    n_orbits : str
        The number of orbits in the transit.
    remove : int, optional
        If set, removes the nth orbit. 
    Returns
    -------
    dropped_mjd : list
        An array of the mjd without the first of the orbit.
    dropped_flux : list
        An array of the flux without the first of the orbit.
    """

    data = ascii.read(data_table)
    mjd, flux = data['mjd'], data['flux']

    # Sort by mjd
    sorted_tups = sorted(list(zip(mjd, flux)), key=lambda x: x[0])
    mjd, flux = [tup[0] for tup in sorted_tups], [tup[1] for tup in sorted_tups]
    
    diffs = np.diff(mjd)
    big_diffs = sorted(diffs)[-(n_orbits-1):]
    indeces = []
    for diff in big_diffs:
        indeces += [index for index in np.where(diff == diffs)[0]]
    indeces = sorted(list(set(indeces)))

    dropped_mjd = []
    dropped_flux = []
    init_index = 1
    for count, index in enumerate(indeces):
        index += 1
        if count+1 != remove:
            dropped_mjd += mjd[init_index:index]
            dropped_flux += flux[init_index:index]
        init_index = index+1
    dropped_mjd +=  mjd[init_index:]
    dropped_flux += flux[init_index:]
    
    return dropped_mjd, dropped_flux

    


if __name__ == "__main__":

    output_data= sys.argv[1]
    name = 'test.png'
    plot_light_curve(output_data, name)
