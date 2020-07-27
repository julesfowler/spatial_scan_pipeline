"""Fits a light curve...

Authors
------- 
    Jules Fowler, Giovanni Bruno, 2018

"""

## -- IMPORTS
import json
import os
import glob
import sys 

from astropy.io import ascii, fits
import corner
import mpld3
from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
from scipy.optimize import minimize

import emcee
from pdb import set_trace
import batman


## -- FUNCTIONS

# Plotting set up 
rc('font', **{'family': 'serif', 'serif':['Computer Modern Roman'],'size':14})
rc('text', usetex=True)

LBLUE = '#88CCEE'
DBLUE = '#332288'
LGREEN = '#44AA99'
DGREEN = '#117733'
YGREEN = '#999933'
TAN = '#DDCC77'
PANK = '#CC6677'
MAROON = '#882255'
LPURPLE = '#AA4499'

def main(white_light, binned_curves, ignore, coeffs_dict, planet_parameters): 
    """Main function to fit the transit.

    Parameters
    ----------
    white_light : str
        Path to the white light curve.
    binned_curves : str
        Path to the binned curves.
    ignore : float
        Point to ignore data before.
    coeffs_dict : dict
        Initial parameters and boundaries.
    planet_parameters : dict
        Static parameters for the planet. 
    """
    
    white_light = glob.glob(white_light)[0]
    binned_curves = glob.glob(binned_curves)
    print('Running on white light curve : {}.'.format(white_light))
    print('Running on binned curves : {}.'.format(','.join(binned_curves)))
    print('\n')

    #### --  Work with white light first
    print('Running on initial white light curve.')
    print('\n')
    data = ascii.read(white_light)
    mjd, flux = np.array(data['mjd']), np.array(data['flux'])
    
    # Select limb darkening coeffs
    ld = select_limb_darkening_coeffs(np.mean([1.1e4, 1.7e4]), dict_key='0b')
    coeffs_dict['ld'] = {'init' : ld}
    
    # Normalize the data and make it fit-ready
    t, y, y_err = normalize_data(mjd, flux, ignore)
    
    # Fit the transit with emcee
    non_fixed_keys = ['kr', 't0', 'r0', 'r1', 'r2', 'r3', 'r4', 'c', 'shift']
    print('Fitting transit with optimize and MCMC.')
    print('\n')
    coeffs_solution, emcee_samples, chains, percentiles = fit_transit(non_fixed_keys, t, y, y_err, coeffs_dict, planet_parameters)
    mega_dict = {}
    transit_compare(coeffs_solution, t, y, y_err, 'white_light', mega_dict)


    # Corner plot
    titles = [r'$k_r$', r'$t_0$', r'$r_0$', r'$r_1$', r'$r_2$', r'$r_3$', r'$r_4$', r'$C$']
    rc('text', usetex=False)
    #cornerplot(emcee_samples, titles, None, 'white_light')
    plt.clf()
    rc('text', usetex=True)
    # Replace paramters in the original coeffs dict with our solution
    for key in coeffs_solution:
        coeffs_dict[key]['init'] = coeffs_solution[key]
    coeffs_solution['ld'] = coeffs_dict['ld']['init']
    coeffs_solution['shift'] = coeffs_dict['shift']['init']
    
    check_initial_solution(t, y, y_err, coeffs_solution, planet_parameters, percentiles, 'white_light')
    
    # Divide out the systematics 
    corrected_y, residuals, delta = correct_spectra(t, y, y_err, coeffs_solution, planet_parameters, 'white_light')
    print('Data corrected.')
    print('\n')

    #### -- Now for some binned curves
    curve_dict = {'wv_bin': [], 'transit_depth': [], 'upper_err': [], 'lower_err': []}
    for curve in binned_curves:
        curve_key = curve.split('_corrected')[0].split('/')[-1]
        print('Running on wavelength bin : {}.'.format(curve_key))
        print('\n')
        
        # Read in binned curve
        data = ascii.read(curve)
        mjd, flux = data['mjd'], data['flux']
        
        # Select Ld coeffs and normalize data
        coeffs_dict['ld'] = {'init': select_limb_darkening_coeffs(curve_key)}
        t, y, y_err = normalize_data(mjd, flux, ignore)
        
        # Run the MCMC on more limited set of parameters
        non_fixed_keys = ['kr', 'r0', 'r1', 'r2', 'r3', 'r4', 'c']
        print('Fitting transit with optimize and MCMC.')
        print('\n')
        
        coeffs_solution, emcee_samples, chains, percentiles = fit_transit(non_fixed_keys, t, y, y_err, coeffs_dict, planet_parameters)
        transit_compare(coeffs_solution, t, y, y_err, curve_key, mega_dict)
        rc('text', usetex=False)
        cornerplot(emcee_samples, titles, None, curve_key)
        plt.clf()
        rc('text', usetex=True)
        coeffs_solution['ld'] = coeffs_dict['ld']['init']
        coeffs_solution['shift'] = coeffs_dict['shift']['init']
        coeffs_solution['t0'] = coeffs_dict['t0']['init']
        check_initial_solution(t, y, y_err, coeffs_solution, planet_parameters, percentiles, curve_key)
        
        # Correct the transit
        corrected_y, residuals, delta = correct_spectra(t, y, y_err, coeffs_solution, planet_parameters, curve_key)
        print('Data corrected.')
        print('\n')

        # Calculate the final tranismission spectrum
        transit_depth, upper_err, lower_err = calculate_transmission(curve_key, percentiles)

        curve_dict['wv_bin'].append(float(curve_key))
        curve_dict['transit_depth'].append(transit_depth)
        curve_dict['upper_err'].append(upper_err)
        curve_dict['lower_err'].append(lower_err)


    # Plot and save final transimission spectrum
    ascii.write(curve_dict, 'transmission_spectrum.csv')
    print('Final transimission spectrum saved to transmission_spectrum.csv')
    print('\n')
    plot_transmission_spectrum(curve_dict)
    

def batman_transit(coeffs_init, t, planet_parameters):
    """Creates a transit model with batman.

    Parameters
    ----------
    coeffs_init : dict
        Dictionary of initial guesses of coefficients.
    t : np.array    
        Array of time values over which the transit goes.
    planet_parameters : dict
        Dictionary of planet parameters.
    
    Returns
    -------
    model_flux : np.array
        The flux from the model matched to the input time array.
    """

    params = batman.TransitParams()
    params.per = planet_parameters['period']
    params.a = planet_parameters['a_r_star']
    params.inc = planet_parameters['inclination']
    params.ecc = planet_parameters['eccentricity']
    params.w = planet_parameters['omega']
    params.u = coeffs_init['ld']
    params.rp, params.t0 = coeffs_init['kr'], coeffs_init['t0']
    params.limb_dark = 'quadratic'

    model = batman.TransitModel(params, t)
    model_flux = model.light_curve(params)

    return model_flux 


def build_transit_model(coeffs_init, t, planet_parameters):
    """ Build the full model. 
    
    Parameters
    ----------
    coeffs_init : dict
        Initial guess at coefficients.
    t : np.array
        The time values over which the transit goes.
    planet_parameters : dict
        A dictionary of known parameters of the planet.

    Returns
    -------
    model_out : np.array
        The intial model to feed in. 
    """
    
    # Build base transit + ramp model
    model = batman_transit(coeffs_init, t, planet_parameters) * \
            ramp(coeffs_init, t, planet_parameters)
    
    return model


def chi_squared(model, y, y_err):
    """ Return the chi**2 value of a given model.

    Parameters
    ----------
    model : np.array
        Array of flux from model.
    y : np.array
        Actual flux array.
    y_err np.array
        Error on the flux.

    Returns
    -------
    chi2 : float
        The chi**2 value.
    """

    chi2 = np.sum(((model - y)**2)/(y_err**2))
    return chi2


def calculate_transmission(curve_key, percentiles):
    """ Calculates the final transmission spectrum for a given bin.

    Parameters
    ----------
    curve_key : str of float
        The middle of the bin.
    percentiles : dict
        A dictionary of Bayesian percentiles for each parameter.
    
    Returns
    -------
    depth : float   
        The transmission depth.
    upper_err : float
        The upper error bound.
    lower_err : float
        The lower error bound.
    """

    kr_lower, kr_mid, kr_upper = percentiles['kr']
    depth = kr_mid**2*1e6
    lower_err = 2*kr_mid*(kr_mid - kr_lower)*1e6
    upper_err = 2*kr_mid*(kr_upper - kr_mid)*1e6

    return depth, upper_err, lower_err


def check_initial_solution(t, y, y_err, coeffs, planet_parameters, percentiles, name):
    """ Plots the inital solution for inspection purposes and prints some
    useful stuff.
    
    Parameters
    ----------
    t : np.array
        The time array.
    y : np.array
        The flux array.
    y_err : np.arary
        The error on the flux array.
    coeffs : dict
        A dictionary of coefficients.
    planet_parameters : dict
        A dictionary of static parameters about the planet.
    name : str
        Naming convention for the plot out.
    """
    
    model_range = np.linspace(np.min(t), np.max(t), 200)
    model = build_transit_model(coeffs, model_range, planet_parameters)
    transit_model = batman_transit(coeffs, model_range, planet_parameters)
    plt.clf() 
    rc('figure', figsize=[6.4, 4.8])
    plt.plot(model_range, model, color=DGREEN, label='Full Model')
    plt.plot(model_range, transit_model, color=DBLUE, label='Transit Model')
    plt.scatter(t, y, color='black', s=8, alpha=.5, label='data')

    plt.xlabel('Normalized MJD')
    plt.ylabel('Normalized Transit Flux')
    plt.ylim(np.min(y)-.001, np.max(y)+.001)
    plt.legend()
    plt.savefig('inital_check_{}.png'.format(name))
    plt.clf()

    chi_model = build_transit_model(coeffs, t, planet_parameters)
    chi2 = chi_squared(chi_model, y, y_err)
    percentile_err = percentiles['kr'][2]**2 - percentiles['kr'][1]**2
    print('Estimated transit depth from best fit : {}'.format(coeffs['kr']**2*1e6))
    print('Estimated transit depth from 50 % : {}'.format(percentiles['kr'][1]**2*1e6))
    print('Bayesian uncertainty : {}'.format((percentile_err)*1e6))
    print('Reduced Chi Squared : {}'.format(chi2/(len(t) - len(percentiles.keys()))))
    print('\n')


def correct_spectra(t, y, y_err, coeffs, planet_parameters, name):
    """ Corrects the data, outputs residuals, and makes a plot of the corrected
    data. 

    Parameters
    ----------
    t : np.array
        Array of time data.
    y : np.array
        Array of flux data.
    y_err : np.array
        Error in the flux data.
    coeffs : dict
        Dictionary of the fit parameters for the transit.
    planet_parameters : dict
        Dictionary of planet-specific parameters.
    name : str
        Naming convention for the figure.

    Returns
    -------
    corrected_y : np.array
        Array of correct flux data.
    residuals : np.array
        The correction applied to the flux.
    delta : np.array
        The affect on the error data.
    """

    transit_model = batman_transit(coeffs, t, planet_parameters)
    residuals = y/transit_model
    delta = y_err/transit_model
    corrected_y = y/residuals
    
    mod_range = np.linspace(np.min(t), np.max(t), 200)
    transit_mod = batman_transit(coeffs, mod_range, planet_parameters)
    
    rc('figure', figsize=[6.4, 4.8])
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    
    ax1.plot(mod_range, transit_mod, color=DBLUE, label='Transit Model')
    ax1.scatter(t, y, color=DGREEN, alpha=.3, s=8, label='Data')
    ax1.scatter(t, corrected_y, color=DBLUE, alpha=.5, s=9, label='Corrected Data')
    ax1.set_ylabel('Transit')

    ax2.errorbar(t, residuals, delta, color=DGREEN, fmt='.')
    ax2.set_ylabel('Residuals')
    ax2.set_xlabel('Normalized MJD')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('corrected_transit_{}.png'.format(name))
    plt.clf()

    return corrected_y, residuals, delta


def fit_transit(non_fixed_keys, t, y, y_err, coeffs_dict, planet_parameters):
    """ Fit the transit with MCMC.
    Parameters
    ----------
    non_fixed_keys : list of str
        List of parameters that will be optimized.
    t : np.array
        Array of time.
    y : np.array
        Array of flux.
    y_err : np.array
        Array of error on flux.
    coeffs_dict : dict
        Initial guesses and bounds for parameters.
    planet_paramters : dict
        Dictionary of static parameters about the planet. 

    Returns
    -------

    """

    # Set the initial parameters
    coeffs_init = {}
    for key in coeffs_dict:
        coeffs_init[key] = coeffs_dict[key]['init']
    # Run a minimization
    # Create the alias form of the model
    
    non_fixed_init = [coeffs_dict[key]['init'] for key in non_fixed_keys]
    non_fixed_bounds = [coeffs_dict[key]['bounds'] for key in non_fixed_keys]
    
    neg_likelihood = lambda *args: -likelihood_func(*args)
    min_solution = minimize(neg_likelihood, non_fixed_init, jac=False, method='L-BFGS-B', 
            args=(non_fixed_keys, coeffs_init, t, y, y_err, planet_parameters), options={'maxiter':1000}, bounds=non_fixed_bounds)
    print(non_fixed_keys)
    print(min_solution.x)
    # Run an MCMC
    model_init = min_solution.x
    ndim, nwalkers = len(model_init), 100
    sampler = emcee.EnsembleSampler(nwalkers, ndim, likelihood_prob,
        args=(non_fixed_keys, coeffs_init, t, y, y_err, planet_parameters),
            threads=8, live_dangerously=False)
    
    # Iteration 1
    nsteps, width = 1000, 30
    perturbed_start = model_init + 1e-2*np.random.randn(nwalkers, ndim)
    next_start = list(sampler.sample(perturbed_start, iterations=nsteps))[-1][0]
    sampler.reset()
    print('Iteration 1 complete!')

    # Iteration 2
    nsteps, width= 2000, 30
    next_iteration = list(sampler.sample(next_start, iterations=nsteps, thin=10))
    print('Iteration 2 complete!')
    print('\n')
    
    emcee_samples = sampler.flatchain
    likelihood_probability = sampler.flatlnprobability
    best_solution = emcee_samples[likelihood_probability.argmax()]
    percentiles = {}
    for index, key in enumerate(non_fixed_keys):
        percentiles[key] = np.percentile(emcee_samples[:, index],[15.9,  50, 84.1])

    chains = {}
    chains['max ML'] = min_solution.x
    chains['chains'] = emcee_samples
    chains['mean_frac'] = np.mean(sampler.acceptance_fraction)
    #chains['autocorr_time'] = emcee.autocorr.integrated_time(emcee_samples, c=10)
    chains['probability'] = likelihood_probability
    
    coeffs_solution = {}
    for index, key in enumerate(non_fixed_keys):
        coeffs_solution[key] = best_solution[index]
    return coeffs_solution, emcee_samples, chains, percentiles


def likelihood_func(params, key, coeffs_init, t, y, y_err, planet_parameters):
    """ Calculates the likelihood for a model.
    
    Parameters
    ----------
    params : list
        List of parameters to change in coeffs_init.
    key : list  
        The matching keys. 
    coeffs_init : dict
        A dictionary of coefficients and values. 
    t : np.array
        Time over which the transit runs.
    y : np.array
        The actual flux of the light curve.
    y_err : np.array
        The associated error.
    planet_parameters : dict
        Parameters about the planet and transit. 
        
    Returns
    -------
    likelihood : float  
        The likelihood of the model solution.
    """
    
    # Replace coefficients
    if len(params) > 0:
        for index, coeff_key in enumerate(key):
            coeffs_init[coeff_key] = params[index]
    
    # Build transit model
    transit_model = build_transit_model(coeffs_init, t, planet_parameters)
    # Build sigma and chi**2
    sigma = np.mean(y_err)
    chi2 = chi_squared(transit_model, y, y_err)
    likelihood = -len(y)*np.log(sigma) - 0.5*len(y)*np.log(2*np.pi) - .5*chi2
    
    return likelihood


def likelihood_prob(params, key, coeffs_init, t, y, y_err, planet_parameters):

    """The likelihood probability for the MCMC.

    Parameters
    ----------
    params : np.array
        An array of params for emcee or 'default' to pass to coeffs.
    key : np.array or False
        The key for the parameters.
    coeffs_init : dict
        A dictionary of the coefficients. 
    t : np.array
        The time over which the transit runs.
    y : np.array
        The flux.
    y_err : np.array    
        Error in the flux data.

    Returns
    -------
    np.array or inf
        The likelihood function or an infinite value.
    """
    if len(params) > 0:
        for index, coeff_key in enumerate(key):
            coeffs_init[coeff_key] = params[index]

    kr, t0 = coeffs_init['kr'], coeffs_init['t0']
    kr_init = -1*(np.log(1e-2) + 0.5*np.log(2*np.pi) + (kr - 0.09)**2/(2*1e-2)**2)
    t0_init = -1*(np.log(3e-2) + 0.5*np.log(2*np.pi) + (t0 - 0.17)**2/(2*3e-2)**2)
    l_init = kr_init + t0_init
    
    if not np.isfinite(l_init):
        return -np.inf
    else:
        return l_init + likelihood_func(params, key, coeffs_init, t, y, y_err, planet_parameters)


def normalize_data(mjd, flux, ignore):
    """ Take mjd and flux from the data table and returns
    it normalized, ordered, and with an error array.

    Parameters
    ----------
    mjd : np.array
        Array of MJD.
    flux : np.array
        Array of flux.
    ignore : float
        If > 0 a value to exclude data based on.

    Returns
    -------
    t : np.array
        Array of normalized time.
    y : np.array
        Array of normalized flux.
    y_err : np.array
        Array of normalized square root of flux.
    """

    t = mjd - np.min(mjd)
    y_err = np.sqrt(flux) 
    y = flux[t > ignore]
    y_err = y_err[t > ignore]
    t = t[t > ignore]
    y_err /= np.max(y)
    y /= np.max(y)

    sorted_tups = sorted(list(zip(t, y, y_err)), key=lambda x: x[0])
    t = np.array([tup[0] for tup in sorted_tups])
    y = np.array([tup[1] for tup in sorted_tups])
    y_err = np.array([tup[2] for tup in sorted_tups])

    return t, y, y_err
    

def plot_transmission_spectrum(curve_dict):
    """ Plots the tranmission spectrum.

    Parameters
    ----------
    curve_dict : dict
        Dictionary of wavelength bin, transit depth, and error.
    """
    rc('figure', figsize=[6.4, 4.8])
    plt.errorbar(curve_dict['wv_bin'], curve_dict['transit_depth'],
            yerr=[curve_dict['lower_err'], curve_dict['upper_err']], fmt='.',
            color='black')
    plt.xlabel('Wavelength [micron]')
    plt.ylabel('Transit Depth [ppm]')
    plt.savefig('transmission_spectrum.png')
    plt.clf()


def ramp(coeffs_init, t, planet_parameters):
    """Creates a ramp model.

    Parameters
    ----------
    coeffs_init : dict
        Dictionary of initial guesses of coefficients.
    t : np.array
        Array of time values over which the transit goes.
    planet_parameters : dict
        Dictionary of static properties of the planet.
    
    Returns
    -------
    ramp_model : np.array
        The factor that the ramp adds at each time value.
    """

    # Pull intial coeffs
    r0 = coeffs_init['r0']
    r1 = coeffs_init['r1']
    r2 = coeffs_init['r2']
    r3 = coeffs_init['r3']
    r4 = coeffs_init['r4']
    c = coeffs_init['c']
    shift = coeffs_init['shift']
    
    t_planet = planet_parameters['period']
    t_hst = planet_parameters['hst_period']

    theta = 2*np.pi*(t % t_planet)/t_planet
    phi = 2*np.pi*((t + shift) % t_hst)/t_hst

    ramp_model = c* (1 + r0*theta + r1*theta**2)* (1 - np.e**(r2*phi + r3) + r4*phi)
    
    return ramp_model


def select_limb_darkening_coeffs(curve_key, dict_key=False):
    """ Match the curve key to the limb darkening coefficients.

    Parameters
    ----------
    curve_key : str of float
        The midpoint between a wavelength bin.

    Returns
    -------
    u1 : float
      The u1 ld coefficient.
    u2 : float
        The u2 ld coefficient.
    """
    with open('ld_dict.json', 'r') as f:
        ld_coeffs = json.load(f)
    
    if dict_key:
        u1, u2 = ld_coeffs[dict_key]['u1'], ld_coeffs[dict_key]['u2']
    else: 
        for key in ld_coeffs:
            if ld_coeffs[key]['bin_mid'] == float(curve_key):
                coeffs = ld_coeffs[key]
                print('Selected bin {} -- {} to {}.'.format(key, coeffs['bin_min'], coeffs['bin_max']))
                print('\n')
                u1 = coeffs['u1']
                u2 = coeffs['u2']

    return [u1, u2]


def cornerplot(chains, titles, truths, name):
    """Builds a cornerplot.

    Parameters
    ----------
    chains : np.array
        Array of chains from MCMC.
    titles : list
        List of titles. 
    truths : list
        List of truths as entry for corner.
    name : str
        Name for the plot.
    """

    plt.rcParams['axes.formatter.useoffset'] = False
    print('Plotting corner plot')

    title_keys             = {}
    title_keys['fontsize'] = 18
    title_keys['loc']      = 'left'
    title_keys['va']       = 'bottom'
    label_keys             = {}
    label_keys['fontsize'] = 18
    label_keys['labelpad'] = 16

    hist_keys = {}
    hist_keys['log'] = False

    rr = [0.99]*chains.shape[1]
    corner.corner(chains, labels = titles, use_math_text = False, 
            quantiles=[.16,0.5,.84], show_titles = 'False', title_fmt = ".2f", 
            title_kwargs = title_keys, label_kwargs = label_keys, hist_kwargs = hist_keys, 
            smooth = False, range = rr, plot_datapoints = True, max_n_ticks = 3, truths = truths)
    plt.savefig('corner_plot_{}.png'.format(name))
    plt.clf()


def transit_compare(coeffs, t, y, y_err, name, mega_dict):
    """ Plots transit comparison figure and writes out some data that may prove
    helpful.

    Parameters
    ----------
    """
    mega_dict[name] = {'coeffs': coeffs, 't': list(t), 'y': list(y), 'y_err':list(y_err)}
    with open('results.json', 'w') as out:
        json.dump(mega_dict, out)
    print('Writing out results for {} to results.json'.format(name))
    return 


## -- RUN

