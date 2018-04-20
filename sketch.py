#-- RN this is just a sketch of the basic functions
# -- I'll refit it/pipeline/module-ify it if it begs for that complexity

## -- IMPORTS
import glob
from multiprocessing import Pool
import os

from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.stats import sigma_clipped_stats
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from photutils import daofind

## -- FUNCTIONS

def main(data_dir = '../data/'):
    """
    Parameters
    ----------
    data_dir : str, optional
        The path to the files to reduce.
    """
    # Make directory structure for plots/outputs
    for folder in ['regions', 'spectra']:
        if not os.path.exists(data_dir + folder):
            os.makedirs(data_dir + folder)

    # Collect grism data
    files = glob.glob(data_dir + '*q_ima.fits')
    grism_files = []
    for infile in files:
        with fits.open(infile) as hdu:
            if 'G' in hdu[0].header['FILTER']:
                grism_files.append(infile)
            else:
                direct_file = infile

    #reject_cosmic_rays(grism_files)
    print('CR rejection complete.')

    corrected_files = glob.glob('{}*crcorr*.fits'.format(data_dir))

    for grism_file in corrected_files:
        
        # Pull out the rootname for plotting porpoises.
        name = grism_file.split('_')[0]

        # Isolate scan region
        scan, bkg = fit_rectangle(grism_file, name)
        
        # Subtract background
        subtracted_scan = scan - bkg

        # Sum and create spectra plots
        plot_spectra_from_scan(subtracted_scan, name, direct_file, grism_file)
    plt.savefig('test.png')
    print('Plotted spectra for : {}'.format(name))

def common_min_max(dat, slice_index, common=3):
    """ Takes slices as determined by slice_index
    and finds the maximum difference between two pixels.

    Parameters
    ----------
    dat : np.array
        Data array of pixel values.
    slice_index: np.array
        List of slices to test.
    common : int, optional
        How common the max/min values have to be to count.

    Returns
    -------
    common_max : int
        The row with the highest diff.
    common_min : int
        The row with the lowest diff.
    """
    
    # Features in the scan that gunk up the works
    features = [119]


    max_index, min_index = [], []
    for index in slice_index:
        
        test_slice = dat[:, index]
        diff = [test_slice[n] - test_slice[n+1] for n in range(len(test_slice) -1)]
        max_diff_index = np.where(diff == np.max(diff))
        min_diff_index = np.where(diff == np.min(diff))
        if len(max_diff_index[0]) == 1:
            max_index.append(max_diff_index[0][0])
        if len(min_diff_index[0]) == 1 and min_diff_index[0][0] not in features:
            min_index.append(min_diff_index[0][0])
        
    print(max_index)
    print(min_index)
    print('---')
    common_max = max(max_index, key=max_index.count)
    common_min = max(min_index, key=min_index.count)
    
    print(common_max, common_min)
    return common_max, common_min


def find_bkg_region(data, horizontal_range, vertical_range):
    """ Select out the best background region to subtract from data.
    Uses same horiztonal range so any wavelength dep background is 
    accounted for.

    Parameters
    ----------
    data : np.array
        Array of data from the spatial scan.
    horizontal_range : tup of ints
        Tuple containing (min, max) of the horizontal extent of the
        selected out scan.
    vertical_range : tup of ints
        Tuple containing (min, max) of the vertical extent of the 
        selected out scan.

    Returns
    -------
    bkg : np.array
        Array of data the same size as the selected out 
        scan but of background data.
    """
    
    y, x = np.shape(data)
    v_min, v_max = vertical_range
    
    # See if background should be selected from above or below scan.
    if y - v_max > v_min:
        new_min = int(y - (y-v_max)/2 - (v_max - v_min)/2)
        new_max = int(y - (y-v_max)/2 + (v_max - v_min)/2)
    else:
        new_min = int(v_min/2 - (v_max-v_min)/2)
        new_max = int(v_min/2 + (v_max-v_min)/2)

    bkg = data[new_min:new_max, horizontal_range[0]:horizontal_range[1]]
    return bkg


def fit_rectangle(grism_file, name, rectangle=False):
    """
    Selects a rectangle fit. Saves a plot
    so you can go check it looks okay.
    
    If 1. It isn't obvious using max/min pixel changes or 
    2. It seems slanted -- the function will print a warning
    instead and skip the file.

    Parameters
    ----------
    grism_file : str
        The path to the file.
    name : str
        What name to save the plots to.

    Returns
    -------
    scan : np.array
        The area of the scan.
    bkg : float
        A mean background level.
    """
    with fits.open(grism_file) as hdu:
        data = hdu[1].data
    clean_data = data.copy()

    # First the vertical trim
    slice_index = np.arange(78, 188, 10)
    if rectangle:
        common_max, common_min = rectangle[0]
    else:
        common_max, common_min = common_min_max(data, slice_index)
    common_max += 5
    common_min -= 5
    
    vertical_region = clean_data[common_min:common_max, :]
    vertical_range = (common_min, common_max)
    replace_val = 300*np.max(data)
    data[common_max, :] = replace_val
    data[common_min, :] = replace_val
    
    # Now horizontal
    m, n = np.shape(vertical_region)
    slice_index = np.arange(0, m, 5)
    if rectangle:
        common_max, common_min = rectangle[1]
    else:
        common_max, common_min = common_min_max(np.transpose(vertical_region), slice_index)
    common_max += 5
    common_min -= 5
    scan = vertical_region[:, common_min:common_max]
    horizontal_range = (common_min, common_max)
    data[:, common_max] = replace_val
    data[:, common_min] = replace_val
    
    # Save a plot of the region
    plt.imshow(data, cmap='viridis', vmin=21784.708984375, vmax=30807.462890625)
    plt.savefig('{0}/regions/{1}_cutoffs.png'.format('/'.join(name.split('/')[:-1]), name.split('/')[-1]))
    plt.clf()
   
    plt.imshow(scan, cmap='viridis', vmin=21784.708984375, vmax=30807.462890625)
    plt.savefig('{0}/regions/{1}_scans.png'.format('/'.join(name.split('/')[:-1]), name.split('/')[-1]))
    plt.clf()
    
    bkg = find_bkg_region(clean_data, horizontal_range, vertical_range)

    return scan, bkg


def plot_spectra_from_scan(scan, name, direct_file, grism_file):
    """ Sum over each column to make the spectra.
    Save a plot.

    Parameters
    ----------
    scan : np.array
       A data array of the scan.
    name : str
        The name to save the plot to.
    """

    tr_scan = np.transpose(scan)
    column_sums = [sum(scan_col) for scan_col in tr_scan[5:-5]]
    x = np.arange(len(column_sums))
    wv = convert_rows_to_wv(direct_file, grism_file, x)
    plt.plot(wv, column_sums)
    plt.savefig('{0}/spectra/{1}_spectrum.png'.format('/'.join(name.split('/')[:-1]), name.split('/')[-1]))
    plt.clf()
    return wv, column_sums

def find_cosmic_rays(time_col):
    """
    Identifies cosmic rays. 
    CR is defined as > 5 sigma from time column.

    Parameters
    ----------
    time_col : np.array
        A column through time of a point in the image,

    Returns
    -------
    cosmic_rays : list of tuples
        A list of cosmic rays.
    """
    time_col = time_col.copy() 
    clipped_col = sigma_clip(time_col, sigma=5)
    crs = clipped_col.data[clipped_col.mask]
     
    cosmic_rays = []
    if len(crs) > 0:
        for cr in crs:
            index = np.where(cr == time_col)[0]
            time_col[np.where(cr == time_col)] = np.median(time_col)
    
    return time_col


def convert_rows_to_wv(direct_file, grism_file, rows):
    """ Converts the rows to wavelength bins. 

    Parameters
    ----------
    direct_file : str
        The path to the direct file.
    grism_file : str
        The path to the grism file.
    rows : array
        The array of rows that correspond to the spatial scan.

    Returns
    ------
    wv : array
        The array solution for for wavelength.
    """

    # Collect data from FITS headers
    with fits.open(grism_file) as hdu:
        hdr = hdu[0].header
        hdr1 = hdu[1].header
        sci_postarg_1 = hdr['POSTARG1']
        sci_postarg_2 = hdr['POSTARG2']
        sci_crpix_1 = hdr1['CRPIX1'] # this isn't a real keyword...
        sci_crpix_2 = hdr1['CRPIX2'] 

    with fits.open(direct_file) as hdu:
        hdr = hdu[0].header
        hdr1 = hdu[1].header
        data = hdu[1].data
        cal_postarg_1 = hdr['POSTARG1']
        cal_postarg_2 = hdr['POSTARG2']
        cal_crpix_1 = hdr1['CRPIX1']
        cal_crpix_2 = hdr1['CRPIX2']


    # Find the central source
    mean, med, std = sigma_clipped_stats(data, sigma=3.0, iters=5)
    sources = daofind(data-med, fwhm=3.0, threshold=5.*std)
    
    source = sources[np.where(sources['flux'] == np.max(sources['flux']))]
    x_cen, y_cen = source['xcentroid'], source['ycentroid']


    # Calculate the offset
    x_offset = sci_crpix_1 - cal_crpix_1 + (sci_postarg_1 - cal_postarg_1)/0.135
    y_offset = sci_crpix_2 - cal_crpix_2 + (sci_postarg_2 - cal_postarg_2)/0.121

    pos_x, pos_y = x_cen + x_offset, y_cen + y_offset

    constants_0 = [8.95E3, 9.35925E-2, 0.0, 0.0, 0.0, 0.0]
    constants_1 = [4.51423E1, 3.17239E-4, 2.17055E-3, -7.42504E-7, 3.4863E-7, 3.09213E-7]

    coords_0 = constants_0[0] + constants_0[1]*pos_x + constants_0[2]*pos_y
    coords_1 = constants_1[0] + constants_1[1]*pos_x + constants_1[2]*pos_y + constants_1[3]*pos_x**2 + constants_1[4]*pos_x*pos_y + constants_1[5]*pos_y**2
    
    wv = coords_0 + coords_1*(rows-pos_x) + pos_y

    return wv


def reject_cosmic_rays(grism_files, n_iter=3):
    """ Routine to isolate and reject cosmic rays
    and replace them with the median value in the 
    image.

    Parameters
    ----------
    grism_files : list of str
        List of files to check for CRs.
    n_iter : int, optional
        The number of times to do interatively flag
        crs.
    """

    with fits.open(grism_files[0]) as hdu:
        cr_stack = np.array([hdu[1].data.copy()])

    for grism_file in grism_files[1:]:
        with fits.open(grism_file) as hdu:
            cr_stack = np.vstack((cr_stack, np.array([hdu[1].data.copy()])))

    d, m, n = np.shape(cr_stack)

    count = 0
    while count < n_iter:
        flat_stack = cr_stack.reshape((d, m*n)).transpose() 
        p = Pool(8)
        results = np.array(p.map(find_cosmic_rays, flat_stack)).transpose()
        reshaped_results = np.array(results).reshape((d, m, n))
        count += 1
        print('Iteration {} complete!'.format(count))
        
    for index, grism_file in enumerate(grism_files):
        with fits.open(grism_file) as hdu:
            hdu[1].data = cr_stack[index]
            name_bits = grism_file.split('_ima')
            new_file = '{}_crcorr_ima{}'.format(name_bits[0], name_bits[1])
            hdu.writeto(new_file, overwrite=True)
            print('New file written to {}.'.format(new_file))

    


## -- RUN

if __name__ == "__main__":
    main()
