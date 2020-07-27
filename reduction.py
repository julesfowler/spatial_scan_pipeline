#-- RN this is just a sketch of the basic functions
# -- I'll refit it/pipeline/module-ify it if it begs for that complexity

## -- IMPORTS
import datetime
import glob
import json
from multiprocessing import Pool
import os
import sys

from astropy.io import ascii
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.stats import sigma_clipped_stats
from astropy.time import Time
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from photutils import daofind

## -- FUNCTIONS

def main(data_dir, natural=False):
    """
    Parameters
    ----------
    data_dir : str
        The path to the files to reduce.
    natural : bool, optional
        Whether to fit for the best aperture or 
        force it to the natural aperture.
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

    corrected_files = glob.glob('{}*crcorr2*.fits'.format(data_dir))
    if not natural:
        rectangle = find_best_fit(corrected_files)
    else:
        rectangle = False
    #rectangle = (143, 174)
    wvs, sums = [], [] 
    output_dict = {}
    for grism_file in corrected_files:
        
        # This is cheating but specify a global scale
        with fits.open(grism_file) as hdu:
            data = hdu[1].data
            med = np.median(data)
            std = np.std(data)

        global VMIN
        VMIN = 21784.708984375
        global VMAX
        VMAX = 30807.462890625

        # Pull out the rootname for plotting porpoises.
        name = grism_file.split('_')[0]
        
        # Isolate scan region
        scan, bkg = fit_rectangle(grism_file, name, rectangle=rectangle)
        bkg_avg = np.median(bkg, axis=0)
        
        # Subtract background
        subtracted_scan = scan - np.array([bkg_avg for row in scan])
        print(np.shape(subtracted_scan))
        #subtracted_img = [data_row[10:-10] - bkg_avg for data_row in data]
        
        #with fits.open(grism_file) as hdu:
        #    hdu[1].data = subtracted_img
        #    new_file = '{}_sub_img.fits'.format(name)
        #    hdu.writeto(new_file, overwrite=True)
        
        # Sum and create spectra plots
        wv, col_sums = plot_spectra_from_scan(subtracted_scan, name, direct_file, grism_file)
        wvs.append(wv)
        sums.append(col_sums)
        
        output_dict['{}_wv'.format(name)] = wv
        output_dict['{}_sum'.format(name)] = col_sums

    ascii.write(output_dict, 'output.csv')

    for n, wv in enumerate(wvs):
        plt.plot(wv, sums[n])
    plt.savefig('spectrum_overlay.png')


def common_min_max(dat, slice_index=np.arange(78,188,10), common=3):
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
    #features = [119]
    features = []

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
        
    common_max = max(max_index, key=max_index.count)
    common_min = max(min_index, key=min_index.count)
    print(common_min, common_max) 
    return common_max, common_min


def find_best_fit(files):
    """ Goes through the list of files and finds the aperture
    that leads to the lest std in a given light curve group.
    
    Paramters
    ---------
    files : list of str
        List of CR corrected files.

    Returns
    -------
    rectangle : tuple
        A list of the form (vertical_min, vertical_max).
    """

    # Find the optimal test set
    test_files = find_test_set(files)
    
    edges = []
    for test_file in test_files:
        with fits.open(test_file) as hdu:
            data = hdu[1].data
        common_max, common_min = common_min_max(data)
        edges.append((common_min, common_max))

    c_min = int(np.median([edge[0] for edge in edges]))
    c_max = int(np.median([edge[1] for edge in edges]))

    tests = [1, 3, 5]
    test_edges = [(c_min, c_max)]
    for test in tests:
        test_edges.append((c_min-test, c_max+test))
        test_edges.append((c_min+test, c_max-test))
    
    total_std = []
    for edge in test_edges:
        scans = []
        for test_file in test_files:
            scan, bkg = fit_rectangle(test_file, False, rectangle=edge)
            bkg_avg = np.nanmedian(bkg, axis=0)

            subtracted_scan = scan - np.array([bkg_avg for row in scan])
            scans.append(np.nansum(subtracted_scan))
        total_std.append((edge, np.nanstd(scans)))
        print(edge, np.nanstd(scans), np.abs(edge[0]-edge[1]))

    total_std.sort(key=lambda x:x[1])
    rectangle = total_std[0][0]
    print(c_min, c_max)
    print(rectangle)
    return rectangle

    
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

    print(new_min, new_max)
    #bkg = data[new_min:new_max, horizontal_range[0]:horizontal_range[1]]
    bkg = data[20:100, horizontal_range[0]:horizontal_range[1]]
    return bkg


def find_test_set(files):                                         
    """ Find a set part of the light curve to check for optimal extraction.
    It pulls the 'EXPEND' header keyword and looks for the set of larger time
    differentials. Right now it will split when the time difference is > 5* the
    exposure time. It selects the set with the smallest std -- which is least
    likely to be a set that includes a transit. 

    Parameters
    ----------
    files : list of str
        List of files in the visit.
    
    Returns
    -------
    test_set : list of str
        List of best set of images to test.
    """
    
    # Remove the one direct exposure.
    grism_files = []
    for file in files:
        if 'G' in fits.getval(file, 'FILTER', ext=0):
            grism_files.append(file)
    
    end_times = [fits.getval(file, 'EXPEND', ext=0) for file in grism_files]
    tups = zip(grism_files, end_times)
    tups = sorted(tups, key=lambda x:x[1])
        
    ends = [tup[1] for tup in tups]
    g_files = [tup[0] for tup in tups]
    
    ends = np.array([Time(end, format='mjd').datetime for end in ends])
    diffs = [datetime.timedelta(0,0,0)]
    for index, end in enumerate(ends):
         if index > 0:
             diffs.append(end - ends[index-1])
    
    diffs = [diff.seconds for diff in diffs]
    splits = np.arange(len(g_files))[diffs > 5*np.median(diffs)]
    splits = list(splits) + [len(g_files)]
    file_sets = []
    for index, split in enumerate(splits):
        if index > 0:
            file_sets.append(np.array(files)[np.arange(splits[index-1],split)])

    stds = []
    for index, file_set in enumerate(file_sets):
        file_set_trim = file_set[3:]
        if index == 0:
            stds.append(np.nan)
        else:
            sums = [np.sum(fits.open(file)[1].data) for file in file_set_trim]
            stds.append(np.std(sums))
    split_index = np.where(stds == np.nanmin(stds))[0][0]+1
    print(split_index)
    splits = [0] + splits
    test_set = np.array(g_files)[np.arange(splits[split_index-1], splits[split_index])]
    
    return test_set


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
    name : str or bool
        What name to save the plots to or False if you don't want them.
    rectangle : list of tuples, optional
        If you already know the aperture and just want the function to
        return the bkg subtracted scan, a list of tuple of the form
        (vertical_max, vertical_min).


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
    if rectangle:
        common_min, common_max = rectangle
    else:
        common_max, common_min = common_min_max(data)
    
    vertical_region = clean_data[common_min:common_max, :]
    vertical_range = (common_min, common_max)
    replace_val = 300*np.max(data)
    data[common_max, :] = replace_val
    data[common_min, :] = replace_val
    
    # Now horizontal
    m, n = np.shape(vertical_region)
    
    horizontal_range = (5, n-5)
    scan = vertical_region[:, 5:n-5]
    if name:
        # Save a plot of the region
        plt.imshow(data, cmap='viridis', vmin=VMIN, vmax=VMAX)
        plt.savefig('{0}/regions/{1}_cutoffs.png'.format('/'.join(name.split('/')[:-1]), name.split('/')[-1]))
        plt.clf()
   
        plt.imshow(scan, cmap='viridis', vmin=VMIN, vmax=VMAX)
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
    x = np.arange(len(column_sums))+10
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
    old_col = time_col.copy()
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
    
    # Create a stack of the visit
    with fits.open(grism_files[0]) as hdu:
        cr_stack = np.array([hdu[1].data.copy()])

    for grism_file in grism_files[1:]:
        with fits.open(grism_file) as hdu:
            cr_stack = np.vstack((cr_stack, np.array([hdu[1].data.copy()])))

    d, m, n = np.shape(cr_stack)

    count = 0
    cosmic_rays = True
    while cosmic_rays:

        # Flatten the stack for parallelization
        flat_stack = cr_stack.reshape((d, m*n)).transpose() 
        p = Pool(8)
        results = np.array(p.map(find_cosmic_rays, flat_stack)).transpose()
        reshaped_results = np.array(results).reshape((d, m, n))
        
        # Remove any extended cosmic ray that probably isn't a cosmic ray
        diff = np.array(reshaped_results == cr_stack, dtype=int)
        for index_d, img in enumerate(diff):
            for index_m, row in enumerate(img):
                row_str = ''.join(str(elem) for elem in row)
                extended_cr = [len(cr) > 5 for cr in row_str.split('1')]
                if True in extended_cr:
                    print('Found a fake/extended CR in {}.'.format(grism_files[index_d]))
                    reshaped_results[index_d, index_m, :] = cr_stack[index_d, index_m, :]
        
        # See if any cosmic rays remain
        if np.sum(reshaped_results - cr_stack) == 0:
            cosmic_rays = False
        cr_stack = reshaped_results
        count += 1
        print('Iteration {} complete!'.format(count))
    
    # Write out corrections
    for index, grism_file in enumerate(grism_files):
        with fits.open(grism_file) as hdu:
            hdu[1].data = cr_stack[index]
            name_bits = grism_file.split('_ima')
            new_file = '{}_crcorr2_ima{}'.format(name_bits[0], name_bits[1])
            hdu.writeto(new_file, overwrite=True)
            print('New file written to {}.'.format(new_file))

    

## -- RUN

if __name__ == "__main__":
    path = sys.argv[1]
    if len(sys.argv) > 2:
        print('test')
        natural = sys.argv[2]
        main(path, natural=natural)
    else:
        main(path)
