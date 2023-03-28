#######################
#### ExoTiC - JEDI ####
#######################
# Exoplanet Timeseries Characterisation - JWST Extraction and Diagnostic Investigatior


# Functions for NIRSpec pixels to spectra analysis!


# Authors: Lili Alderson lili.alderson@bristol.ac.uk
#          David Grant david.grant@bristol.ac.uk
#          Hannah R Wakeford hannah.wakeford@bristol.ac.uk


# File Created : 18 July 2022
#                Lili Alderson lili.alderson@bristol.ac.uk



# Imports
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm

import os
from tqdm import tqdm

from astropy.io import fits
from astropy.timeseries import LombScargle
from astropy.stats import sigma_clipped_stats

from scipy.optimize import curve_fit
from scipy.signal import medfilt
from scipy.interpolate import interp1d
from scipy import interpolate
from scipy import signal

import pickle



def unsegment(pathlist, exten):
    '''
    # From Jeff Valenti:
    # Read data from segmented files into a single stack.
    # Updated by H.Wakeford
    
    # Inputs
    # pathlist : list of filepaths for the segments in the observation
    # exten : which fits file extension to unsegment (e.g., 1=data, 3=dq_flags)

    # Outputs
    # stack : 3D array of the unsegmented fits extention
    # [x0, y0] : tuple of x0 and y0
    '''

    for path in pathlist:
        with fits.open(path) as hdu:
            data = hdu[exten].data
            exthead = hdu[exten].header
            primhead = hdu[0].header
            if path is pathlist[0]:
                stack = data
                try:
                    x0 = primhead['SUBSTRT1'] + exthead['SLTSTRT1'] - 2
                    y0 = primhead['SUBSTRT2'] + exthead['SLTSTRT2'] - 2
                except KeyError:
                    x0 = primhead['SUBSTRT1'] - 1
                    y0 = primhead['SUBSTRT2'] - 1
            else:
                stack = np.append(stack, data, axis=0)
                try:
                    assert x0 == primhead['SUBSTRT1'] + exthead['SLTSTRT1'] - 2
                    assert y0 == primhead['SUBSTRT2'] + exthead['SLTSTRT2'] - 2
                except KeyError:
                    assert x0 == primhead['SUBSTRT1'] - 1
                    assert y0 == primhead['SUBSTRT2'] - 1
    return stack, [x0, y0]





def dq_flag_metrics(data_cube, dq_cube, plot_bit=None):
    '''
    # Provides a list of the number of pixels flagged with each data quality flag, 
    # both the DQ bit and the readable name of the flag are given
    
    # Inputs
    # data_cube : 3D array of the science extention from the fits file
    # dq_cube : 3D array of the data quality extention from the fits files
    # plot_bit=None : whether to plot the flagged pixels in an imshow
    '''
    
    flags_dict = {0: "DO_NOT_USE", 1: "SATURATED", 2: "JUMP_DET",
                  3: "DROPOUT", 4: "OUTLIER", 5: "PERSISTENCE",
                  6: "AD_FLOOR", 7: "RESERVED", 8: "UNRELIABLE_ERROR",
                  9: "NON_SCIENCE", 10: "DEAD", 11: "HOT", 12: "WARM",
                  13: "LOW_QE", 14: "RC", 15: "TELEGRAPH", 16: "NONLINEAR",
                  17: "BAD_REF_PIXEL", 18: "NO_FLAT_FIELD", 19: "NO_GAIN_VALUE",
                  20: "NO_LIN_CORR", 21: "NO_SAT_CHECK", 22: "UNRELIABLE_BIAS",
                  23: "UNRELIABLE_DARK", 24: "UNRELIABLE_SLOPE",
                  25: "UNRELIABLE_FLAT", 26: "OPEN", 27: "ADJ_OPEN",
                  28: "UNRELIABLE_RESET", 29: "MSA_FAILED_OPEN",
                  30: "OTHER_BAD_PIXEL", 31: "REFERENCE_PIXEL"}

    # Find flags.
    flags_int, flags_row, flags_col = np.where(dq_cube != 0)

    # Iterate flags replacing if flags specified by user.
    dq_tesseract_bits = np.zeros(data_cube.shape + (32,))
    for f_int, f_row, f_col in zip(flags_int, flags_row, flags_col):

        # Flag value is a sum of associated flags.
        value_sum = dq_cube[f_int, f_row, f_col]

        # Unpack flag value as array of 32 bits comprising the integer.
        # NB. this array is flipped, value of flag 1 is first bit on the left.
        bit_array = np.flip(np.array(list(
            np.binary_repr(value_sum, width=32))).astype(int))

        # Track replacements.
        dq_tesseract_bits[f_int, f_row, f_col, :] = bit_array

    # Cleaned metrics.
    total_cleaned = 0
    total_pixels = np.prod(data_cube.shape)
    print('===== DQ flags info =====')
    for bit_idx in range(32):
        nf_found = int(np.sum(dq_tesseract_bits[:, :, :, bit_idx]))
        print('Found {} pixels with DQ bit={} name={}.'.format(
            nf_found, bit_idx, flags_dict[bit_idx]))
        total_cleaned += nf_found
    print('DQ fraction of total pixels={} %'.format(
        round(total_cleaned / total_pixels * 100., 3)))

    if plot_bit is not None:
        plt.imshow(np.max(dq_tesseract_bits[:, :, :, plot_bit], axis=0),
                   origin='lower', aspect='auto', interpolation='none')
        plt.show()


def noise_calculator(residuals, maxnbins=None, binstep=1, rndm_rlz=10,
                     plot_bit=True, beta_nbin_range=None):
    """
    Calculate the noise properties and Allan Variance plot for a given set
    of fit residuals.
    Author: Hannah R. Wakeford, University of Bristol, edited by MCR
    Email: hannah.wakeford@bristol.ac.uk
    Citation: Laginja & Wakeford, 2020, JOSS, 5, 51 (https://joss.theoj.org/papers/10.21105/joss.02281)

    Parameters
    ----------
    residuals : array-like
        Fit residuals in ppm.
    maxnbins : int, None
        Maxmimum number of integrations to put in a single bin.
    binstep : int
        Bin step size.
    rndm_rlz : int
        Number of random noise realizations to generate (only relevant for
        plotting).
    plot_bit : bool
        If True, show Allan Variance Plot.
    beta_nbin_range : tuple, None
        Range of binned integrations over which to calculate beta factor.

    Returns
    -------
    white_noise : float
        Estimate of white noise in ppm.
    red_noise : float
        Estimate of red noise in ppm.
    beta : float
        Beta scaling factor to account for correlated noise.
    """

    # Bin data into multiple bin sizes
    if maxnbins is None:
        maxnbins = len(residuals) / 10.

    # Create an array of the bin sizes in integrations.
    binz = np.arange(1, maxnbins + binstep, step=binstep, dtype=int)

    # Initialize some storage arrays.
    nbins = np.zeros(len(binz), dtype=int)
    root_mean_square = np.zeros(len(binz))
    root_mean_square_err = np.zeros(len(binz))

    # Loop over each bin size and bin the requisite number of integrations.
    for i in range(len(binz)):
        # Total number of bins binning binz[i] integrations per bin
        nbins[i] = int(np.floor(residuals.size / binz[i]))
        bindata = np.zeros(nbins[i], dtype=float)

        # Do the binning
        for j in range(nbins[i]):
            bindata[j] = np.nanmean(residuals[j * binz[i]: (j + 1) * binz[i]])

        # Get RMS statistic and associated error
        root_mean_square[i] = np.sqrt(np.nanmean(bindata ** 2))
        root_mean_square_err[i] = root_mean_square[i] / np.sqrt(nbins[i])

    # Get expected white noise trend
    expected_noise = (np.nanstd(residuals) / np.sqrt(binz)) * np.sqrt(
        nbins / (nbins - 1.))

    # If generating a plot, it can be useful to show multiple random
    # generations of white noise instead of just the "expected photon noise"
    # trend. Repeat the above for random white noise realizations.
    if plot_bit is True and rndm_rlz != 0:
        random_samples = np.zeros((rndm_rlz, len(binz)))
        # Loop over each realization
        for g in range(rndm_rlz):
            this_sample = []
            # Generate white noise with same std dev as the residuals.
            wht_residuals = np.random.normal(loc=0.,
                                             scale=np.nanstd(residuals),
                                             size=residuals.size)
            # Repeat the above binning process.
            for i in range(len(binz)):
                bindata = np.zeros(nbins[i], dtype=float)
                for j in range(nbins[i]):
                    bindata[j] = np.nanmean(
                        wht_residuals[j * binz[i]: (j + 1) * binz[i]])
                this_sample.append(np.sqrt(np.nanmean(bindata ** 2)))
            random_samples[g] = this_sample
        # Calculate 1 and 2 sigma envelopes of the white noise trends.
        pctl = np.percentile(random_samples, [2.3, 15.9, 84.1, 97.7], axis=0)
        low2, low1, up1, up2 = pctl

    # Calculate beta factor.
    if beta_nbin_range is None:
        beta_nbin_range = (10, int(maxnbins))
    bin_low = np.argmin(abs(binz - beta_nbin_range[0]))
    bin_up = np.argmin(abs(binz - beta_nbin_range[1]))
    # Estimate the white and red noise variance over the desired range.
    white = expected_noise[bin_low:bin_up] ** 2
    red = root_mean_square[bin_low:bin_up] ** 2
    white_noise = np.sqrt(np.mean(white))
    red_noise = np.sqrt(np.abs(np.mean(red - white)))
    # Calculate the beta scaling factor.
    beta = np.sqrt(np.mean(red / white))

    # Plot up the binned statistic against the expected statistic.
    if plot_bit is True:
        plt.figure(facecolor='white', figsize=(7, 5))
        plt.errorbar(binz, root_mean_square, yerr=root_mean_square_err,
                     fmt='-o', ms=2,
                     color='black', label=' Residual RMS')
        plt.plot(binz, expected_noise, color='red', ls='-', lw=2,
                 label='White Noise')
        if rndm_rlz != 0:
            plt.fill_between(binz, low2, up2, facecolor='black',
                             edgecolor='None', alpha=0.2)
            plt.fill_between(binz, low1, up1, facecolor='black',
                             edgecolor='None', alpha=0.2)
        plt.xlabel('Bin Size [integrations]', fontsize=16)
        plt.ylabel('RMS [ppm]', fontsize=16)
        plt.xscale('log')
        plt.xticks(fontsize=12)
        plt.yscale('log')
        plt.yticks(fontsize=12)
        plt.legend(fontsize=14)
        plt.show()

    return white_noise, red_noise, beta, binz, root_mean_square, root_mean_square_err, expected_noise