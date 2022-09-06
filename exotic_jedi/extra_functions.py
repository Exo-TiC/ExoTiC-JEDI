#######################
#### ExoTiC - JEDI ####
#######################
# Exoplanet Timeseries Characterisation - JWST Extraction and Diagnostic Investigatior


# Functions for NIRSpec pixels to spectra analysis!


# Authors: Lili Alderson lili.alderson@bristol.ac.uk
#          David Grant david.grant@bristol.ac.uk
#          Hannah R Wakeford hannah.wakeford@bristol.ac.uk


# File Created : 18 July 2022
#				 Lili Alderson lili.alderson@bristol.ac.uk



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
    From Jeff Valenti:
    Read data from segmented files into a single stack.
    Updated by H.Wakeford
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


def noise_calculator(data, maxnbins=None, binstep=1, plot_bit=None):
    import numpy as np
    import matplotlib.pyplot as plt
    """
    Author: Hannah R. Wakeford, University of Bristol
    Email: hannah.wakeford@bristol.ac.uk
    Citation: Laginja & Wakeford, 2020, JOSS, 5, 51 (https://joss.theoj.org/papers/10.21105/joss.02281)
    
    Calculate the noise parameters of the data by using the residuals of the fit
    :param data: array, residuals of (2nd) fit
    :param maxnbins: int, maximum number of bins (default is len(data)/10)
    :param binstep: bin step size
    :return:
        red_noise: float, correlated noise in the data
        white_noise: float, statistical noise in the data
        beta: float, scaling factor to account for correlated noise
        
    History: 
        6 Sep 2022: change mean and std functions to nanmean and nanstd
    """

    # bin data into multiple bin sizes
    npts = len(data)
    if maxnbins is None:
        maxnbins = npts/10.

    # create an array of the bin steps to use
    binz = np.arange(1, maxnbins+binstep, step=binstep, dtype=int)

    # Find the bin 2/3rd of the way down the bin steps
    midbin = int((binz[-1]*2)/3)

    nbins = np.zeros(len(binz), dtype=int)
    standard_dev = np.zeros(len(binz))
    root_mean_square = np.zeros(len(binz))
    root_mean_square_err = np.zeros(len(binz))
    
    for i in range(len(binz)):
        nbins[i] = int(np.floor(data.size/binz[i]))
        bindata = np.zeros(nbins[i], dtype=float)
        
        # bin data - contains the different arrays of the residuals binned down by binz
        for j in range(nbins[i]):
            bindata[j] = np.nanmean(data[j*binz[i] : (j+1)*binz[i]])

        # get root_mean_square statistic
        root_mean_square[i] = np.sqrt(np.nanmean(bindata**2))
        root_mean_square_err[i] = root_mean_square[i] / np.sqrt(2.*nbins[i])
      
    expected_noise = (np.nanstd(data)/np.sqrt(binz)) * np.sqrt(nbins/(nbins - 1.))
 
    final_noise = np.nanmean(root_mean_square[midbin:])
    bnoise = abs(final_noise**2 - root_mean_square[0]**2)
    base_noise = np.sqrt(bnoise / nbins[midbin])

    # Calculate the random noise level of the data
    white_noise = np.sqrt(root_mean_square[0]**2 - base_noise**2)
    # Determine if there is correlated noise in the data
    cnoise = abs(final_noise**2 - white_noise**2)
    red_noise = np.sqrt(cnoise / nbins[midbin])
    # Calculate the beta scaling factor
    beta = np.sqrt(root_mean_square[0]**2 + nbins[midbin] * red_noise**2) / root_mean_square[0]

    # If White, Red, or Beta return NaN's replace with 0, 0, 1
    white_noise = np.nan_to_num(white_noise, copy=True)
    red_noise = np.nan_to_num(red_noise, copy=True)
    beta = 1 if np.isnan(beta) else beta
    
    # Plot up the bin statistic against the expected statistic
    # This can be used later when we are setting up unit testing.
    if plot_bit is not None:
        plt.figure()
        plt.errorbar(binz, root_mean_square, yerr=root_mean_square_err, color='k', lw=1.5, label='RMS')
        plt.plot(binz, expected_noise, color='r', ls='-', lw=2, label='expected noise')
        
        plt.title('Expected vs. measured noise binning statistic')
        plt.xlabel('Number of bins')
        plt.ylabel('RMS')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return white_noise, red_noise, beta
