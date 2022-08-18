#######################
#### ExoTiC - JEDI ####
#######################
# Exoplanet Timeseries Characterisation - JWST Extraction and Diagnostic Investigatior


# Functions for NIRSpec pixels to spectra analysis!


# Authors: Lili Alderson lili.alderson@bristol.ac.uk
#          David Grant david.grant@bristol.ac.uk


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


