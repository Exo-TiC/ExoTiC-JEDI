import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy import interpolate
import matplotlib.pyplot as plt
from exotic_ld import StellarLimbDarkening
from scipy.ndimage import gaussian_filter1d
from scipy.special import roots_legendre
from scipy.interpolate import interp1d


# for slicing and dicing your lightcurves

def saber(file_code, jedi_path):

    bins_read=False

    if file_code == 'wide':
        file_path = jedi_path+'exotic_jedi/stage_3/bins/bins_wide.csv'
        bins_read = True

    if file_code == '10pix':
        file_path = jedi_path+'exotic_jedi/stage_3/bins/bins_10pix.csv'
        bins_read = True

    if file_code == '30pix':
        file_path = jedi_path+'exotic_jedi/stage_3/bins/bins_30pix.csv'
        bins_read = True

    if file_code == '60pix':
        file_path = jedi_path+'exotic_jedi/stage_3/bins/bins_60pix.csv'
        bins_read = True

    if file_code == '100pix':
        file_path = jedi_path+'exotic_jedi/stage_3/bins/bins_100pix.csv'
        bins_read = True

    if file_code == 'R100_M':
        file_path = jedi_path+'exotic_jedi/stage_3/bins/bins_G395M_R100.csv'
        bins_read = True

    elif bins_read == False:
        raise Exception("That binning scheme does not appear to exist!")

    return(file_path)




#bins_low, bins_up = np.loadtxt("/home/ym20900/ExoTiC-JEDI/exotic_jedi/stage_3/bins/bins_G395M_R100.txt", unpack=True)
#
#bin_widths = bins_up-bins_low
#bin_centers = bins_low+(bin_widths/2)
#
#bin_file = {
#    "#Bin Edge short" : bins_low,
#    "Bin Edge long" : bins_up,
#    "Bin center" : bin_centers,
#    "Bin width" : bin_widths
#}

#bins_df = pd.DataFrame(bin_file)
#bins_df.to_csv('bins_G395M_R100.csv', index=False)