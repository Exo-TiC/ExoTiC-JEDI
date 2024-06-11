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


def make_custom_throughput(_s_mus, _s_wvs, _s_ints, _d_wvs, _d_fls):
    rs = (1 - _s_mus ** 2) ** 0.5
    roots, weights = roots_legendre(500)
    a, b = (0., 1.)
    t = (b - a) / 2 * roots + (a + b) / 2

    spectrum = []
    for wv_idx in range(_s_wvs.shape[0]):
        i_interp_func = interp1d(
            rs, _s_ints[wv_idx, :], kind='linear',
            bounds_error=False, fill_value=0.)

        def integrand(_r):
            return i_interp_func(_r) * _r * 2. * np.pi

        spectrum.append((b - a) / 2. * integrand(t).dot(weights))

    spectrum = np.array(spectrum)

    interp_function = interpolate.interp1d(
        _s_wvs * 1e-4, spectrum, kind="linear")
    spectrum_interp = interp_function(_d_wvs)
    _throughput = gaussian_filter1d(
        np.median(_d_fls, axis=0) / spectrum_interp, 100)

    return _throughput