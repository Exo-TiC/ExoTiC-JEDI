# Modified by H.R.Wakeford May 2023

import batman
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def fit_white_light_curve(times, flux, flux_err, x, y, sys_select,
                          ld_coefs, law, planet_params,
                          sys_params=np.array([1.,0.,0.,0.,0.]),
                          outlier_threshold=5.,
                          draw_fits=False):
    
    sys_fit = np.empty(len(sys_params), dtype=float)

    params = batman.TransitParams()
    params.t0 = planet_params['t0'][0]
    params.per = planet_params['period'][0]  # fixed.
    params.rp = planet_params['rp_rs'][0] #rp/rs
    params.a = planet_params['a_rs'][0] #a/r*
    params.inc = planet_params['inclination'][0]
    params.ecc = planet_params['eccentricity'][0]  # fixed.
    params.w = planet_params['w_omega'][0]  # fixed.
    params.u = ld_coefs  # fixed.
    params.limb_dark = law  # fixed.
    m = batman.TransitModel(params, times)

    _iter = 0
    mask = np.ones(times.shape).astype(bool)
    while True:
        def model(_, t0, rp, a, inc, s0=None, s1=None, s2=None, s3=None, s4=None, use_mask=True):

            # Free physical params.
            params.t0 = t0
            params.rp = rp
            params.a = a
            params.inc = inc
            light_curve = m.light_curve(params)

            # Systematics
            if (sys_select == 'x_y'):
                sys = s0 + (s1 * x) + (s2 * y) + (s3 * times)
            if (sys_select == 'xy'):
                aby = abs(y-np.median(y))
                sys = s0 + (s1 * (x*aby)) + (s2 * times)
            if (sys_select == 'xyt2'):
                aby = abs(y-np.median(y))
                sys = s0 + (s1 * (x*aby)) + (s2*times + s3*times*times)
            if (sys_select == 'ramp'):
                sys = s0 - np.exp(-s1 * times + s2) + (s3 * times)            
            
            light_curve[:] *= sys

            if use_mask:
                return light_curve[mask]
            else:
                return light_curve

        _iter += 1       
        p0_guess = [planet_params['t0'][0], planet_params['rp'][0], planet_params['a_rs'][0], planet_params['inclination'][0]]
        p0_guess.extend(sys_params.tolist())
        popt, pcov = curve_fit(
            model, times[mask], flux[mask], sigma=flux_err[mask],
            p0=p0_guess,
            method='lm')

        perr = np.sqrt(np.diag(pcov))
        rp = popt[1]
        rp_err = perr[1]
        transit_depth = rp**2
        transit_depth_err = rp_err/rp * 2 * transit_depth
        print('Rp/Rs={} +- {}'.format(rp, rp_err))
        print('Transit depth={} +- {}'.format(transit_depth, transit_depth_err))
        print('t0={}'.format(popt[0]))
        print('a={}'.format(popt[2]))
        print('inc={}'.format(popt[3]))

        opt_model = model(times, *popt)
        residuals = flux - opt_model
        print('Residuals={} ppm'.format(np.std(residuals) * 1.e6))
        dev_trace = np.abs(residuals) / np.std(residuals)
        dev_trace = np.ma.array(dev_trace, mask=~mask)
        max_deviation_idx = np.ma.argmax(dev_trace)

        if dev_trace[max_deviation_idx] > outlier_threshold:
            mask[max_deviation_idx] = False
            continue
        else:
            break

    if draw_fits:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7))
        ax1.get_shared_x_axes().join(ax1, ax2)
        ax1.errorbar(times[mask], flux[mask], yerr=flux_err[mask], fmt='.', zorder=0, alpha=0.2)
        ax1.plot(times, opt_model, zorder=1)
        ax2.errorbar(times[mask], residuals[mask], yerr=flux_err[mask], fmt='.')
        plt.tight_layout()
        plt.show()

    t0 = popt[0]
    a = popt[2]
    inc = popt[3]
    sys_fit = popt[4:]
    print('sys_fit values = ', sys_fit)

    no_planet_popt = np.copy(popt)
    no_planet_popt[1] = 0.
    sys_model = model(times, *no_planet_popt, use_mask=False)

    fit_dict = {
        "light_curve_model": opt_model,
        "corrected_flux": flux - sys_model + 1,
        "corrected_flux_error": flux_err,
        "systematic_model": sys_model,
        "residual": residuals,
    }

    return t0, a, inc, sys_fit, fit_dict


def fit_spec_light_curve(times, flux, flux_err, x, y,
                         ld_coefs, law, sys_select, planet_params, 
                         t0, a, inc,
                         sys_params=np.array([1.,0.,0.,0.,0.]),
                         outlier_threshold=5., draw_fits=False):
    params = batman.TransitParams()
    params.t0 = t0  # fixed from wlc fit.
    params.per = planet_params['period'][0]  # fixed.
    params.rp = planet_params['rp_rs'][0]
    params.a = a  # fixed from wlc fit.
    params.inc = inc  # fixed from wlc fit.
    params.ecc = planet_params['eccentricity'][0]  # fixed.
    params.w = planet_params['w_omega'][0]  # fixed.
    params.u = ld_coefs  # fixed.
    params.limb_dark = law  # fixed.
    m = batman.TransitModel(params, times)

    _iter = 0
    mask = np.ones(times.shape).astype(bool)
    while True:

        def model(_, rp, s0=None, s1=None, s2=None, s3=None, s4=None, use_mask=True):

            # Free physical params.
            params.rp = rp
            light_curve = m.light_curve(params)

            # Systematics
            if (sys_select == 'x_y'):
                sys = s0 + (s1 * x) + (s2 * y) + (s3 * times)
            if (sys_select == 'xy'):
                aby = abs(y-np.median(y))
                sys = s0 + (s1 * (x*aby)) + (s2 * times)
            if (sys_select == 'xyt2'):
                aby = abs(y-np.median(y))
                sys = s0 + (s1 * (x*aby)) + (s2*times + s3*times*times)
            if (sys_select == 'ramp'):
                sys = s0 - np.exp(-s1 * times + s2) + (s3 * times)

            light_curve[:] *= sys

            if use_mask:
                return light_curve[mask]
            else:
                return light_curve

        _iter += 1
        p0_guess = [planet_params['rp_rs'][0]]
        p0_guess.extend(sys_params.tolist())
        popt, pcov = curve_fit(
            model, times[mask], flux[mask], sigma=flux_err[mask],
            p0=p0_guess,
            method='lm')

        perr = np.sqrt(np.diag(pcov))
        rp = popt[0]
        rp_err = perr[0]
        transit_depth = rp**2
        transit_depth_err = rp_err/rp * 2 * transit_depth
        print('Rp/Rs={} +- {}'.format(rp, rp_err))
        print('Transit depth={} +- {}'.format(transit_depth, transit_depth_err))

        opt_model = model(times, *popt, use_mask=False)
        residuals = flux - opt_model
        residuals_precision = np.std(residuals) * 1.e6
        print('Residuals={} ppm'.format(residuals_precision))
        print('Done in {} iterations.'.format(_iter))

        dev_trace = np.abs(residuals) / np.std(residuals)
        dev_trace = np.ma.array(dev_trace, mask=~mask)
        max_deviation_idx = np.ma.argmax(dev_trace)

        if dev_trace[max_deviation_idx] > outlier_threshold:
            mask[max_deviation_idx] = False
            continue
        else:
            break

    if draw_fits:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7))
        ax1.get_shared_x_axes().join(ax1, ax2)
        ax1.errorbar(times[mask], flux[mask], yerr=flux_err[mask], fmt='.')
        ax1.plot(times, opt_model)
        ax2.errorbar(times[mask], flux[mask] - opt_model[mask],
                     yerr=flux_err[mask], fmt='.')
        plt.tight_layout()
        plt.show()

    no_planet_popt = np.copy(popt)
    no_planet_popt[0] = 0.
    sys_model = model(times, *no_planet_popt, use_mask=False)

    fit_dict = {
        "light_curve_model": opt_model,
        "corrected_flux": flux - sys_model + 1,
        "corrected_flux_error": flux_err,
        "systematic_model": sys_model,
        "residual": residuals,
    }

    return transit_depth, transit_depth_err, residuals_precision, \
           popt, fit_dict