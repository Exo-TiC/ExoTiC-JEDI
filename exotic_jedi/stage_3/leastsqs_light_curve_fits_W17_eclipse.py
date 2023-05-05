# Created by H.R.Wakeford May 2023
import batman
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def fit_white_light_curve(times, flux, flux_err, x, y, sys_select, planet_params,
                            sys_params=np.array([1.,0.,0.,0.,0.]),
                            outlier_threshold=5.,
                            draw_fits=False):
    
    # sys_params
        # this is a 5 element array of priors for the systematic model
        # sys_params = np.zeros([5])
        # sys_parmas[0] is always the baseline flux level.
        # # Systematic model selection: 'x_y', 'xy', 'ramp'
        #     if (sys_select == 'x_y'):
        #         sys = s0 + (s1 * x) + (s2 * y) + (s3 * times)
        #     if (sys_select == 'xy'):
        #         sys = s0 + (s1 * (x*y)) + (s2 * times)
        #     if (sys_select == 'ramp'):
        #         sys = s0 - np.exp(-s1 * times + s2) + (s3 * times)

    # For a Transit
    # params = batman.TransitParams()
    # params.t0 = planet_params['t0'][0]
    # params.per = planet_params['period'][0]  # fixed.
    # params.fp = planet_params['rp_rs'][0] #rp/rs
    # params.a = planet_params['a_rs'][0] #a/r*
    # params.inc = planet_params['inclination'][0]
    # params.ecc = planet_params['eccentricity'][0]  # fixed.
    # params.w = planet_params['w_omega'][0]  # fixed.
    # # params.u = ld_coefs  # fixed.
    # # params.limb_dark = law  # fixed.
    # m = batman.TransitModel(params, times, transittype="secondary")
    sys_fit = np.empty(len(sys_params), dtype=float)

    # For an Eclipse
    params = batman.TransitParams()
    params.t_secondary = planet_params['t0'][0] #fit
    params.fp = planet_params['fp'][0] #0.003
    params.per = planet_params['period'][0]  # fixed.
    params.rp = planet_params['rp_rs'][0] #rp/rs #0.12462  # DG transit MIRI value.
    params.a = planet_params['a_rs'][0] #a/r*
    params.inc = planet_params['inclination'][0]
    params.ecc = planet_params['eccentricity'][0]  # fixed.
    params.w = planet_params['w_omega'][0]  # fixed.
    params.limb_dark = "uniform"  # redundant.
    params.u = []  # redundant.
    m = batman.TransitModel(params, times, transittype="secondary")
    

    _iter = 0
    mask = np.ones(times.shape).astype(bool)
    while True:

        def model(_, t_secondary, fp, a, inc, s0=None, s1=None, s2=None, s3=None, s4=None, use_mask=True):

            # Free physical params.
            # params.t0 = t0
            params.t_secondary = t_secondary
            params.fp = fp
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
            if (sys_select == 'rampt2'):
                sys = s0 - np.exp(-s1 * times + s2) + (s3*times + s4*times*times)

            light_curve[:] *= sys

            if use_mask:
                return light_curve[mask]
            else:
                return light_curve


        _iter += 1
        p0_guess = [planet_params['t0'][0], planet_params['fp'][0], planet_params['a_rs'][0], planet_params['inclination'][0]]
        p0_guess.extend(sys_params.tolist())
        popt, pcov = curve_fit(
            model, times[mask], flux[mask], sigma=flux_err[mask],
            p0=p0_guess, #[planet_params['t0'][0], planet_params['fp'][0], planet_params['a_rs'][0], planet_params['inclination'][0],
                # s0,s1,s2,s3,s4],
            method='lm', maxfev=20000)

        perr = np.sqrt(np.diag(pcov))
        eclipse_depth = popt[1]
        eclipse_depth_err = perr[1]
        print('Fp/Fs={} +- {}'.format(eclipse_depth, eclipse_depth_err))
        # print('Transit depth={} +- {}'.format(transit_depth, transit_depth_err))
        print('t0={}'.format(popt[0]))
        print('a={}'.format(popt[3]))
        print('inc={}'.format(popt[4]))

        opt_model = model(times, *popt, use_mask=False)
        residuals = flux - opt_model
        print('Residuals={} ppm'.format(np.std(residuals) * 1.e6))
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


def fit_spec_light_curve(times, flux, flux_err, x, y, sys_select,
                         planet_params, t0, a, inc,
                         sys_params=np.array([1.,0.,0.,0.,0.]),
                         outlier_threshold=5., draw_fits=False):
    # params = batman.TransitParams()
    # params.t0 = t0  # fixed from wlc fit.
    # params.per = planet_params['period'][0]  # fixed.
    # params.fp = planet_params['fp_fs'][0]
    # params.a = a  # fixed from wlc fit.
    # params.inc = inc  # fixed from wlc fit.
    # params.ecc = planet_params['eccentricity'][0]  # fixed.
    # params.w = planet_params['w_omega'][0]  # fixed.
    # # params.u = ld_coefs  # fixed.
    # # params.limb_dark = law  # fixed.
    # m = batman.TransitModel(params, times, transittype="secondary")
    
    params = batman.TransitParams()
    params.t_secondary = t0 #Fixed from wlc
    params.fp = planet_params['fp'][0] #Fit
    params.per = planet_params['period'][0]  # fixed.
    params.rp = planet_params['rp_rs'][0] #Fixed #rp/rs #0.12462  # DG transit MIRI value.
    params.a = a #Fixed from wlc
    params.inc = inc #Fixed from wlc
    params.ecc = planet_params['eccentricity'][0]  # fixed.
    params.w = planet_params['w_omega'][0]  # fixed.
    params.limb_dark = "uniform"  # redundant.
    params.u = []  # redundant.
    m = batman.TransitModel(params, times, transittype="secondary")

    _iter = 0
    mask = np.ones(times.shape).astype(bool)
    while True:

        def model(_, fp, s0=None, s1=None, s2=None, s3=None, s4=None, use_mask=True):
            # Free physical params.
            # params.t0 = t0
            params.fp = fp

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
            if (sys_select == 'rampt2'):
                sys = s0 - np.exp(-s1 * times + s2) + (s3*times + s4*times*times)

            light_curve[:] *= sys
            
            if use_mask:
                return light_curve[mask]
            else:
                return light_curve


        _iter += 1
        p0_guess = [planet_params['fp'][0]]
        p0_guess.extend(sys_params.tolist())
        popt, pcov = curve_fit(
            model, times[mask], flux[mask], sigma=flux_err[mask],
            p0=p0_guess,
            method='lm', maxfev=20000)

        perr = np.sqrt(np.diag(pcov))
        eclipse_depth = popt[0]
        eclipse_depth_err = perr[0]
        print('Fp/Fs={} +- {}'.format(eclipse_depth, eclipse_depth_err))
        # print('Transit depth={} +- {}'.format(transit_depth, transit_depth_err))

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
        ax2.errorbar(times[mask], residuals[mask],
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

    return eclipse_depth, eclipse_depth_err, residuals_precision, \
           popt, fit_dict