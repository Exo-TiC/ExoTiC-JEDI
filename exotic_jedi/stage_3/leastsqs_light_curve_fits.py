import batman
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib import gridspec


def fit_white_light_curve(times, flux, flux_err, x, y, sys_select,
                          ld_coefs, law, planet_params,
                          sys_params=np.array([1.,0.,0.,0.,0.]),
                          outlier_threshold=5., draw_fits=False,
                          print_full_output=False):
    
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

        p0_guess = [planet_params['t0'][0], planet_params['rp_rs'][0], planet_params['a_rs'][0], planet_params['inclination'][0]]
        p0_guess.extend(sys_params.tolist())
        popt, pcov = curve_fit(
            model, times[mask], flux[mask], sigma=flux_err[mask],
            p0=p0_guess,
            method='lm')#, maxfev=20000)

        perr = np.sqrt(np.diag(pcov))
        rp = popt[1]
        rp_err = perr[1]
        transit_depth = rp**2
        transit_depth_err = rp_err/rp * 2 * transit_depth
        print('Rp/Rs={} +- {}'.format(rp, rp_err))
        print('Transit depth={} +- {}'.format(transit_depth, transit_depth_err))
        print()

        opt_model = model(times, *popt, use_mask=False)
        residuals = flux - opt_model
        

        dev_trace = np.abs(residuals) / np.std(residuals)
        dev_trace = np.ma.array(dev_trace, mask=~mask)
        max_deviation_idx = np.ma.argmax(dev_trace)

        if dev_trace[max_deviation_idx] > outlier_threshold:
            mask[max_deviation_idx] = False
            continue
            
        else:
            no_planet_popt = np.copy(popt)
            no_planet_popt[1] = 0.
            sys_model = model(times, *no_planet_popt, use_mask=False)
            print('Rp/Rs={} +- {}'.format(rp, rp_err))
            print('Transit depth={} +- {}'.format(transit_depth, transit_depth_err))
            print('t0={}'.format(popt[0]))
            print('a={}'.format(popt[2]))
            print('inc={}'.format(popt[3]))
            print('Residuals={} ppm'.format(np.std(residuals) * 1.e6))
            print('Done in {} iterations.'.format(_iter))
            break


    if draw_fits:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 7))
        ax1.get_shared_x_axes().join(ax1, ax2, ax3)
        ax1.errorbar(times[mask], flux[mask], yerr=flux_err[mask], fmt='.', zorder=0, alpha=0.4)
        ax1.plot(times, opt_model)
        ax1.set_ylabel("Normalised Flux")
        ax2.set_ylabel("Flux - Systematic Model")
        ax2.errorbar(times[mask], (flux[mask] - sys_model[mask] + 1), yerr=flux_err[mask], fmt='.', zorder=0, alpha=0.4)
        ax2.plot(times[mask], opt_model[mask]-sys_model[mask]+1)
        ax3.set_ylabel("Residuals (ppm)")
        ax3.set_xlabel("Time (BJD)")
        ax3.errorbar(times[mask], (flux[mask] - opt_model[mask])*1e6,
                     yerr=(flux_err[mask])*1e6, fmt='.')
        ax3.axhline(0, ls=':', color='k')
        plt.tight_layout()
        plt.show()

        fig = plt.figure(figsize=(9, 3))
        plots = gridspec.GridSpec(ncols=2, nrows=1,
                                 width_ratios=[4, 1])
        ax1 = fig.add_subplot(plots[0])
        ax2 = fig.add_subplot(plots[1])
        ax1.get_shared_y_axes().join(ax1, ax2)
        ax1.set_ylabel("Residuals (ppm)")
        ax1.set_xlabel("Time (BJD)")
        ax1.errorbar(times[mask], (flux[mask] - opt_model[mask])*1e6,
                     yerr=(flux_err[mask])*1e6, fmt='.')
        ax1.axhline(0, ls=':', color='k')
        ax2.hist((flux[mask]-opt_model[mask])*1e6, orientation='horizontal', bins=15)
        ax2.axhline(0, ls=':', color='k')
        ax2.axis('off')
        plt.tight_layout()
        plt.show()


    t0 = popt[0]
    a = popt[2]
    inc = popt[3]

    t0_err = perr[0]
    a_err = perr[2]
    inc_err = perr[3]

    no_planet_popt = np.copy(popt)
    no_planet_popt[1] = 0.
    sys_model = model(times, *no_planet_popt, use_mask=False)

    sys_fit = popt[4:]
    print('sys_fit values = ', sys_fit)

    fit_dict = {
        "light_curve_model": opt_model,
        "mask": mask,
        "corrected_flux": flux - sys_model + 1,
        "corrected_flux_error": flux_err,
        "systematic_model": sys_model,
        "residual": residuals,
    }

    return t0, a, inc, t0_err, a_err, inc_err, sys_fit, fit_dict


def fit_spec_light_curve(times, flux, flux_err, x, y, sys_select,
                         ld_coefs, law, planet_params, t0, a, inc,
                         sys_params=np.array([1.,0.,0.,0.,0.]),
                         outlier_threshold=5., draw_fits=False,
                         print_full_output=False):
    
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
            if (sys_select == 'ramp2'):
                sys = s0 - np.exp(-s1 * times + s2) + (s3 * times) + (s4*times*times)

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
            method='lm')#, maxfev=20000)

        perr = np.sqrt(np.diag(pcov))
        rp = popt[0]
        rp_err = perr[0]
        transit_depth = rp**2
        transit_depth_err = rp_err/rp * 2 * transit_depth
        if print_full_output==True:
            print('Rp/Rs={} +- {}'.format(rp, rp_err))
            print('Transit depth={} +- {}'.format(transit_depth, transit_depth_err))

        opt_model = model(times, *popt, use_mask=False)
        residuals = flux - opt_model
        residuals_precision = np.std(residuals) * 1.e6
        if print_full_output==True:
            print('Residuals={} ppm'.format(residuals_precision))
            print('Done in {} iterations.'.format(_iter))

        dev_trace = np.abs(residuals) / np.std(residuals)
        dev_trace = np.ma.array(dev_trace, mask=~mask)
        max_deviation_idx = np.ma.argmax(dev_trace)

        if dev_trace[max_deviation_idx] > outlier_threshold:
            mask[max_deviation_idx] = False
            continue
        else:
            print('Rp/Rs={} +- {}'.format(rp, rp_err))
            print('Transit depth={} +- {}'.format(transit_depth, transit_depth_err))

            print('Residuals={} ppm'.format(residuals_precision))
            print('Done in {} iterations.'.format(_iter))

            break


    no_planet_popt = np.copy(popt)
    no_planet_popt[0] = 0.
    sys_model = model(times, *no_planet_popt, use_mask=False)


    if draw_fits:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 7))
        ax1.get_shared_x_axes().join(ax1, ax2, ax3)
        ax1.errorbar(times[mask], flux[mask], yerr=flux_err[mask], fmt='.', zorder=0, alpha=0.4)
        ax1.plot(times, opt_model)
        ax1.set_ylabel("Normalised Flux")
        ax2.set_ylabel("Flux - Systematic Model")
        ax2.errorbar(times[mask], (flux[mask] - sys_model[mask] + 1), yerr=flux_err[mask], fmt='.', zorder=0, alpha=0.4)
        ax2.plot(times[mask], opt_model[mask]-sys_model[mask]+1)
        ax3.set_ylabel("Residuals (ppm)")
        ax3.set_xlabel("Time (BJD)")
        ax3.errorbar(times[mask], (flux[mask] - opt_model[mask])*1e6,
                     yerr=(flux_err[mask])*1e6, fmt='.')
        ax3.axhline(0, ls=':', color='k')
        plt.tight_layout()
        plt.show()

        fig = plt.figure(figsize=(9, 3))
        plots = gridspec.GridSpec(ncols=2, nrows=1,
                                 width_ratios=[4, 1])
        ax1 = fig.add_subplot(plots[0])
        ax2 = fig.add_subplot(plots[1])
        ax1.get_shared_y_axes().join(ax1, ax2)
        ax1.set_ylabel("Residuals (ppm)")
        ax1.set_xlabel("Time (BJD)")
        ax1.errorbar(times[mask], (flux[mask] - opt_model[mask])*1e6,
                     yerr=(flux_err[mask])*1e6, fmt='.')
        ax1.axhline(0, ls=':', color='k')
        ax2.hist((flux[mask]-opt_model[mask])*1e6, orientation='horizontal', bins=15)
        ax2.axhline(0, ls=':', color='k')
        ax2.axis('off')
        plt.tight_layout()
        plt.show()


    no_planet_popt = np.copy(popt)
    no_planet_popt[0] = 0.
    sys_model = model(times, *no_planet_popt, use_mask=False)

    fit_dict = {
        "light_curve_model": opt_model,
        "mask": mask,
        "corrected_flux": flux - sys_model + 1,
        "corrected_flux_error": flux_err,
        "systematic_model": sys_model,
        "residual": residuals,
    }

    return transit_depth, transit_depth_err, residuals_precision, \
           popt, fit_dict