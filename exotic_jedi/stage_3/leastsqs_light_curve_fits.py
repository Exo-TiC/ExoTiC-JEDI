import batman
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def fit_white_light_curve(times, flux, flux_err, pre_tilt_x, pre_tilt_y,
                          pst_tilt_x, pst_tilt_y, u1, u2, u3, u4, tilt_idx,
                          draw_fits=False):
    params = batman.TransitParams()
    params.t0 = 59791.113
    params.per = 4.05527999  # fixed.
    params.rp = 0.146178
    params.a = 11.55
    params.inc = 87.93
    params.ecc = 0.  # fixed.
    params.w = 0.  # fixed.
    params.u = [u1, u2, u3, u4]  # fixed.
    params.limb_dark = "nonlinear"  # fixed.
    m = batman.TransitModel(params, times)

    def model(_, t0, rp, a, inc, s1, s2, s5, l1, l2, l5, a1, b1):

        # Free physical params.
        params.t0 = t0
        params.rp = rp
        params.a = a
        params.inc = inc
        light_curve = m.light_curve(params)

        # Systematics
        pre_tilt_sys = s1 * pre_tilt_x + s2 * pre_tilt_y \
                       + s5 * pre_tilt_x * pre_tilt_y
        pst_tilt_sys = l1 * pst_tilt_x + l2 * pst_tilt_y \
                       + l5 * pst_tilt_x * pst_tilt_y
        light_curve[:tilt_idx] += pre_tilt_sys
        light_curve[tilt_idx:] += pst_tilt_sys

        # Baseline.
        light_curve[:tilt_idx] += a1
        light_curve[tilt_idx:] += b1

        return light_curve

    popt, pcov = curve_fit(
        model, times, flux, sigma=flux_err,
        p0=[59791.113, 0.146178, 11.55, 87.93,
            0., 0., 0., 0., 0., 0., 0., 0.],
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

    if draw_fits:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7))
        ax1.get_shared_x_axes().join(ax1, ax2)
        ax1.errorbar(times, flux, yerr=flux_err, fmt='.')
        ax1.plot(times, opt_model)
        ax2.errorbar(times, flux - opt_model, yerr=flux_err, fmt='.')
        plt.tight_layout()
        plt.show()

    t0 = popt[0]
    a = popt[2]
    inc = popt[3]

    no_planet_popt = np.copy(popt)
    no_planet_popt[1] = 0.
    sys_model = model(times, *no_planet_popt)

    fit_dict = {
        "light_curve_model": opt_model,
        "corrected_flux": flux - sys_model + 1,
        "corrected_flux_error": flux_err,
        "systematic_model": sys_model,
        "residual": residuals,
    }

    return t0, a, inc, fit_dict


def fit_spec_light_curve(times, flux, flux_err, pre_tilt_x, pre_tilt_y,
                         pst_tilt_x, pst_tilt_y, u1, u2, u3, u4, t0, a, inc,
                         tilt_idx, outlier_threshold=5., draw_fits=False):
    params = batman.TransitParams()
    params.t0 = t0  # fixed from wlc fit.
    params.per = 4.05527999  # fixed.
    params.rp = 0.146178
    params.a = a  # fixed from wlc fit.
    params.inc = inc  # fixed from wlc fit.
    params.ecc = 0.  # fixed.
    params.w = 0.  # fixed.
    params.u = [u1, u2, u3, u4]  # fixed.
    params.limb_dark = "nonlinear"  # fixed.
    m = batman.TransitModel(params, times)

    _iter = 0
    mask = np.ones(times.shape).astype(bool)
    while True:

        def model(_, rp, s1, s2, s5, l1, l2, l5, a1, b1, use_mask=True):

            # Free physical params.
            params.rp = rp
            light_curve = m.light_curve(params)

            # Systematics
            pre_tilt_sys = s1 * pre_tilt_x + s2 * pre_tilt_y \
                           + s5 * pre_tilt_x * pre_tilt_y
            pst_tilt_sys = l1 * pst_tilt_x + l2 * pst_tilt_y \
                           + l5 * pst_tilt_x * pst_tilt_y
            light_curve[:tilt_idx] += pre_tilt_sys
            light_curve[tilt_idx:] += pst_tilt_sys

            # Baseline.
            light_curve[:tilt_idx] += a1
            light_curve[tilt_idx:] += b1

            if use_mask:
                return light_curve[mask]
            else:
                return light_curve

        _iter += 1
        popt, pcov = curve_fit(
            model, times[mask], flux[mask], sigma=flux_err[mask],
            p0=[0.146178, 0., 0., 0., 0., 0., 0., 0., 0.],
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
