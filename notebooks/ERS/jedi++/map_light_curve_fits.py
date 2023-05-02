import batman
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def min_white_light_curve(times, flux, flux_err, pre_tilt_x, pre_tilt_y,
                          pst_tilt_x, pst_tilt_y, u1, u2, u3, u4, tilt_idx,
                          draw_fits=False):
    params = batman.TransitParams()
    params.t0 = 59791.112
    params.per = 4.05527999  # fixed.
    params.rp = 0.146178
    params.a = 11.55
    params.inc = 87.93
    params.ecc = 0.  # fixed.
    params.w = 0.  # fixed.
    params.u = [u1, u2, u3, u4]  # fixed.
    params.limb_dark = "nonlinear"  # fixed.
    m = batman.TransitModel(params, times)

    def model(t0, rp, a, inc, s1, s2, s5, l1, l2, l5, a1, b1, _):

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

    def neg_log_prob(theta):
        t0, rp, a, inc, s1, s2, s5, l1, l2, l5, a1, b1, sigma_beta = theta

        # Ln prior.
        ln_prior = -0.5 * ((t0 - 59791.112) / 0.01) ** 2
        ln_prior += -0.5 * ((rp - 0.146142) / 0.01)**2
        ln_prior += -0.5 * ((a - 11.55) / 0.13)**2
        ln_prior += -0.5 * ((inc - 87.93) / 0.14)**2

        ln_prior += -0.5 * (s1 / 0.001)**2
        ln_prior += -0.5 * (s2 / 0.001)**2
        ln_prior += -0.5 * (s5 / 0.001)**2
        ln_prior += -0.5 * (l1 / 0.001)**2
        ln_prior += -0.5 * (l2 / 0.001)**2
        ln_prior += -0.5 * (l5 / 0.001)**2

        ln_prior += -0.5 * (a1 / 0.001)**2
        ln_prior += -0.5 * (b1 / 0.001)**2

        # Ln likelihood.
        light_curve = model(*theta)
        inflated_err = flux_err * sigma_beta
        ln_like = -0.5 * np.sum((flux - light_curve)**2 / inflated_err**2
                                + np.log(2 * np.pi * inflated_err**2))

        return -(ln_like + ln_prior)

    res = minimize(
        neg_log_prob,
        x0=np.array([59791.112, 0.146142,
                     11.55, 87.93,
                     0., 0., 0., 0., 0., 0., 0., 0.,
                     1.]),
        method='BFGS')
    popt = res.x
    perr = np.sqrt(np.diag(res.hess_inv))

    rp = popt[1]
    rp_err = perr[1]
    transit_depth = rp**2
    transit_depth_err = rp_err/rp * 2 * transit_depth
    print('Rp/Rs={} +- {}'.format(rp, rp_err))
    print('Transit depth={} +- {}'.format(transit_depth, transit_depth_err))
    print('t0={}'.format(popt[0]))
    print('a={}'.format(popt[2]))
    print('inc={}'.format(popt[3]))

    opt_model = model(*popt)
    residuals = flux - opt_model
    print('Sigmas inflated by {}.'.format(popt[-1]))
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
    sys_model = model(*no_planet_popt)

    fit_dict = {
        "light_curve_model": opt_model,
        "corrected_flux": flux - sys_model + 1,
        "corrected_flux_error": flux_err * popt[-1],
        "systematic_model": sys_model,
        "residual": residuals,
    }

    return t0, a, inc, fit_dict


def min_spec_light_curve(times, flux, flux_err, pre_tilt_x, pre_tilt_y,
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

        def model(rp, s1, s2, s5, l1, l2, l5, a1, b1, _):

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

            return light_curve

        def neg_log_prob(theta, use_mask=True):
            rp, s1, s2, s5, l1, l2, l5, a1, b1, sigma_beta = theta

            # Ln prior.
            ln_prior = -0.5 * ((rp - 0.146142) / 0.01)**2

            ln_prior += -0.5 * (s1 / 0.001)**2
            ln_prior += -0.5 * (s2 / 0.001)**2
            ln_prior += -0.5 * (s5 / 0.001)**2
            ln_prior += -0.5 * (l1 / 0.001)**2
            ln_prior += -0.5 * (l2 / 0.001)**2
            ln_prior += -0.5 * (l5 / 0.001)**2

            ln_prior += -0.5 * (a1 / 0.001)**2
            ln_prior += -0.5 * (b1 / 0.001)**2

            # Ln likelihood.
            light_curve = model(*theta)
            inflated_err = flux_err * sigma_beta
            ln_like_i = (flux - light_curve) ** 2 / inflated_err ** 2 \
                        + np.log(2 * np.pi * inflated_err ** 2)
            if use_mask:
                ln_like = -0.5 * np.sum(ln_like_i[mask])
            else:
                ln_like = -0.5 * np.sum(ln_like_i)

            return -(ln_like + ln_prior)

        _iter += 1
        res = minimize(
            neg_log_prob,
            x0=np.array([0.146142,
                         0., 0., 0., 0., 0., 0., 0., 0.,
                         1.]),
            method='BFGS')
        popt = res.x
        perr = np.sqrt(np.diag(res.hess_inv))

        rp = popt[0]
        rp_err = perr[0]
        transit_depth = rp**2
        transit_depth_err = rp_err/rp * 2 * transit_depth
        print('Rp/Rs={} +- {}'.format(rp, rp_err))
        print('Transit depth={} +- {}'.format(transit_depth, transit_depth_err))

        opt_model = model(*popt)
        residuals = flux - opt_model
        residuals_precision = np.std(residuals) * 1.e6
        print('Sigmas inflated by {}.'.format(popt[-1]))
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
    sys_model = model(*no_planet_popt)

    fit_dict = {
        "light_curve_model": opt_model,
        "corrected_flux": flux - sys_model + 1,
        "corrected_flux_error": flux_err * popt[-1],
        "systematic_model": sys_model,
        "residual": residuals,
    }

    return transit_depth, transit_depth_err, residuals_precision, \
           popt, fit_dict
