import batman
import emcee
import corner
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count


class MCMCWhiteLightCurve(object):

    def __init__(self, times, flux, flux_err, pre_tilt_x, pre_tilt_y,
                 pst_tilt_x, pst_tilt_y, tilt_idx, u1, u2, u3, u4):

        self.n_cpus = 8
        self.times = times
        self.flux = flux
        self.flux_err = flux_err
        self.pre_tilt_x = pre_tilt_x
        self.pre_tilt_y = pre_tilt_y
        self.pst_tilt_x = pst_tilt_x
        self.pst_tilt_y = pst_tilt_y
        self.tilt_idx = tilt_idx

        self.params = batman.TransitParams()
        self.params.t0 = 59791.112
        self.params.per = 4.05527999  # fixed.
        self.params.rp = 0.146178
        self.params.a = 11.55
        self.params.inc = 87.93
        self.params.ecc = 0.  # fixed.
        self.params.w = 0.  # fixed.
        self.params.u = [u1, u2, u3, u4]  # fixed.
        self.params.limb_dark = "nonlinear"  # fixed.
        self.m = batman.TransitModel(self.params, self.times)

    def sample_white_light_curve(self, draw_fits=False):

        x0 = np.array([59791.112, 0.146142, 11.55, 87.93,
                       0., 0., 0., 0., 0., 0., 0., 0.,
                       1.])
        coords = x0 + 1.e-6 * np.random.randn(64, x0.shape[0])
        print('CPUs available={}, using={}.'.format(cpu_count(), self.n_cpus))
        with Pool(processes=self.n_cpus) as pool:
            sampler = emcee.EnsembleSampler(
                coords.shape[0], coords.shape[1],
                self.log_prob_wlc, pool=pool)
            state = sampler.run_mcmc(coords, 6000, progress=True)
            chain = sampler.get_chain(discard=3000, flat=True)

        emcee_data = az.from_emcee(sampler)
        print(az.summary(emcee_data, round_to=6).to_string())

        popt = np.median(chain, axis=0)
        perr = np.std(chain, axis=0)

        rp = popt[1]
        rp_err = perr[1]
        transit_depth = rp ** 2
        transit_depth_err = rp_err / rp * 2 * transit_depth
        print('Rp/Rs={} +- {}'.format(rp, rp_err))
        print('Transit depth={} +- {}'.format(transit_depth,
                                              transit_depth_err))
        print('t0={}'.format(popt[0]))
        print('a={}'.format(popt[2]))
        print('inc={}'.format(popt[3]))

        opt_model = self.model_wlc(*popt)
        residuals = self.flux - opt_model
        print('Sigmas inflated by {}.'.format(popt[-1]))
        print('Residuals={} ppm'.format(np.std(residuals) * 1.e6))

        if draw_fits:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7))
            ax1.get_shared_x_axes().join(ax1, ax2)
            ax1.errorbar(self.times, self.flux,
                         yerr=self.flux_err, fmt='.')
            ax1.plot(self.times, opt_model)
            ax2.errorbar(self.times, self.flux - opt_model,
                         yerr=self.flux_err, fmt='.')
            plt.tight_layout()
            plt.show()

            var_names = ['t0', 'rp', 'a', 'inc',
                         's1', 's2', 's5', 'l1', 'l2', 'l5',
                         'a1', 'b1',
                         'sigma_beta']
            figure = corner.corner(chain, labels=var_names)
            plt.show()

        t0 = popt[0]
        a = popt[2]
        inc = popt[3]

        no_planet_popt = np.copy(popt)
        no_planet_popt[1] = 0.
        sys_model = self.model_wlc(*no_planet_popt)

        fit_dict = {
            "light_curve_model": opt_model,
            "corrected_flux": self.flux - sys_model + 1,
            "corrected_flux_error": self.flux_err * popt[-1],
            "systematic_model": sys_model,
            "residual": residuals,
        }

        return t0, a, inc, fit_dict

    def log_prob_wlc(self, theta):
        t0, rp, a, inc, s1, s2, s5, l1, l2, l5, a1, b1, sigma_beta = theta

        # Ln prior.
        ln_prior = -0.5 * ((t0 - 59791.112) / 0.01)**2
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
        light_curve = self.model_wlc(*theta)
        inflated_err = self.flux_err * sigma_beta
        ln_like = -0.5 * np.sum(
            (self.flux - light_curve)**2 / inflated_err**2
            + np.log(2 * np.pi * inflated_err**2))

        return ln_like + ln_prior

    def model_wlc(self, t0, rp, a, inc, s1, s2, s5, l1, l2, l5, a1, b1, _):
        # Free physical params.
        self.params.t0 = t0
        self.params.rp = rp
        self.params.a = a
        self.params.inc = inc
        light_curve = self.m.light_curve(self.params)

        # Systematics
        pre_tilt_sys = s1 * self.pre_tilt_x + s2 * self.pre_tilt_y \
                       + s5 * self.pre_tilt_x * self.pre_tilt_y
        pst_tilt_sys = l1 * self.pst_tilt_x + l2 * self.pst_tilt_y \
                       + l5 * self.pst_tilt_x * self.pst_tilt_y
        light_curve[:self.tilt_idx] += pre_tilt_sys
        light_curve[self.tilt_idx:] += pst_tilt_sys

        # Baseline.
        light_curve[:self.tilt_idx] += a1
        light_curve[self.tilt_idx:] += b1

        return light_curve


class MCMCSpecLightCurve(object):

    def __init__(self, times, flux, flux_err, pre_tilt_x, pre_tilt_y,
                 pst_tilt_x, pst_tilt_y, tilt_idx,
                 u1, u2, u3, u4, t0, a, inc):

        self.n_cpus = 8
        self.times = times
        self.flux = flux
        self.flux_err = flux_err
        self.pre_tilt_x = pre_tilt_x
        self.pre_tilt_y = pre_tilt_y
        self.pst_tilt_x = pst_tilt_x
        self.pst_tilt_y = pst_tilt_y
        self.tilt_idx = tilt_idx

        self.params = batman.TransitParams()
        self.params.t0 = t0  # fixed from wlc fit.
        self.params.per = 4.05527999  # fixed.
        self.params.rp = 0.146178
        self.params.a = a  # fixed from wlc fit.
        self.params.inc = inc  # fixed from wlc fit.
        self.params.ecc = 0.  # fixed.
        self.params.w = 0.  # fixed.
        self.params.u = [u1, u2, u3, u4]  # fixed.
        self.params.limb_dark = "nonlinear"  # fixed.
        self.m = batman.TransitModel(self.params, self.times)

    def sample_spec_light_curve(self, draw_fits=False):

        x0 = np.array([0.146142,
                       0., 0., 0., 0., 0., 0., 0., 0.,
                       1.])
        coords = x0 + 1.e-6 * np.random.randn(64, x0.shape[0])
        print('CPUs available={}, using={}.'.format(cpu_count(), self.n_cpus))
        with Pool(processes=self.n_cpus) as pool:
            sampler = emcee.EnsembleSampler(
                coords.shape[0], coords.shape[1],
                self.log_prob_lc, pool=pool)
            state = sampler.run_mcmc(coords, 6000, progress=True)
            chain = sampler.get_chain(discard=3000, flat=True)

        emcee_data = az.from_emcee(sampler)
        print(az.summary(emcee_data, round_to=6).to_string())

        popt = np.median(chain, axis=0)
        perr = np.std(chain, axis=0)

        rp = popt[0]
        rp_err = perr[0]
        transit_depth = rp**2
        transit_depth_err = rp_err/rp * 2 * transit_depth
        print('Rp/Rs={} +- {}'.format(rp, rp_err))
        print('Transit depth={} +- {}'.format(transit_depth, transit_depth_err))

        opt_model = self.model_lc(*popt)
        residuals = self.flux - opt_model
        residuals_precision = np.std(residuals) * 1.e6
        print('Sigmas inflated by {}.'.format(popt[-1]))
        print('Residuals={} ppm'.format(residuals_precision))
        print('Done.')

        if draw_fits:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7))
            ax1.get_shared_x_axes().join(ax1, ax2)
            ax1.errorbar(self.times, self.flux,
                         yerr=self.flux_err, fmt='.')
            ax1.plot(self.times, opt_model)
            ax2.errorbar(self.times, self.flux - opt_model,
                         yerr=self.flux_err, fmt='.')
            plt.tight_layout()
            plt.show()

            var_names = ['rp',
                         's1', 's2', 's5', 'l1', 'l2', 'l5',
                         'a1', 'b1',
                         'sigma_beta']
            figure = corner.corner(chain, labels=var_names)
            plt.show()

        no_planet_popt = np.copy(popt)
        no_planet_popt[0] = 0.
        sys_model = self.model_lc(*no_planet_popt)

        fit_dict = {
            "light_curve_model": opt_model,
            "corrected_flux": self.flux - sys_model + 1,
            "corrected_flux_error": self.flux_err * popt[-1],
            "systematic_model": sys_model,
            "residual": residuals,
        }

        return transit_depth, transit_depth_err, residuals_precision, \
               popt, fit_dict

    def log_prob_lc(self, theta):
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
        light_curve = self.model_lc(*theta)
        inflated_err = self.flux_err * sigma_beta
        ln_like = -0.5 * np.sum(
            (self.flux - light_curve)**2 / inflated_err**2
            + np.log(2 * np.pi * inflated_err**2))

        return ln_like + ln_prior

    def model_lc(self, rp, s1, s2, s5, l1, l2, l5, a1, b1, _):
        # Free physical params.
        self.params.rp = rp
        light_curve = self.m.light_curve(self.params)

        # Systematics
        pre_tilt_sys = s1 * self.pre_tilt_x + s2 * self.pre_tilt_y \
                       + s5 * self.pre_tilt_x * self.pre_tilt_y
        pst_tilt_sys = l1 * self.pst_tilt_x + l2 * self.pst_tilt_y \
                       + l5 * self.pst_tilt_x * self.pst_tilt_y
        light_curve[:self.tilt_idx] += pre_tilt_sys
        light_curve[self.tilt_idx:] += pst_tilt_sys

        # Baseline.
        light_curve[:self.tilt_idx] += a1
        light_curve[self.tilt_idx:] += b1

        return light_curve
