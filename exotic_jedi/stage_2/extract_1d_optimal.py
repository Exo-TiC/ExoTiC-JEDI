import numpy as np
from jwst import datamodels
from jwst.stpipe import Step
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class Extract1DOptimalStep(Step):
    """ Optimal extraction step.

    This steps enables the user extract 1d stellar spectra using
    optimal extraction.

    """

    spec = """
    start_trace_col = integer(default=0)  # start of findable trace by column index.
    end_trace_col = integer(default=2047)  # end of findable trace by column index.
    poly_order = integer(default=2)  # trace polynomial order for fit.
    n_sigma_trace_outlier = float(default=4.)  # number of position sigmas for trace outlier.
    median_spatial_profile = boolean(default=False)  # use median spatial profile.
    spatial_profile_windows = int_list(default=None)  # int_idx windows within which to median spatial profile, eg. [0, 270, 464].
    draw_spatial_profile = boolean(default=False)  # draw spatial profile.
    draw_spectra = boolean(default=False)  # draw extracted spectra.
    """

    def process(self, input, wavelength_map, P, readnoise):
        """Execute the step.

        Parameters
        ----------
        input: JWST data model, wavelength map, spatial profile, and readnoise.
            A data model of type CubeModel, a wavelength map array,
            a spatial profile cube, and a readnoise value.
        Returns
        -------
        wavelengths, spectra, and spectra uncertainties

        """
        with datamodels.open(input) as input_model:

            # Check input model type.
            if not isinstance(input_model, datamodels.CubeModel):
                self.log.error('Input is a {} which was not expected for '
                               'Extract1DOptimalStep, skipping step.'.format(
                                str(type(input_model))))
                return None, None, None

            # Define spectral trace position.
            _, trace_position = self._define_spectral_trace_region(
                input_model.data)

            # Get wavelengths on trace.
            self.log.info('Assigning wavelengths using trace centres.')
            trace_pixels = np.rint(trace_position).astype(int)
            wavelengths = []
            for col_idx, tp in enumerate(trace_pixels):
                wavelengths.append(wavelength_map[tp, col_idx])
            wavelengths = np.array(wavelengths)

            # Extract.
            self.log.info('Optimal extraction in progress.')
            if self.median_spatial_profile:
                if self.spatial_profile_windows is None:
                    P = np.median(P, axis=0)
                    P /= np.sum(P, axis=0)[np.newaxis, :]
                    P = np.broadcast_to(
                        P[np.newaxis, :, :], shape=input_model.data.shape)
                    self.log.info('Using median spatial profile.')
                else:
                    P_concat = []
                    for win_idx in range(len(self.spatial_profile_windows) - 1):
                        win_start = self.spatial_profile_windows[win_idx]
                        win_end = self.spatial_profile_windows[win_idx + 1]
                        P_win = np.median(P[win_start:win_end, :, :], axis=0)
                        P_win /= np.sum(P_win, axis=0)[np.newaxis, :]
                        P_win = np.broadcast_to(
                            P_win[np.newaxis, :, :],
                            shape=input_model.data[win_start:win_end].shape)
                        P_concat.append(P_win)
                        self.log.info('Using median spatial profile from '
                                      'integration {} to {}.'.format(
                                       win_start, win_end))
                    P = np.concatenate(P_concat, axis=0)

            # Iterate integrations.
            fs_opt = []
            var_fs_opt = []
            for int_idx in range(input_model.data.shape[0]):
                integration = input_model.data[int_idx, :, :]
                variance = input_model.err[int_idx, :, :]**2
                spatial_profile = P[int_idx, :, :]

                # Extract standard spectrum.
                f, var_f = self.extract_standard_spectra(
                    integration, variance)

                # Revise variance estimate.
                var_revised = self.revise_variance_estimates(
                    f, spatial_profile, readnoise)

                # Extract optimal spectrum.
                f_opt, var_f_opt = self.extract_optimal_spectrum(
                    integration, spatial_profile, var_revised)
                fs_opt.append(f_opt)
                var_fs_opt.append(var_f_opt)

        fs_opt = np.array(fs_opt)
        var_fs_opt = np.array(var_fs_opt)

        if self.draw_spectra:
            self._draw_extracted_spectra(wavelengths, fs_opt)

        return wavelengths, fs_opt, var_fs_opt**0.5

    def extract_standard_spectra(self, D_S, V):
        """ f and var_f as per Horne 1986 table 1 (step 4). """
        f = np.sum(D_S, axis=0)
        var_f = np.sum(V, axis=0)
        return f, var_f

    def revise_variance_estimates(self, f, P, V_0, S=0., Q=1.):
        """ V revised as per Horne 1986 table 1 (step 6). """
        V_rev = V_0 + np.abs(f[np.newaxis, :] * P + S) / Q
        return V_rev

    def extract_optimal_spectrum(self, D_S, P, V_rev):
        """ f optimal as per Horne 1986 table 1 (step 8). """
        f_opt = np.sum(P * D_S / V_rev, axis=0) / np.sum(P ** 2 / V_rev, axis=0)
        var_f_opt = np.sum(P, axis=0) / np.sum(P ** 2 / V_rev, axis=0)
        return f_opt, var_f_opt

    def _define_spectral_trace_region(self, data_cube):
        # Median stack integrations.
        mstack = np.median(data_cube[:, :, :], axis=0)

        # Find trace position per column with gaussian fits.
        trace_position, trace_sigmas = self._find_trace_position_per_col(
            mstack)

        # Fit polynomial to trace positions.
        poly_trace_position, psf_sigma = self._fit_trace_position(
            trace_position, trace_sigmas)

        return None, poly_trace_position

    def _find_trace_position_per_col(self, mstack, sigma_guess=0.72):
        trace_position = []
        trace_sigmas = []
        row_pixels = np.arange(0, mstack.shape[0], 1)
        for col_idx, col_data in enumerate(mstack.T):

            if not self.start_trace_col < col_idx <= self.end_trace_col:
                trace_position.append(np.nan)
                trace_sigmas.append(np.nan)
                continue

            try:
                popt, pcov = curve_fit(
                    self._amp_gaussian, row_pixels, col_data,
                    p0=[np.max(col_data), row_pixels[np.argmax(col_data)],
                        sigma_guess, 0.], method='lm')
                trace_position.append(popt[1])
                trace_sigmas.append(popt[2])
            except ValueError as err:
                self.log.warn('Gaussian fitting failed, nans present '
                              'for column={}.'.format(col_idx))
                trace_position.append(np.nan)
                trace_sigmas.append(np.nan)
            except RuntimeError as err:
                self.log.warn('Gaussian fitting failed to find optimal trace '
                              'centre for column={}.'.format(col_idx))
                trace_position.append(np.nan)
                trace_sigmas.append(np.nan)

        return np.array(trace_position), np.array(trace_sigmas)

    def _fit_trace_position(self, trace_position, trace_sigmas):
        _iter = 0
        col_pixels = np.arange(0, trace_position.shape[0], 1)
        while True:

            # Update mask.
            nan_mask = np.isfinite(trace_position)

            # Fit polynomial to column.
            try:
                p_coeff = np.polyfit(col_pixels[nan_mask],
                                     trace_position[nan_mask],
                                     deg=self.poly_order)
                poly_trace_position = np.polyval(p_coeff, col_pixels)
            except np.linalg.LinAlgError as err:
                self.log.error('Poly fit error when fitting spectral trace.')
                return None

            # Check residuals to polynomial fit.
            res_trace = trace_position - poly_trace_position
            dev_trace = np.abs(res_trace) / np.nanstd(res_trace)
            max_deviation_idx = np.nanargmax(dev_trace)
            _iter += 1
            if dev_trace[max_deviation_idx] > self.n_sigma_trace_outlier:
                # Outlier: mask and repeat poly fitting.
                trace_position[max_deviation_idx] = np.nan
                trace_sigmas[max_deviation_idx] = np.nan
                continue
            else:
                break

        psf_sigma = np.median(trace_sigmas[nan_mask])
        self.log.info('Trace found in {} iterations.'.format(_iter))
        self.log.info('Trace psf sigma={} pixels.'.format(np.round(psf_sigma, 3)))

        return poly_trace_position, psf_sigma

    def _amp_gaussian(self, x_vals, a, mu, sigma, base=0.):
        y = a * np.exp(-(x_vals - mu)**2 / (2. * sigma**2))
        return base + y

    def _draw_gaussian_fit(self, x_data, y_data, popt, pcov):
        fig, ax1 = plt.subplots(1, 1, figsize=(9, 7))

        # Data and fit.
        ax1.scatter(x_data, y_data, s=10, c='#000000',
                    label='Data')
        xs_hr = np.linspace(np.min(x_data), np.max(x_data), 1000)
        ax1.plot(xs_hr, self._amp_gaussian(
            xs_hr, popt[0], popt[1], popt[2], popt[3]), c='#bc5090',
                 label='Gaussian fit, mean={}.'.format(popt[1]))

        # Gaussian centre and sigma.
        centre = popt[1]
        centre_err = np.sqrt(np.diag(pcov))[1]
        ax1.axvline(centre, ls='--', c='#000000')
        ax1.axvspan(centre - centre_err, centre + centre_err,
                    alpha=0.25, color='#000000')

        ax1.set_xlabel('Row pixels')
        ax1.set_ylabel('DN')
        ax1.set_title('$\mu$={}, and $\sigma$={}.'.format(
            round(popt[1], 3), round(popt[2], 3)))
        plt.tight_layout()
        plt.show()

    def _draw_extracted_spectra(self, wavelengths, spec_box):
        fig, ax1 = plt.subplots(1, 1, figsize=(13, 5))
        for i in range(spec_box.shape[0]):
            ax1.plot(wavelengths, spec_box[i, :], c='#bc5090', alpha=0.02)
        ax1.set_ylabel('Electrons')
        ax1.set_xlabel('Wavelength')
        plt.tight_layout()
        plt.show()
