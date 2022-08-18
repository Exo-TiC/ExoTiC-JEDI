import numpy as np
from scipy import signal
from jwst import datamodels
from jwst.stpipe import Step
from scipy import interpolate
import matplotlib.pyplot as plt


class AlignSpectraStep(Step):
    """ Align spectra step.

    This steps enables the user align 1d stellar spectra.

    """

    spec = """
    align_spectra = boolean(default=True)  # interpolate spectra based on shifts.
    draw_cross_correlation_fits = boolean(default=False)  # draw cross-correlation function fits.
    draw_trace_positions = boolean(default=False)  # draw trace positions.
    """

    def process(self, input, spec, spec_unc):
        """Execute the step.

        Parameters
        ----------
        input: JWST data model, spectra, and spectra uncertainties

        Returns
        -------
        wavelengths, spectra, and spectra uncertainties

        """
        with datamodels.open(input) as input_model:

            # Check input model type.
            if not isinstance(input_model, datamodels.CubeModel):
                self.log.error('Input is a {} which was not expected for '
                               'AlignSpectraStep, skipping step.'.format(
                                str(type(input_model))))
                return None, None, None, None

            rateimages_cube = input_model.data

        # Find shifts in spectra, dispersion direction.
        x_shifts = []
        aligned_spec = []
        aligned_spec_err = []
        col_pixels = np.arange(spec.shape[1])
        spec_template = np.median(spec, axis=0)
        for s, s_err in zip(spec, spec_unc):
            x_shift = self.cross_correlator(
                s, spec_template, trim_spec=3, high_res_factor=0.005, trim_fit=7)
            x_shifts.append(x_shift)

            if self.align_spectra:
                shifted_pixels = col_pixels - x_shift
                interp_function = interpolate.interp1d(
                    col_pixels, s, kind="linear", fill_value="extrapolate")
                aligned_spec.append(interp_function(shifted_pixels))

                interp_function = interpolate.interp1d(
                    col_pixels, s_err, kind="linear", fill_value="extrapolate")
                aligned_spec_err.append(interp_function(shifted_pixels))
            else:
                aligned_spec.append(s)
                aligned_spec_err.append(s_err)

        aligned_spec = np.array(aligned_spec)
        aligned_spec_err = np.array(aligned_spec_err)

        # Find shifts in psf, cross-dispersion direction.
        y_shifts = []
        psfs = np.sum(rateimages_cube, axis=2)
        psf_template = np.median(psfs, axis=0)
        for p in psfs:
            y_shift = self.cross_correlator(
                p, psf_template, trim_spec=1, high_res_factor=0.005, trim_fit=7)
            y_shifts.append(y_shift)

        x_shifts = np.array(x_shifts)
        y_shifts = np.array(y_shifts)

        if self.draw_trace_positions:
            self._draw_trace_positions(x_shifts, y_shifts)

        return aligned_spec, aligned_spec_err, x_shifts, y_shifts

    def cross_correlator(self, spec, template, trim_spec=3,
                         high_res_factor=0.01, trim_fit=10):
        # Trim x function, otherwise overlap always greatest at zero shift.
        x = np.copy(spec[trim_spec:-trim_spec])
        y = np.copy(template)

        # Interpolate to higher-resolution.
        interp_fx = interpolate.interp1d(
            np.arange(0, x.shape[0]), x, kind='cubic')
        x_hr = interp_fx(np.arange(0, x.shape[0] - 1, high_res_factor))
        interp_fy = interpolate.interp1d(
            np.arange(y.shape[0]), y, kind='cubic')
        y_hr = interp_fy(np.arange(0, x.shape[0] - 1, high_res_factor))

        # Level functions required.
        x_hr -= np.linspace(x_hr[0], x_hr[-1], x_hr.shape[0])
        y_hr -= np.linspace(y_hr[0], y_hr[-1], y_hr.shape[0])

        # Cross-correlate.
        correlation = signal.correlate(x_hr, y_hr, mode="full")
        lags = signal.correlation_lags(x_hr.size, y_hr.size, mode="full")
        coarse_lag_idx = np.argmax(correlation)
        coarse_lag = lags[coarse_lag_idx] * high_res_factor  # Nearest sub pixel shift.

        # Fit parabola.
        trim_lags = lags[coarse_lag_idx - trim_fit:
                         coarse_lag_idx + trim_fit + 1]
        trim_norm_cc = correlation[coarse_lag_idx - trim_fit:
                                   coarse_lag_idx + trim_fit + 1]
        trim_norm_cc -= np.min(trim_norm_cc)
        trim_norm_cc /= np.max(trim_norm_cc)
        p_coeffs = np.polyfit(trim_lags, trim_norm_cc, deg=2)
        lag_parab_hr = -p_coeffs[1] / (2 * p_coeffs[0]) * high_res_factor

        if self.draw_cross_correlation_fits:
            self._draw_cross_correlation_fit(
                trim_lags, trim_norm_cc, p_coeffs, lag_parab_hr, high_res_factor)

        return lag_parab_hr + trim_spec

    def _draw_cross_correlation_fit(self, trim_lags, trim_norm_cc, p_coeffs,
                                    lag_parab_hr, high_res_factor):
        fig, ax1 = plt.subplots(1, 1, figsize=(13, 5))
        ax1.scatter(trim_lags, trim_norm_cc, c='#000000')
        ax1.plot(trim_lags, np.polyval(p_coeffs, trim_lags))
        ax1.scatter(lag_parab_hr / high_res_factor, 1.)
        plt.tight_layout()
        plt.show()

    def _draw_trace_positions(self, x_shifts, y_shifts):
        fig, ax1 = plt.subplots(1, 1, figsize=(13, 5))
        ax1.plot(x_shifts, label='x shifts')
        ax1.plot(y_shifts, label='y shifts')
        ax1.legend()
        plt.tight_layout()
        plt.show()
