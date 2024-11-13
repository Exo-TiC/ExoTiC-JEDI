import numpy as np
from jwst import datamodels
from jwst.stpipe import Step
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.timeseries import LombScargle


class DestripingGroupsStep(Step):
    """ Destriping group level images.

    This steps enables the user to `destripe` the data during
    stage 1, at the group level. This removes the background
    and combats 1/f noise.

    """

    spec = """
    start_trace_col = integer(default=0)  # start of findable trace by column index.
    end_trace_col = integer(default=2047)  # end of findable trace by column index.
    poly_order = integer(default=2)  # trace polynomial order for fit.
    n_sigma_trace_mask = float(default=10.)  # number of psf sigmas to mask around trace.
    n_sigma_trace_outlier = float(default=4.)  # number of position sigmas for trace outlier.
    dq_bits = int_list(default=None)  # dq flags to mask in column median-ing, bit values.
    keep_mean_bkd_level = boolean(default=False)  # add mean background level back on after destriping.
    draw_psf_fits = boolean(default=False)  # draw gauss fits to each column.
    draw_trace_position = boolean(default=False)  # draw trace fits and position.
    draw_mask = boolean(default=False)  # draw trace and dq flags mask.
    draw_bkg_medians = boolean(default=False)  # draw bkg medians.
    draw_bkg_pixels_power_spectrum = boolean(default=False)  # draw bkg pixels power spectrum.
    """

    def process(self, input):
        """Execute the step.

        Parameters
        ----------
        input: JWST data model
            A data model of type RampModel.
        Returns
        -------
        JWST data model
            A RampModel with `destriped` groups, unless the
            step is skipped in which case `input_model` is returned.

        """
        with datamodels.open(input) as input_model:

            # Copy input model.
            destriped_model = input_model.copy()

            # Check input model type.
            if not isinstance(input_model, datamodels.RampModel):
                self.log.error('Input is a {} which was not expected for '
                               'DestripingGroupsStep, skipping step.'.format(
                                str(type(input_model))))
                destriped_model.meta.cal_step.destriped_groups = 'SKIPPED'
                return destriped_model

            # Define mask of spectral trace region.
            trace_mask = self._define_spectral_trace_region(
                input_model.data)

            # Get mask of user specified dq flags.
            self.dq_bits = np.unique(self.dq_bits)
            dq_pixel_mask = self._select_pixel_flags(input_model.pixeldq)
            dq_group_mask = self._select_group_flags(input_model.groupdq)

            # Combine mask uniformly to all groups in an integration.
            dq_mask = np.logical_or.reduce(dq_group_mask, axis=1)
            dq_mask = np.logical_or(dq_mask, dq_pixel_mask[np.newaxis, :, :])
            trace_dq_mask = np.logical_or(dq_mask, trace_mask[np.newaxis, :, :])
            mask_tesseract = np.broadcast_to(
                trace_dq_mask[:, np.newaxis, :, :], shape=input_model.data.shape)
            if self.draw_mask:
                self._draw_mask(input_model.data, trace_dq_mask)

            # Destripe per group: subtract median bkg values per column.
            self.log.info('Destriping in progress.')
            col_bkg_medians = np.ma.median(np.ma.array(
                input_model.data, mask=mask_tesseract), axis=2)
            destriped_model.data -= col_bkg_medians[:, :, np.newaxis, :]
            if self.draw_bkg_medians:
                self._draw_bkg_medians(col_bkg_medians)

            if self.keep_mean_bkd_level:
                # Optional, add mean background level back on. This
                # may otherwise affect ramp fitting noise profiles?
                self._add_mean_bkd_ramps(input_model.data,
                                         destriped_model.data,
                                         mask_tesseract)

            if self.draw_bkg_pixels_power_spectrum:
                self._draw_bkg_pixels_power_spectrum(input_model.data,
                                                     destriped_model.data,
                                                     mask_tesseract)

            # Update meta.
            destriped_model.meta.cal_step.destriped_groups = 'COMPLETE'

        return destriped_model

    def _define_spectral_trace_region(self, data_tesseract):
        # Median stack last group of each integration.
        mstack = np.nanmedian(data_tesseract[:, -1, :, :], axis=0)

        # Find trace position per column with gaussian fits.
        trace_position, trace_sigmas = self._find_trace_position_per_col(
            mstack)

        # Fit polynomial to trace positions.
        poly_trace_position, psf_sigma = self._fit_trace_position(
            trace_position, trace_sigmas)

        #  Define masked region.
        row_pixels = np.arange(0, mstack.shape[0], 1)
        col_pixels = np.arange(0, trace_position.shape[0], 1)
        xx, yy = np.meshgrid(col_pixels, row_pixels)
        mask_bottom_edge = np.rint(
            poly_trace_position - psf_sigma * self.n_sigma_trace_mask)
        mask_top_edge = np.rint(
            poly_trace_position + psf_sigma * self.n_sigma_trace_mask)
        trace_mask = (yy >= mask_bottom_edge[np.newaxis, ...]) \
                     & (yy <= mask_top_edge[np.newaxis, ...])
        self.log.info('Trace mask made.')
        if self.draw_trace_position:
            self._draw_trace_position(trace_position, poly_trace_position,
                                      trace_sigmas, psf_sigma, mstack,
                                      mask_bottom_edge, mask_top_edge)

        return trace_mask

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
                if self.draw_psf_fits:
                    self._draw_gaussian_fit(row_pixels, col_data, popt, pcov)
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

    def _select_pixel_flags(self, pixel_dq):
        # Find flags.
        p_flags_row, p_flags_col = np.where(pixel_dq != 0)

        # Iterate flags selecting if flags specified by user.
        selected_mask = np.full(pixel_dq.shape, False)
        dq_list_bits = []
        for f_row, f_col in zip(p_flags_row, p_flags_col):

            # Flag value is a sum of associated flags.
            value_sum = pixel_dq[f_row, f_col]

            # Unpack flag value as array of 32 bits comprising the integer.
            # NB. this array is flipped, value of flag 1 is first bit on the left.
            bit_array = np.flip(np.array(list(
                np.binary_repr(value_sum, width=32))).astype(int))

            # Check if any user specified flags are present.
            if np.any((bit_array[self.dq_bits]).astype(int)):
                # Select as masked.
                selected_mask[f_row, f_col] = True

            # Track selections.
            dq_list_bits.extend(np.where(bit_array)[0].tolist())

        # Selected metrics.
        total_selected = 0
        total_pixels = np.prod(pixel_dq.shape)
        dq_list_bits = np.array(dq_list_bits).astype(int)
        self.log.info('===== Selected pixel dq flags for mask info =====')
        for bit_idx in self.dq_bits:
            nf_select = int(np.sum(dq_list_bits == bit_idx))
            self.log.info('Selected {} pixels with DQ bit={} name={}.'
                .format(nf_select, bit_idx, self.flags_dict[bit_idx]))
            total_selected += nf_select
        self.log.info('Total pixel fraction selected={} %'.format(
            round(total_selected / total_pixels * 100., 3)))

        # Not cleaned metrics.
        self.log.info('===== Other pixel dq flags found but not selected =====')
        for bit_idx in range(32):
            if bit_idx in self.dq_bits:
                continue
            nf_not_select = int(np.sum(dq_list_bits == bit_idx))
            self.log.info('Did not select {} pixels with DQ bit={} name={}.'
                .format(nf_not_select, bit_idx, self.flags_dict[bit_idx]))

        return selected_mask

    def _select_group_flags(self, group_dq):
        # Find flags.
        g_flags_int, g_flags_group, g_flags_row, g_flags_col = \
            np.where(group_dq != 0)

        # Iterate flags selecting if flags specified by user.
        selected_mask = np.full(group_dq.shape, False)
        dq_list_bits = []
        for f_int, f_grp, f_row, f_col in zip(g_flags_int, g_flags_group,
                                              g_flags_row, g_flags_col):

            # Flag value is a sum of associated flags.
            value_sum = group_dq[f_int, f_grp, f_row, f_col]

            # Unpack flag value as array of 32 bits comprising the integer.
            # NB. this array is flipped, value of flag 1 is first bit on the left.
            bit_array = np.flip(np.array(list(
                np.binary_repr(value_sum, width=32))).astype(int))

            # Check if any user specified flags are present.
            if np.any((bit_array[self.dq_bits]).astype(int)):
                # Select as masked.
                selected_mask[f_int, f_grp, f_row, f_col] = True

            # Track selections.
            dq_list_bits.extend(np.where(bit_array)[0].tolist())

        # Selected metrics.
        total_selected = 0
        total_pixels = np.prod(group_dq.shape)
        dq_list_bits = np.array(dq_list_bits).astype(int)
        self.log.info('===== Selected group dq flags for mask info =====')
        for bit_idx in self.dq_bits:
            nf_select = int(np.sum(dq_list_bits == bit_idx))
            self.log.info('Selected {} pixels with DQ bit={} name={}.'
                .format(nf_select, bit_idx, self.flags_dict[bit_idx]))
            total_selected += nf_select
        self.log.info('Total pixel fraction selected={} %'.format(
            round(total_selected / total_pixels * 100., 3)))

        # Not cleaned metrics.
        self.log.info('===== Other group dq flags found but not selected =====')
        for bit_idx in range(32):
            if bit_idx in self.dq_bits:
                continue
            nf_not_select = int(np.sum(dq_list_bits == bit_idx))
            self.log.info('Did not select {} pixels with DQ bit={} name={}.'
                .format(nf_not_select, bit_idx, self.flags_dict[bit_idx]))

        return selected_mask

    @property
    def flags_dict(self):
        return {0: "DO_NOT_USE", 1: "SATURATED", 2: "JUMP_DET",
                3: "DROPOUT", 4: "OUTLIER", 5: "PERSISTENCE",
                6: "AD_FLOOR", 7: "RESERVED", 8: "UNRELIABLE_ERROR",
                9: "NON_SCIENCE", 10: "DEAD", 11: "HOT", 12: "WARM",
                13: "LOW_QE", 14: "RC", 15: "TELEGRAPH", 16: "NONLINEAR",
                17: "BAD_REF_PIXEL", 18: "NO_FLAT_FIELD", 19: "NO_GAIN_VALUE",
                20: "NO_LIN_CORR", 21: "NO_SAT_CHECK", 22: "UNRELIABLE_BIAS",
                23: "UNRELIABLE_DARK", 24: "UNRELIABLE_SLOPE",
                25: "UNRELIABLE_FLAT", 26: "OPEN", 27: "ADJ_OPEN",
                28: "UNRELIABLE_RESET", 29: "MSA_FAILED_OPEN",
                30: "OTHER_BAD_PIXEL", 31: "REFERENCE_PIXEL"}

    def _add_mean_bkd_ramps(self, input_data, destriped_data, mask_tesseract):
        first_group_mean_level = np.ma.median(np.ma.array(
            input_data[:, 0, :, :], mask=mask_tesseract[:, 0, :, :]))
        final_group_mean_level = np.ma.median(np.ma.array(
            input_data[:, -1, :, :], mask=mask_tesseract[:, -1, :, :]))
        mean_level = np.linspace(first_group_mean_level,
                                 final_group_mean_level,
                                 input_data.shape[1])
        destriped_data += mean_level[np.newaxis, :, np.newaxis, np.newaxis]
        self.log.info('Bkg ramp added back in with gradient={} DN/group.'.format(
            np.round((mean_level[-1] - mean_level[0]) / mean_level.shape[0], 3)))

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

    def _draw_trace_position(self, trace_position, poly_trace_position,
                             trace_sigmas, psf_sigma, mstack,
                             mask_bottom_edge, mask_top_edge):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 7))
        ax1.get_shared_x_axes().join(ax1, ax2, ax3)

        # Poly fit to trace location.
        ax1.scatter(np.arange(trace_position.shape[0]), trace_position,
                    c='#000000', s=5)
        ax1.plot(np.arange(trace_position.shape[0]), poly_trace_position,
                 c='#bc5090')

        # Trace widths.
        ax2.scatter(np.arange(trace_position.shape[0]), trace_sigmas,
                    c='#000000', s=5)
        ax2.axhline(psf_sigma, ls='--', c='#bc5090')

        # Trace mask.
        im = mstack
        ax3.imshow(im, origin='lower', aspect='auto', interpolation='none',
                   vmin=np.percentile(im.ravel(), 1.),
                   vmax=np.percentile(im.ravel(), 99.))
        ax3.plot(np.arange(mask_bottom_edge.shape[0]), mask_bottom_edge,
                 c='#bc5090')
        ax3.plot(np.arange(mask_top_edge.shape[0]), mask_top_edge,
                 c='#bc5090')

        ax1.set_ylabel('Trace centre / pixels')
        ax2.set_ylabel('Trace sigma / pixels')
        ax3.set_xlabel('Col pixels')
        plt.tight_layout()
        plt.show()

    def _draw_mask(self, data_tesseract, trace_dq_mask):
        for int_idx in range(data_tesseract.shape[0]):
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 7))
            ax1.get_shared_x_axes().join(ax1, ax2, ax3)
            ax1.get_shared_y_axes().join(ax1, ax2)

            # Data.
            im = data_tesseract[int_idx, -1, :, :]
            ax1.imshow(im, origin='lower', aspect='auto', interpolation='none',
                       vmin=np.percentile(im.ravel(), 1.),
                       vmax=np.percentile(im.ravel(), 99.))

            # Mask.
            im = trace_dq_mask[int_idx]
            ax2.imshow(im, origin='lower', aspect='auto', interpolation='none')

            # Number of good pixels for median calculation.
            ax3.plot(np.sum(~trace_dq_mask[int_idx], axis=0))
            ax3.set_ylim(0, trace_dq_mask[int_idx].shape[0])

            ax1.set_title('Integration={}/{}.'.format(
                int_idx, data_tesseract.shape[0]))
            ax1.set_ylabel('Row pixels')
            ax2.set_ylabel('Row pixels')
            ax3.set_ylabel('Number of good pixels for median-ing')
            ax3.set_xlabel('Col pixels')
            plt.tight_layout()
            plt.show()

    def _draw_bkg_medians(self, col_bkg_medians):
        col_pixels = np.arange(col_bkg_medians.shape[2])
        for int_idx, int_medians in enumerate(col_bkg_medians):
            for group_idx, group_medians in enumerate(int_medians):
                fig, ax1 = plt.subplots(1, 1, figsize=(9, 7))
                ax1.plot(col_pixels, group_medians)
                ax1.set_title('Integration={}/{}, group={}/{}.'.format(
                    int_idx, col_bkg_medians.shape[0],
                    group_idx, col_bkg_medians.shape[1]))
                ax1.set_ylabel('Column bkg median')
                ax1.set_xlabel('Col pixels')
                plt.tight_layout()
                plt.show()

    def _draw_bkg_pixels_power_spectrum(self, original_data, destriped_data,
                                        trace_tesseract):
        fig, ax1 = plt.subplots(1, 1, figsize=(9, 7))

        # Define frequency grid of appropriate timescales.
        pixel_read_time = 10. * 1.e-6  # us.
        col_wait_time = 120. * 1.e-6  # us.
        frame_time = (original_data.shape[2] * pixel_read_time + col_wait_time) \
                     * original_data.shape[3]
        freqs = np.logspace(np.log10(1. / frame_time),
                            np.log10(1. / pixel_read_time), 100)

        # Compute power spectra.
        original_ps = self._median_power_spectrum(
            original_data, trace_tesseract, freqs, pixel_read_time, col_wait_time)
        destriped_ps = self._median_power_spectrum(
            destriped_data, trace_tesseract, freqs, pixel_read_time, col_wait_time)

        ax1.plot(freqs, original_ps, label='Bkg pixels')
        ax1.plot(freqs, destriped_ps, label='Destriped bkg pixels')

        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel("Frequency / Hz")
        ax1.set_ylabel("Power")
        ax1.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

    def _median_power_spectrum(self, data_tesseract, trace_tesseract, freqs,
                               pixel_read_time, col_wait_time):
        power = []
        for int_idx, data_cube in enumerate(data_tesseract):
            for grp_idx, im in enumerate(data_cube):
                time = 0.
                read_times = []
                pixel_counts = []
                for j in range(im.shape[1]):
                    for i in range(im.shape[0]):
                        if not trace_tesseract[int_idx, grp_idx, i, j]:
                            read_times.append(time)
                            pixel_counts.append(im[i, j])
                        time += pixel_read_time
                    time += col_wait_time
                pwr = LombScargle(read_times, pixel_counts).power(freqs)
                power.append(pwr)

        return np.nanmedian(power, axis=0)
