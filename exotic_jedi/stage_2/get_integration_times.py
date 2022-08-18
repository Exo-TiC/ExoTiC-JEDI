import os
import numpy as np
from astropy.io import fits
from jwst import datamodels
from jwst.stpipe import Step


class IntegrationTimesStep(Step):
    """ Get integration times step.

    This steps enables the user to get and save the integration times.

    """

    spec = """
    data_base_name = string(default=None)  # data base name.
    stage_1_dir = string(default=None)  # directory of stage 1 products.
    stage_2_dir = string(default=None)  # directory of stage 2 products.
    """

    def process(self, chunk_names):
        """Execute the step.

        Parameters
        ----------
        input: list
            rate-image chunk names.

        Returns
        -------
        array and float
            Array of integration times (BJD TDB) and duration of an
            integration in seconds.

        """
        # Build paths.
        chunk_paths = [os.path.join(
            self.stage_1_dir, '{}_stage_1.fits'.format(dcn))
            for dcn in chunk_names]

        # Extract mid-integration times in BJD TDB.
        mid_int_times = []
        self.log.info('Getting integration times.')
        for path in chunk_paths:
            with datamodels.open(path) as input_model:
                mid_int_times.append(np.array(
                    input_model.int_times['int_mid_BJD_TDB']))

        # Concat times from all chunks.
        mid_int_times = np.concatenate(mid_int_times)

        int_time_s = np.median(np.diff(mid_int_times)) * 24. * 3600.
        self.log.info('Integration duration={} secs'.format(int_time_s))

        # Save.
        hdu = fits.PrimaryHDU(mid_int_times)
        hdul = fits.HDUList([hdu])
        integrations_name = '{}_stage_2_integration_times.fits'.format(
            self.data_base_name)
        hdul.writeto(os.path.join(
            self.stage_2_dir, integrations_name), overwrite=True)

        return mid_int_times, int_time_s
