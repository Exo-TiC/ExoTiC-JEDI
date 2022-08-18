import os
import numpy as np
from astropy.io import fits
from jwst import datamodels
from jwst.stpipe import Step


class WavelengthMapStep(Step):
    """ Get wavelength map step.

    This steps enables the user to get and save the wavelength map data.

    """

    spec = """
    data_base_name = string(default=None)  # data base name.
    stage_2_dir = string(default=None)  # directory of stage 2 products.
    trim_col_start = integer(default=5)  # trim columns starts at.
    trim_col_end = integer(default=-5)  # trim columns ends at.
    """

    def process(self, input):
        """Execute the step.

        Parameters
        ----------
        input: JWST data model
            A data model of type SlitModel.

        Returns
        -------
        array
            Wavelength map, unless the step is skipped in which case
            `input_model` is returned.

        """
        with datamodels.open(input) as input_model:

            # Check input model type.
            if not isinstance(input_model, datamodels.SlitModel):
                self.log.error('Input is a {} which was not expected for '
                               'WavelengthMapStep, skipping step.'.format(
                                str(type(input_model))))
                return input_model

            # Extract wavelength map.
            self.log.info('Getting wavelength map.')
            row_g, col_g = np.mgrid[0:input_model.data.shape[1],
                                    0:input_model.data.shape[2]]
            wavelength_map = input_model.meta.wcs(
                col_g.ravel(), row_g.ravel())[-1].reshape(
                input_model.data.shape[1:])
            wavelength_map = wavelength_map[
                :, self.trim_col_start:self.trim_col_end]

            # Save.
            hdu = fits.PrimaryHDU(wavelength_map)
            hdul = fits.HDUList([hdu])
            wave_map_name = '{}_stage_2_wavelengthmap.fits'.format(
                self.data_base_name)
            hdul.writeto(os.path.join(
                self.stage_2_dir, wave_map_name), overwrite=True)

        return wavelength_map
