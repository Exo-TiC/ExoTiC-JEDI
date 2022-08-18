import os
import numpy as np
from astropy.io import fits
from jwst import datamodels
from jwst.stpipe import Step


class GainStep(Step):
    """ Get gain step.

    This steps enables the user to get and save gain data.

    """

    spec = """
    data_base_name = string(default=None)  # data base name.
    data_chunk_name = string(default=None)  # any data chunk name.
    stage_1_dir = string(default=None)  # directory of stage 1 products.
    stage_2_dir = string(default=None)  # directory of stage 2 products.
    trim_col_start = integer(default=5)  # trim columns starts at.
    trim_col_end = integer(default=-5)  # trim columns ends at.
    median_value = boolean(default=False)  # only return median value.
    """

    def process(self, input):
        """Execute the step.

        Parameters
        ----------
        input: JWST data model
            A data model of type SlitModel.

        Returns
        -------
        array or float
            Gain array of float if median requested, unless the step
            is skipped in which case `input_model` is returned.

        """
        with datamodels.open(input) as input_model:

            # Check input model type.
            if not isinstance(input_model, datamodels.SlitModel):
                self.log.error('Input is a {} which was not expected for '
                               'GainStep, skipping step.'.format(
                                str(type(input_model))))
                return input_model

            # Data subarray positions.
            self.log.info('Getting subarray position.')
            stage_1_data_chunk = os.path.join(
                self.stage_1_dir, '{}_stage_1.fits'.format(self.data_chunk_name))
            data_header = fits.getheader(stage_1_data_chunk)
            xstart = data_header['SUBSTRT1']
            ystart = data_header['SUBSTRT2']
            nx = data_header['SUBSIZE1']
            ny = data_header['SUBSIZE2']

            # Extract readnoise.
            gain_filename = self.get_reference_file(input_model, 'gain')
            gain_header = fits.getheader(gain_filename)
            xstart_gain = gain_header['SUBSTRT1']
            ystart_gain = gain_header['SUBSTRT2']
            ystart_trim = ystart - ystart_gain + 1
            xstart_trim = xstart - xstart_gain + 1
            self.log.info('Getting gain.')
            with datamodels.ReadnoiseModel(gain_filename) as gain_model:
                gain_model.data = gain_model.data[
                    ystart_trim:ystart_trim + ny,
                    xstart_trim:xstart_trim + nx].copy()
                gain_model.data = gain_model.data[
                    :, self.trim_col_start:self.trim_col_end].copy()

                # Save.
                gain_model.save(path=os.path.join(
                    self.stage_2_dir, '{}_stage_2_gain.fits'.format(
                        self.data_base_name)))

                # Median value.
                med_gain = np.median(gain_model.data)
                self.log.info('Median gain={} electrons/DN.'.format(med_gain))

                if self.median_value:
                    return float(med_gain)
                else:
                    return gain_model.data
