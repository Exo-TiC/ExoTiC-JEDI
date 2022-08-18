import os
import shutil
import numpy as np
from astropy.io import fits
from jwst import datamodels
from jwst.stpipe import Step
from jwst.pipeline.calwebb_spec2 import flat_field_step


class FlatFieldStep(Step):
    """ Get flat field step.

    This steps enables the user to get and save flat field data.

    """

    spec = """
    data_base_name = string(default=None)  # data base name.
    data_chunk_name = string(default=None)  # any data chunk name.
    stage_2_dir = string(default=None)  # directory of stage 2 products.
    trim_col_start = integer(default=5)  # trim columns starts at.
    trim_col_end = integer(default=-5)  # trim columns ends at.
    apply = boolean(default=False)  # apply the flat field.
    """

    def process(self, input):
        """Execute the step.

        Parameters
        ----------
        input: JWST data model
            A data model of type SlitModel.

        Returns
        -------
        JWST data model
            A CubeModel with flat fielding applied if apply=True, unless
            the step is skipped in which case `input_model` is returned.

        """
        with datamodels.open(input) as input_model:

            # Check input model type.
            if not isinstance(input_model, datamodels.SlitModel):
                self.log.error('Input is a {} which was not expected for '
                               'FlatFieldStep, skipping step.'.format(
                                str(type(input_model))))
                return input_model

            # Using stsci flats step.
            stsci_flat_field = flat_field_step.FlatFieldStep()
            stsci_flat_field.call(input_model, save_interpolated_flat=True)

            # Save.
            flat_name = '{}_stage_1_{}.fits'.format(
                self.data_chunk_name, stsci_flat_field.flat_suffix)
            flat_name_new = '{}_stage_2_{}.fits'.format(
                self.data_base_name, stsci_flat_field.flat_suffix)
            shutil.move(flat_name, os.path.join(self.stage_2_dir, flat_name_new))

            if self.apply:
                # NB. update error arrays as per pipeline docs. code snipet below.
                # flat_data_squared = interpolated_flat.data ** 2
                # output_model.var_poisson /= flat_data_squared
                # output_model.var_rnoise /= flat_data_squared
                # output_model.var_flat = output_model.data ** 2 / flat_data_squared * interpolated_flat.err ** 2
                # output_model.err = np.sqrt(
                #     output_model.var_poisson + output_model.var_rnoise + output_model.var_flat
                # )
                self.log.info('Apply not currently implemented, skipping.')

            return input_model
