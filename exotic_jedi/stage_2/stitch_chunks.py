import os
import numpy as np
from jwst import datamodels
from jwst.stpipe import Step


class StitchChunksStep(Step):
    """ Stitch chunks step.

    This steps enables the user to stitch together rateimages from
    several chunks, and optionally trim the arrays due to columns
    such as reference pixels.

    """

    spec = """
    stage_1_dir = string(default=None)  # directory of stage 1 products.
    trim = boolean(default=True)  # trim columns or not.
    trim_col_start = integer(default=5)  # trim columns starts at.
    trim_col_end = integer(default=-5)  # trim columns ends at.
    """

    def process(self, chunk_names):
        """Execute the step.

        Parameters
        ----------
        input: list
            rate-image chunk paths.

        Returns
        -------
        JWST data model
            A CubeModel with stitched data, dq, and error arrays, unless
            the step is skipped in which case `input_model` is returned.

        """
        # Build paths.
        chunk_paths = [os.path.join(
            self.stage_1_dir, '{}_stage_1.fits'.format(dcn))
            for dcn in chunk_names]

        # Unpack chunks.
        data_all = []
        dq_all = []
        err_all = []
        var_poisson_all = []
        var_rnoise_all = []
        var_flat_all = []
        for path in chunk_paths:
            with datamodels.open(path) as input_model:
                if not isinstance(input_model, datamodels.CubeModel):
                    self.log.error('Input is a {} which was not expected for '
                                   'StitchChunksStep, skipping chunk.'.format(
                                    str(type(input_model))))

                self.log.info('Stitching chunk {}.'.format(os.path.basename(path)))
                if self.trim:
                    data_all.append(input_model.data[
                        :, :, self.trim_col_start:self.trim_col_end])
                    dq_all.append(input_model.dq[
                        :, :, self.trim_col_start:self.trim_col_end])
                    err_all.append(input_model.err[
                        :, :, self.trim_col_start:self.trim_col_end])
                    var_poisson_all.append(input_model.var_poisson[
                        :, :, self.trim_col_start:self.trim_col_end])
                    var_rnoise_all.append(input_model.var_rnoise[
                        :, :, self.trim_col_start:self.trim_col_end])
                    var_flat_all.append(input_model.var_flat[
                        :, :, self.trim_col_start:self.trim_col_end])
                else:
                    data_all.append(input_model.data)
                    dq_all.append(input_model.dq)
                    err_all.append(input_model.err)
                    var_poisson_all.append(input_model.var_poisson)
                    var_rnoise_all.append(input_model.var_rnoise)
                    var_flat_all.append(input_model.var_flat)

        # Create stitched rate-image.
        dm_all = datamodels.CubeModel()
        dm_all.data = np.concatenate(data_all)
        dm_all.dq = np.concatenate(dq_all)
        dm_all.err = np.concatenate(err_all)
        dm_all.var_poisson = np.concatenate(var_poisson_all)
        dm_all.var_rnoise = np.concatenate(var_rnoise_all)
        dm_all.var_flat = np.concatenate(var_flat_all)
        self.log.info('Stitched datamodel has shape {}.'.format(
            dm_all.data.shape))

        # Update meta.
        dm_all.meta.cal_step.stitched_chunks = 'COMPLETE'

        return dm_all
