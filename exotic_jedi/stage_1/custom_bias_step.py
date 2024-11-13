import numpy as np
from jwst import datamodels
from jwst.stpipe import Step


class CustomBiasStep(Step):
    """ Apply custom bias step.

    This steps enables the user to subtract a custom bias image
    formed from the median first groups.

    """

    spec = """
    """

    def process(self, input):
        """Execute the step.

        Parameters
        ----------
        input: JWST data model
            A data model of type CubeModel.
        Returns
        -------
        JWST data model
            A CubeModel with bias subtracted, unless the step
            is skipped in which case `input_model` is returned.

        """
        with datamodels.open(input) as input_model:

            # Copy input model.
            debiased_model = input_model.copy()

            # Check input model type.
            if not isinstance(input_model, datamodels.RampModel):
                self.log.error('Input is a {} which was not expected for '
                               'CustomBiasStep, skipping step.'.format(
                                str(type(input_model))))
                debiased_model.meta.cal_step.custom_debiased = 'SKIPPED'
                return debiased_model

        fg_med_bias = np.nanmedian(debiased_model.data[:, 0, :, :], axis=0)
        debiased_model.data -= fg_med_bias[np.newaxis, np.newaxis, :, :]

        debiased_model.meta.cal_step.custom_debiased = 'COMPLETE'

        return debiased_model
