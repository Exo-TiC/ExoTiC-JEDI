import numpy as np
from jwst import datamodels
from jwst.stpipe import Step
import matplotlib.pyplot as plt


class InspectDQFlagsStep(Step):
    """Inspect dq flags step.

    This steps enables the user to inspect the dq flags present in
    the rate-images. NB. bit values as per:

    `https://jwst-pipeline.readthedocs.io/en/latest/jwst/references
    _general/references_general.html?#data-quality-flags`.

    """

    spec = """
    draw_dq_flags = boolean(default=False)  # draw dq flags.
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
            No changes are made, the `input_model` is returned.

        """
        with datamodels.open(input) as input_model:

            # Check input model type.
            if not isinstance(input_model, datamodels.CubeModel):
                self.log.error('Input is a {} which was not expected for '
                               'InspectDQFlagsStep, skipping step.'.format(
                                str(type(input_model))))
                return input_model

            flags_dict = {0: "DO_NOT_USE", 1: "SATURATED", 2: "JUMP_DET",
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

            # Unpack data.
            data_cube = input.data
            dq_cube = input.dq

            # Find flags.
            flags_int, flags_row, flags_col = np.where(dq_cube != 0)

            # Iterate flags replacing if flags specified by user.
            dq_tesseract_bits = np.zeros(data_cube.shape + (32,))
            for f_int, f_row, f_col in zip(flags_int, flags_row, flags_col):

                # Flag value is a sum of associated flags.
                value_sum = dq_cube[f_int, f_row, f_col]

                # Unpack flag value as array of 32 bits comprising the integer.
                # NB. this array is flipped, value of flag 1 is first bit on the left.
                bit_array = np.flip(np.array(list(
                    np.binary_repr(value_sum, width=32))).astype(int))

                # Track replacements.
                dq_tesseract_bits[f_int, f_row, f_col, :] = bit_array

            # Cleaned metrics.
            total_cleaned = 0
            total_pixels = np.prod(data_cube.shape)
            self.log.info('===== DQ flags info =====')
            for bit_idx in range(32):
                nf_found = int(np.sum(dq_tesseract_bits[:, :, :, bit_idx]))
                self.log.info('Found {} pixels with DQ bit={} name={}.'.format(
                    nf_found, bit_idx, flags_dict[bit_idx]))
                total_cleaned += nf_found
            self.log.info('DQ fraction of total pixels={} %'.format(
                round(total_cleaned / total_pixels * 100., 3)))

            if self.draw_dq_flags:
                for int_idx in range(data_cube.shape[0]):
                    fig, _ = plt.subplots(8, 4, figsize=(9, 7))
                    fig.suptitle('Integration={}'.format(int_idx))
                    for plot_bit, ax in enumerate(fig.axes):
                        ax.imshow(dq_tesseract_bits[int_idx, :, :, plot_bit],
                                  origin='lower', aspect='auto', interpolation='none')
                        ax.text(20, 29, flags_dict[plot_bit],
                                color='#ffffff', fontsize=4)
                        ax.set_xticks([])
                        ax.set_yticks([])
                    plt.tight_layout()
                    plt.subplots_adjust(hspace=0, wspace=0)
                    plt.show()

        return input_model
