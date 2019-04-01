import numpy as np
from keras import backend as K

# The GradientSaliency class is modified from:
# https://github.com/experiencor/deep-viz-keras/blob/master/saliency.py


class GradientSaliency(object):
    """ Compute saliency masks with gradient."""

    def __init__(self, model, output_index=0):
        # Define the function to compute the gradient
        input_tensors = [model.layers[0].input,  # placeholder for input image tensor
                         model.layers[4].input,  # chromatin input
                         K.learning_phase(),     # placeholder for mode (train or test) tense
                         ]
        # Taking the gradient w.r.t the sequence. (In the presence of chromatin? Should it change? I think so)
        gradients = model.optimizer.get_gradients(model.output[0][output_index], model.layers[0].input)
        self.compute_gradients = K.function(inputs=input_tensors, outputs=gradients)

    def get_mask(self, input_sequence, chrom_value):
        """ Returns a vanilla gradient mask """

        # Execute the function to compute the gradient
        x_value = np.expand_dims(input_sequence, axis=0)  # makes it a 1,500,4 image vector.
        c_value = chrom_value
        gradients = self.compute_gradients([x_value, c_value, 0])[0][0]
        return gradients


class IntegratedGradients(GradientSaliency):
    """ Implement the integrated gradients method"""

    def GetMask(self, input_sequence, chrom_value, input_baseline=None, nsamples=10):
        """Returns an integrated gradients mask"""
        if input_baseline is None:
            input_baseline = np.zeros_like(input_sequence)

        assert input_baseline.shape == input_sequence.shape

        input_diff = input_sequence - input_baseline

        # define a holding vector for the the input sequence.
        total_gradients = np.zeros_like(input_sequence)
        for alpha in np.linspace(0, 1, nsamples):
            input_step_sequence = input_baseline + alpha * input_diff
            input_step_sequence = input_step_sequence.astype('float64')
            step_gradients = super(IntegratedGradients, self).get_mask(input_step_sequence, chrom_value)
            np.add(total_gradients, step_gradients, out=total_gradients, casting='unsafe')

        return total_gradients * input_diff


def random_baseline_attribution(gs, boundX, boundC, no_of_chroms):
    system_attribution = []
    for idx in range(boundX.shape[0]):
        print idx
        baseline = np.zeros_like(boundX) + 0.25
        grads = gs.GetMask(boundX[idx], boundC[idx].reshape(-1, (no_of_chroms * 10)),
                           input_baseline=baseline[0])  # the baseline[0] cause we go one seq at a time.
        attribution = np.sum(grads, axis=1)  # this should be a (500,) vector.
        system_attribution.append(attribution)
    return np.array(system_attribution)


def get_sequence_attribution(datapath, model, input_data, no_of_chrom_datatracks):
    boundX, boundC = input_data
    grad_sal = IntegratedGradients(model)
    rb = random_baseline_attribution(grad_sal, boundX, boundC, no_of_chrom_datatracks)
    return rb
