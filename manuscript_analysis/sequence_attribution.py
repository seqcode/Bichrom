import numpy as np
from keras import backend as K

# The GradientSaliency class is modified from:
# https://github.com/experiencor/deep-viz-keras/blob/master/saliency.py


class GradientSaliency(object):
    """ Compute saliency masks with gradient."""
    def __init__(self, model, output_index=0):
        # Define the function to compute the gradient
        input_tensors = [model.layers[0].input,  # placeholder: input sequence
                         model.layers[4].input,  # placeholder: input chromatin
                         K.learning_phase()]
        # Taking the gradient w.r.t the sequence
        gradients = model.optimizer.get_gradients(model.output[0][output_index],
                                                  model.layers[0].input)
        self.compute_gradients = K.function(inputs=input_tensors,
                                            outputs=gradients)

    def get_mask(self, input_sequence, chrom_value):
        """ Returns a vanilla gradient mask """
        # Execute the function to compute the gradient
        x_value = np.expand_dims(input_sequence, axis=0)
        c_value = chrom_value
        gradients = self.compute_gradients([x_value, c_value, 0])[0][0]
        return gradients


class IntegratedGradients(GradientSaliency):
    """ Implement the integrated gradients method"""

    def get_mask_integrated(self, input_sequence, chrom_value,
                            input_baseline=None, nsamples=10):
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
            step_gradients = super(IntegratedGradients, self).get_mask(
                input_step_sequence, chrom_value)
            np.add(total_gradients, step_gradients, out=total_gradients,
                   casting='unsafe')
        return total_gradients * input_diff


def get_gradients(input_data, no_of_chromatin_data, model):
    sequence_onehot_bound, chromatin_bound = input_data
    grad_sal = IntegratedGradients(model)

    attribution = []
    for idx in range(sequence_onehot_bound.shape[0]):
        # define the baseline for integrated gradients:
        baseline = np.zeros_like(sequence_onehot_bound)

        grads = grad_sal.get_mask_integrated(
            sequence_onehot_bound[idx],
            chromatin_bound[idx].reshape(-1, (int(no_of_chromatin_data) * 10)),
            input_baseline=baseline[0])
        attribution.append(np.sum(grads, axis=1))
    return np.array(attribution)

