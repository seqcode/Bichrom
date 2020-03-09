"""
Top-level function to train, test and plot performance metrics for:
- sequence-only network
- bimodal sequence-chromatin network

The module performs the following functions in order:

1. Train a sequence-only network.
2. Train a sequence and prior chromatin network.
3. Test both networks and plot metrics.
"""

import argparse
import numpy as np
from subprocess import call
from keras.models import load_model

from train_seq import build_and_train_net


class Params:
    def __init__(self):
        self.batchsize = 20
        self.dense_layers = 3
        self.n_filters = 128
        self.filter_size = 24
        self.pooling_size = 8
        self.pooling_stride = 8
        self.dropout = 0.5
        self.dense_layer_size = 128


def run_seq_network(train_path, val_path, records_path):
    """
    Train M-SEQ. (Model Definition in README)
    Save the model loss at each epoch & return the model with the lowest validation LOSS.

    :param train_path: Path to the training data
    :param val_path: Path to the validation data
    :param records_path: Directory & prefix for output directory
    :return: Trained M-SEQ model.
    """

    # Make an output directory for saving the sequence models, validation metrics and logs.
    records_path_seq = records_path + '.mseq/'
    call(['mkdir', records_path_seq])
    # Params: current network parameters (Implement option to choose these using a random grid search)
    curr_params = Params()

    # Training the network!
    loss = build_and_train_net(curr_params, train_path, val_path,
                               batch_size=curr_params.batchsize, records_path=records_path_seq)

    # return the model with the lowest validation LOSS
    model_idx = np.argmin(loss)
    # define the model path
    model_file = records_path_seq + 'model_epoch' + str(model_idx) + '.hdf5'
    # load and return the selected model:
    return load_model(model_file)


def run_bimodal_network(base_seq_model):
    pass


def main():
    parser = argparse.ArgumentParser(description='Train M-SEQ and M-SC')
    parser.add_argument('train_path', help='Filepath + prefix to the training data')
    parser.add_argument('val_path', help='Filepath + prefix to the validation data')
    parser.add_argument('test_path', help='Filepath + prefix to the test data')
    # I'm going to change structure such that I have the models & metric out-files at the same place.
    parser.add_argument("out", help="Filepath or prefix for storing the training metrics")

    args = parser.parse_args()

    # Train the sequence-only network
    run_seq_network(train_path=args.train_path, val_path=args.val_path, records_path=args.out)


if __name__ == "__main__":
    main()



