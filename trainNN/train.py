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
from train_sc import transfer_and_train_msc
from scan_genome import evaluate_models


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


def return_best_model(loss_vec, model_path):
    # return the model with the lowest validation LOSS
    model_idx = np.argmin(loss_vec)
    # define the model path
    model_file = model_path + 'model_epoch' + str(model_idx) + '.hdf5'
    # load and return the selected model:
    return load_model(model_file)


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
    model_seq = return_best_model(loss_vec=loss, model_path=records_path_seq)
    return model_seq


def run_bimodal_network(train_path, val_path, records_path, no_of_chrom_tracks,
                        base_seq_model):

    # Make an output directory for saving the M-SC models, validation metrics and logs
    records_path_sc = records_path + '.msc/'
    call(['mkdir', records_path_sc])

    curr_params = Params()
    # loss = transfer_and_train_msc(train_path, val_path, no_of_chrom_tracks, base_seq_model,
    #                               batch_size=curr_params.batchsize,
    #                               records_path=records_path_sc)
    loss = np.loadtxt(records_path_sc + 'trainingLoss.txt')
    model_sc = return_best_model(loss_vec=loss, model_path=records_path_sc)
    return model_sc


def main():
    parser = argparse.ArgumentParser(description='Train M-SEQ and M-SC')
    parser.add_argument('train_path', help='Filepath + prefix to the training data')
    parser.add_argument('val_path', help='Filepath + prefix to the validation data')
    parser.add_argument('test_path', help='Filepath + prefix to the test data')
    parser.add_argument('no_of_chrom_tracks', help='Int, number of chromatin data tracks')

    # I'm going to change structure such that I have the models & metric out-files at the same place.
    parser.add_argument("out", help="Filepath or prefix for storing the training metrics")

    args = parser.parse_args()

    # Train the sequence-only network
    mseq = run_seq_network(train_path=args.train_path, val_path=args.val_path, records_path=args.out)

    # Train the bimodal network
    msc = run_bimodal_network(train_path=args.train_path, val_path=args.val_path,
                              records_path=args.out, no_of_chrom_tracks=int(args.no_of_chrom_tracks),
                              base_seq_model=mseq)

    # Evaluate both models on held-out test sets and plot basic metrics
    evaluate_models(sequence_len=500, filename=args.test_path, probas_out_seq=args.out + '.MSEQ_testprobs.txt',
                    probas_out_sc=args.out + 'MSC_testprobs.txt', model_seq=mseq,
                    model_sc=msc, records_file_path=args.out + 'test')


if __name__ == "__main__":
    main()



