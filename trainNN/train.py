import argparse
import numpy as np
from subprocess import call
from keras.models import load_model

from train_seq import build_and_train_net
from train_sc import transfer_and_train_msc
from scan_genome import evaluate_models


class Params:
    def __init__(self):
        self.batchsize = 512
        self.dense_layers = 3
        self.n_filters = 256
        self.filter_size = 24
        self.pooling_size = 15
        self.pooling_stride = 15
        self.dropout = 0.5
        self.dense_layer_size = 512


def return_best_model(pr_vec, model_path):
    # return the model with the lowest validation LOSS
    model_idx = np.argmax(pr_vec)
    # define the model path (The model files are 1-based)
    model_file = model_path + 'model_epoch' + str(model_idx + 1) + '.hdf5'
    # load and return the selected model:
    return load_model(model_file)


def run_seq_network(train_path, val_path, records_path):
    """
    Train M-SEQ. (Model Definition in README)
    Parameters:
        train_path (str): Path to the training data.
        val_path (str): Path to the validation data
        records_path (str): Directory & prefix for output directory
    Returns:
        M-SEQ model (Model): A keras model
    """
    # Create an output directory for saving models + per-epoch logs.
    records_path_seq = records_path + '/mseq/'
    call(['mkdir', records_path_seq])

    # current hyper-parameters
    curr_params = Params()

    # train the network
    loss, seq_val_pr = build_and_train_net(curr_params, train_path, val_path,
                               batch_size=curr_params.batchsize,
                               records_path=records_path_seq)
    # choose the model with the lowest validation loss
    model_seq = return_best_model(pr_vec=seq_val_pr, model_path=records_path_seq)
    return model_seq


def run_bimodal_network(train_path, val_path, records_path, no_of_chrom_tracks,
                        base_seq_model):
    """
    Train M-SC. (Model Definition in README)
    Parameters:
        train_path (str): Path to the training data.
        val_path (str): Path to the validation data
        records_path (str): Directory & prefix for output directory
        no_of_chrom_tracks (int): Number of prior chromatin sequencing
        experiments used as input
        base_seq_model: keras model
    Returns:
        M-SC model (keras Model): trained model
    """

    # Create an output directory for saving models + per-epoch logs.
    records_path_sc = records_path + '/msc/'
    call(['mkdir', records_path_sc])

    curr_params = Params()

    # train the network
    loss, bimodal_val_pr = transfer_and_train_msc(train_path, val_path, no_of_chrom_tracks,
                                  base_seq_model,
                                  batch_size=curr_params.batchsize,
                                  records_path=records_path_sc)

    # choose the model with the lowest validation loss
    # loss, bimodal_val_pr = np.loadtxt(records_path_sc + 'trainingLoss.txt')
    model_sc = return_best_model(pr_vec=bimodal_val_pr, model_path=records_path_sc)
    return model_sc


def main():
    parser = argparse.ArgumentParser(description='Train M-SEQ and M-SC')
    parser.add_argument('train_path', help='Path for training data')
    parser.add_argument('val_path', help='Path for validation data')
    parser.add_argument('test_path', help='Path for test data')
    parser.add_argument('no_of_chrom_tracks',
                        help='Number of prior chromatin experiments.')
    parser.add_argument('out', help='Output directory')

    args = parser.parse_args()
    # Create output directory:
    outdir = args.out
    call(['mkdir', outdir])

    # Train the sequence-only network (M-SEQ)
    mseq = run_seq_network(train_path=args.train_path, val_path=args.val_path,
                           records_path=outdir)

    # Train the bimodal network (M-SC)
    msc = run_bimodal_network(train_path=args.train_path,
                              val_path=args.val_path, records_path=outdir,
                              no_of_chrom_tracks=int(args.no_of_chrom_tracks),
                              base_seq_model=mseq)

    # Evaluate both models on held-out test sets and plot metrics
    probas_out_seq = outdir + '/mseq/' + 'testProbs.txt'
    probas_out_sc = outdir + '/msc/' + 'testProbs.txt'
    records_file_path = outdir + '/metrics'
    print records_file_path

    evaluate_models(sequence_len=500, filename=args.test_path,
                    probas_out_seq=probas_out_seq, probas_out_sc=probas_out_sc,
                    model_seq=mseq, model_sc=msc,
                    records_file_path=records_file_path)


if __name__ == "__main__":
    main()



