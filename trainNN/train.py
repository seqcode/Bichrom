import argparse
from asyncio import subprocess
from json import load
import numpy as np
import yaml
from subprocess import call
from tensorflow.keras.models import load_model

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
    return model_file


def run_seq_network(h5file, train_path, val_path, records_path, seq_len):
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
    records_path_seq = records_path + '/seqnet/'
    call(['mkdir', records_path_seq])

    # current hyper-parameters
    curr_params = Params()

    # train the network
    loss, seq_val_pr = build_and_train_net(curr_params, h5file, train_path, val_path,
                                           batch_size=curr_params.batchsize,
                                           records_path=records_path_seq,
                                           seq_len=seq_len)
    # choose the model with the lowest validation loss
    model_seq_path = return_best_model(pr_vec=seq_val_pr, model_path=records_path_seq)
    return model_seq_path


def run_bimodal_network(h5file, train_path, val_path, records_path, base_seq_model_path,
                        bin_size, seq_len):
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
    records_path_sc = records_path + '/bichrom/'
    call(['mkdir', records_path_sc])

    curr_params = Params()

    # train the network
    loss, bimodal_val_pr = transfer_and_train_msc(h5file, train_path, val_path,
                                                  base_seq_model_path,
                                                  batch_size=curr_params.batchsize,
                                                  records_path=records_path_sc,
                                                  bin_size=bin_size,
                                                  seq_len=seq_len)

    # choose the model with the lowest validation loss
    # loss, bimodal_val_pr = np.loadtxt(records_path_sc + 'trainingLoss.txt')
    model_sc = return_best_model(pr_vec=bimodal_val_pr, model_path=records_path_sc)
    return model_sc


def train_bichrom(h5file, data_paths, outdir, seq_len, bin_size):
    # Train the sequence-only network (M-SEQ)
    mseq_path = run_seq_network(h5file=h5file,  train_path=data_paths['train'], val_path=data_paths['val'],
                           records_path=outdir, seq_len=seq_len)

    no_of_chromatin_tracks = len(data_paths['train']['chromatin_tracks'])
    # Train the bimodal network (M-SC)
    msc_path = run_bimodal_network(h5file=h5file, train_path=data_paths['train'],
                              val_path=data_paths['val'], records_path=outdir,
                              base_seq_model_path=mseq_path, bin_size=bin_size, seq_len=seq_len)

    # Evaluate both models on held-out test sets and plot metrics
    probas_out_seq = outdir + '/seqnet/' + 'test_probs.txt'
    probas_out_sc = outdir + '/bichrom/' + 'test_probs.txt'
    records_file_path = outdir + '/metrics'
    print(records_file_path)
    # save the best msc model
    call(['cp', msc_path, outdir + '/full_model.best.hdf5'])

    mseq = load_model(mseq_path)
    msc = load_model(msc_path)
    evaluate_models(sequence_len=seq_len, h5file=h5file, path=data_paths['test'],
                    probas_out_seq=probas_out_seq, probas_out_sc=probas_out_sc,
                    model_seq=mseq, model_sc=msc,
                    records_file_path=records_file_path)
