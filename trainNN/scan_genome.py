"""
Iterate over the whole genome (or test chromosome) to measure genome-wide performance of a trained
neural network.
Current Status: Testing for both M-SEQ and M-SC.
"""

from __future__ import division
import sys
import numpy as np
import sklearn.metrics
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from helper import plot_distributions

# user defined module
import iterutils as iu


def merge_generators(filename, batchsize, seqlen, mode):
    X = iu.train_generator(filename + ".seq", batchsize,
                           seqlen, "seq", mode)
    A = iu.train_generator(filename + ".chromtracks", batchsize,
                           seqlen, "accessibility", mode)
    y = iu.train_generator(filename + ".labels", batchsize,
                           seqlen, "labels", mode)
    while True:
        yield [X.next(), A.next()], y.next()


def test_on_batch(batch_generator, model, outfile, mode):
    """
    Get probabilities for each test data point.
    The reason that this is implemented in a batch is because \
    the whole genome cannot be loaded without batching.

    Single held-out chromosomes can be tested directly.

    Parameters:
        batch_generator: generator
            a generator that yields sequence, chromatin and label vectors.
        model: Model
            A trained Keras model
        outfile: str
            The outfile used for storing probabilities.
    Returns: None
        (Saves an output file with the probabilities for the test set )
    """
    counter = 0
    while True:
        try:
            [X_test, acc_test], y = batch_generator.next()
            if mode == 'seq_only':
                batch_probas = model.predict_on_batch([X_test])
            else:
                batch_probas = model.predict_on_batch([X_test, acc_test])
            # saving to file: 
            with open(outfile, "a") as fh:
                np.savetxt(fh, batch_probas)
            counter += 1
        except StopIteration:
            break


def get_metrics(test_labels, test_probas, records_file):
    """
    Takes the test labels and test probabilities, and calculates
    plots the following:
        a. P-R Curves
        b. auPRC
        c. auROC
        d. Posterior Distributions of the Recall at FPR=0.01

    Parameters:
        test_labels: ndarray
            n * 1 vector with the true labels ( 0 or 1 )
        test_probas: ndarray
            n * 1 vector with the network probabilities
    Returns:
         None
    """
    # Calculate auROC
    roc_auc = sklearn.metrics.roc_auc_score(test_labels, test_probas)
    # Calculate auPRC
    prc_auc = sklearn.metrics.average_precision_score(test_labels, test_probas)

    # Write auROC and auPRC to records file
    records_file.write("AUC ROC:{0}".format(roc_auc))
    records_file.write("AUC PRC:{0}".format(prc_auc))


def get_probabilities(filename, seq_len, model, outfile, mode):
    """
    Get network-assigned probabilities

    Parameters:
        filename: str
            Input file to be loaded
        seq_len: int
            Length of input DNA sequence
    Returns:
         probas: ndarray
            An array of probabilities for the test set
         true labels: ndarray
    """
    # Inputing a range of default values here, can be changed later.
    data_generator = merge_generators(filename=filename, batchsize=1000,
                                      seqlen=seq_len, mode='nr')
    # Load the keras model
    # model = load_model(model_file)
    test_on_batch(data_generator, model, outfile, mode)
    probas = np.loadtxt(outfile)
    true_labels = np.loadtxt(filename + '.labels')
    return true_labels, probas


def plot_pr_curve(test_labels, test_probas, color):
    # Get the PR values:
    precision, recall, _ = precision_recall_curve(y_true=test_labels,
                                                  probas_pred=test_probas)
    plt.plot(recall, precision, c=color, lw=2.5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)


def combine_pr_curves(records_file, m_seq_probas, m_sc_probas, labels):
    plot_pr_curve(labels, m_seq_probas, color='#F1C40F')
    plot_pr_curve(labels, m_sc_probas, color='#2471A3')
    plt.savefig(records_file + '.pr_curves.pdf')


def evaluate_models(sequence_len, filename, probas_out_seq, probas_out_sc,
                    model_seq, model_sc, records_file_path):

    # Define the file that contains testing metrics
    records_files = open(records_file_path, "w")

    # Get the probabilities for both M-SEQ and M-SC models:
    # Note: Labels are the same for M-SC and M-SEQ
    true_labels, probas_seq = get_probabilities(filename=filename,
                                                seq_len=sequence_len,
                                                model=model_seq,
                                                outfile=probas_out_seq,
                                                mode='seq_only')

    _, probas_sc = get_probabilities(filename=filename, seq_len=sequence_len,
                                     model=model_sc, outfile=probas_out_sc,
                                     mode='sc')

    # Get the auROC and the auPRC for both M-SEQ and M-SC models:
    get_metrics(true_labels, probas_seq, records_files)
    get_metrics(true_labels, probas_sc, records_files)

    # Plot the P-R curves
    combine_pr_curves(records_file_path, probas_seq, probas_sc, true_labels)

    # Plot the posterior distributions of the recall:
    plot_distributions(records_file_path, probas_seq, probas_sc, true_labels,
                       fpr_thresh=0.01)
