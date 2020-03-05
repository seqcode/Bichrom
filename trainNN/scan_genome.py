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

# keras imports
from keras.models import load_model


def merge_generators(filename, batchsize, seqlen, mode):
    X = iu.train_generator(filename + ".seq", batchsize, seqlen, "seq", mode) 
    A = iu.train_generator(filename + ".chromtracks", batchsize, seqlen, "accessibility", mode)
    y = iu.train_generator(filename + ".labels", batchsize, seqlen, "labels", mode)
    while True:
        yield [X.next(), A.next()], y.next()


def test_on_batch(batch_generator, model, outfile, mode):
    """
    Get probabilities for each test data point.
    The reason that this is implemented in a batch is because \
    the whole genome cannot be loaded without batching.

    Single held-out chromosomes can be tested directly.

    :param batch_generator: a generator that yields sequence, chromatin and label vectors.
    :param model: A trained Keras model
    :param outfile: The outfile used for storing probabilities.
    :return: None (Saves an output file with the probabilities for the test set )
    """
    print "in test_on_batch"
    counter = 0
    while True:
        try:
            print "generating bacth"
            [X_test, acc_test], y = batch_generator.next()
            print "done generating..."
            if mode == 'seq_only':
                print "inmode"
                batch_probas = model.predict_on_batch([X_test])
                print "done round"
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

    Takes the test labels and test probabilities, and calculates/
    plots the following:

        a. P-R Curves
        b. auPRC
        c. auROC
        d. Posterior Distributions of the Recall at FPR=0.01

    :param test_labels: n * 1 vector with the true labels ( 0 or 1 )
    :param test_probas: n * 1 vector with the network probabilities
    :return: None
    """
    # Calculate auROC
    roc_auc = sklearn.metrics.roc_auc_score(test_labels, test_probas)
    # Calculate auPRC
    prc_auc = sklearn.metrics.average_precision_score(test_labels, test_probas)

    # Write auROC and auPRC to records file
    records_file.write("AUC ROC:{0}".format(roc_auc))
    records_file.write("AUC PRC:{0}".format(prc_auc))


def get_probabilities(filename, seq_len, model_file, outfile, mode):
    """
    Get network-assigned probabilities
    :param filename: Input file to be loaded
    :param seq_len: eg. 500
    :return: probas
    """
    # Inputing a range of default values here, can be changed later.
    data_generator = merge_generators(filename=filename, batchsize=1000, seqlen=seq_len,
                                      mode='nr')  # Testing mode = non-repeating
    # Load the keras model
    model = load_model(model_file)
    test_on_batch(data_generator, model, outfile, mode)
    probas = np.loadtxt(outfile)  # Need to change this, but not now! Change structure first.
    true_labels = np.loadtxt(filename + '.labels')

    return true_labels, probas


def plot_pr_curve(test_labels, test_probas, color):
    # Get the PR values:
    precision, recall, _ = precision_recall_curve(y_true=test_labels, probas_pred=test_probas)
    plt.plot(recall, precision, c=color, lw=2.5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)


def combine_pr_curves(records_file, m_seq_probas, m_sc_probas, labels):
    plot_pr_curve(labels, m_seq_probas, color='#F1C40F')
    plot_pr_curve(labels, m_sc_probas, color='#2471A3')
    plt.savefig(records_file + '.pr_curves.pdf')


def main():

    sequence_len = 500

    # Use an argparse module here.
    filename = sys.argv[1]
    # M-SEQ
    probas_out_seq = sys.argv[2]
    model_seq = sys.argv[3]
    # M-SC
    probas_out_sc = sys.argv[4]
    model_sc = sys.argv[5]
    # Output File
    records_files_path = sys.argv[6]
    records_files = open(records_files_path, "w")

    # Get the probabilities for both M-SEQ and M-SC models:
    # Note: Labels are the same for M-SC and M-SEQ
    print "1"
    true_labels, probas_seq = get_probabilities(filename=filename, seq_len=sequence_len,
                                                model_file=model_seq, outfile=probas_out_seq, mode='seq_only')

    print "2"
    _, probas_sc = get_probabilities(filename=filename, seq_len=sequence_len,
                                     model_file=model_sc, outfile=probas_out_sc, mode='sc')

    print "3"
    # Get the auROC and the auPRC for both M-SEQ and M-SC models:
    get_metrics(true_labels, probas_seq, records_files)
    get_metrics(true_labels, probas_sc, records_files)

    print "4"
    # Plot the P-R curves
    combine_pr_curves(records_files_path, probas_seq, probas_sc, true_labels)

    print "5"
    # Plot the posterior distributions of the recall:
    plot_distributions(records_files_path, probas_seq, probas_sc, true_labels, fpr_thresh=0.01)


if __name__ == "__main__":
    main()