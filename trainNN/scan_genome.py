from __future__ import division
import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.metrics import precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# user defined module
from iterutils import train_generator
from helper import plot_distributions


def merge_generators(path, batchsize, seqlen, mode):
    dat_seq = train_generator(path['seq'], batchsize, seqlen, 'seq', mode)
    dat_chromatin = []
    for chromatin_track in path['chromatin_tracks']:
        dat_chromatin.append(
            train_generator(chromatin_track, batchsize, seqlen, 'chrom', mode))
    y = train_generator(path['labels'], batchsize, seqlen, 'labels', mode)
    while True:
        combined_chrom_data = []
        for chromatin_track_generators in dat_chromatin:
            x = next(chromatin_track_generators)
            combined_chrom_data.append(pd.DataFrame(x))
        chromatin_features = pd.concat(combined_chrom_data, axis=1).values
        sequence_features = next(dat_seq)
        labels = next(y)
        yield [sequence_features, chromatin_features], labels


def test_on_batch(batch_generator, model, outfile, mode):
    """
    Get probabilities for each test data point.
    The reason that this is implemented in a batch is because
    the whole genome cannot be loaded without batching.
    Parameters:
        batch_generator (generator): a generator that yields sequence, chromatin
        and label vectors.
        model (keras Model): A trained Keras model
        outfile (str): The outfile used for storing probabilities.
    Returns: None (Saves an output file with the probabilities for the test set )
    """
    counter = 0
    while True:
        try:
            [X_test, acc_test], y = next(batch_generator)
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


def get_metrics(test_labels, test_probas, records_file, model_name):
    """
    Takes the test labels and test probabilities, and calculates and/or
    plots the following:
    a. P-R Curves
    b. auPRC
    c. auROC
    d. Posterior Distributions of the Recall at FPR=0.01
    Parameters:
        test_labels (ndarray): n * 1 vector with the true labels ( 0 or 1 )
        test_probas (ndarray): n * 1 vector with the network probabilities
        records_file (str): Path to output file
        model_name (str): Model being tested
    Returns: None
    """
    # Calculate auROC
    roc_auc = sklearn.metrics.roc_auc_score(test_labels, test_probas)
    # Calculate auPRC
    prc_auc = sklearn.metrics.average_precision_score(test_labels, test_probas)
    records_file.write('')
    # Write auROC and auPRC to records file.
    records_file.write("Model:{0}\n".format(model_name))
    records_file.write("AUC ROC:{0}\n".format(roc_auc))
    records_file.write("AUC PRC:{0}\n".format(prc_auc))


def get_probabilities(path, seq_len, model, outfile, mode):
    """
    Get network-assigned probabilities
    Parameters:
        filename (str): Input file to be loaded
        seq_len (int): Length of input DNA sequence
    Returns:
         probas (ndarray): An array of probabilities for the test set
         true labels (ndarray): True test-set labels
    """
    # Inputing a range of default values here, can be changed later.
    data_generator = merge_generators(path=path, batchsize=1000,
                                      seqlen=seq_len, mode='nr')
    # Load the keras model
    # model = load_model(model_file)
    test_on_batch(data_generator, model, outfile, mode)
    probas = np.loadtxt(outfile)
    true_labels = np.loadtxt(path['labels'])
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


def evaluate_models(sequence_len, path, probas_out_seq, probas_out_sc,
                    model_seq, model_sc, records_file_path):

    # Define the file that contains testing metrics
    records_files = open(records_file_path + '.txt', "w")

    # Get the probabilities for both M-SEQ and M-SC models:
    # Note: Labels are the same for M-SC and M-SEQ
    true_labels, probas_seq = get_probabilities(path=path,
                                                seq_len=sequence_len,
                                                model=model_seq,
                                                outfile=probas_out_seq,
                                                mode='seq_only')

    _, probas_sc = get_probabilities(path=path, seq_len=sequence_len,
                                     model=model_sc, outfile=probas_out_sc,
                                     mode='sc')

    # Get the auROC and the auPRC for both M-SEQ and M-SC models:
    get_metrics(true_labels, probas_seq, records_files, 'MSEQ')
    get_metrics(true_labels, probas_sc, records_files, 'MSC')

    # Plot the P-R curves
    combine_pr_curves(records_file_path, probas_seq, probas_sc, true_labels)

    # Plot the posterior distributions of the recall:
    plot_distributions(records_file_path, probas_seq, probas_sc, true_labels,
                       fpr_thresh=0.01)
