from __future__ import division
import h5py
import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.metrics import precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

# user defined module
import iterutils
from helper import plot_distributions


def TFdataset(path, batchsize, dataflag):

    TFdataset_batched = iterutils.train_TFRecord_dataset(path, batchsize, dataflag)

    return TFdataset_batched


def test_on_batch(TFdataset, model, outfile, mode):
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
    # erase the contents in outfile
    file = open(outfile, "w")
    file.close()

    probas = np.array([])
    true_labels = np.array([])
    for x_vals, y_val in TFdataset:
        if mode == 'seq_only':
            X_test = tf.data.Dataset.from_tensors(x_vals["seq"])
        else:
            ds = [tf.data.Dataset.from_tensors(val) for key, val in x_vals.items()]
            X_test = tf.data.Dataset.zip((tuple(ds),))
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE
        X_test = X_test.with_options(options)
        batch_probas = model.predict(X_test)
        # saving to file: 
        with open(outfile, "a") as fh:
            np.savetxt(fh, batch_probas)
        # save predictions and true labels
        probas = np.concatenate([probas, batch_probas.flatten()])
        true_labels = np.concatenate([true_labels, y_val])
    
    return true_labels, probas


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


def get_probabilities(path, model, outfile, mode):
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
    dataset = TFdataset(path=path, batchsize=1000, dataflag="all")
    # Load the keras model
    # model = load_model(model_file)
    true_labels, probas = test_on_batch(dataset, model, outfile, mode)

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


def evaluate_models(path, probas_out_seq, probas_out_sc,
                    model_seq, model_sc, records_file_path):

    # Define the file that contains testing metrics
    records_files = open(records_file_path + '.txt', "w")

    # Get the probabilities for both M-SEQ and M-SC models:
    # Note: Labels are the same for M-SC and M-SEQ
    true_labels, probas_seq = get_probabilities(path=path,
                                                model=model_seq,
                                                outfile=probas_out_seq,
                                                mode='seq_only')

    _, probas_sc = get_probabilities(path=path, 
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
