from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from sklearn.metrics import roc_curve
import seaborn as sns

sns.set_style('whitegrid')


def identify_proba_thresh(proba_vector, labels, fpr_thresh):
    """
    Identify the probability threshold at which the FPR = 0.01
    :param proba_vector: A vector of probabilities across a test set (chromosomes/whole genome)
    :param labels: A vector of labels for the test set
    :return: Recall at a fixed FPR, no_of_bound_sites
    """
    fpr, tpr, thresholds = roc_curve(labels, proba_vector)
    fpr = list(fpr)
    arg = [idx for idx, curr_fpr in enumerate(fpr) if curr_fpr > fpr_thresh][0]
    # probability threshold:
    fpr_proba_thresh = thresholds[arg]  # TEST THIS FURTHER
    tpr_at_fixed_fpr = tpr[arg]
    print tpr_at_fixed_fpr

    # Calculate the total number of bound sites:
    no_of_bound_sites = int(np.sum(labels))
    true_positives = int(tpr_at_fixed_fpr * no_of_bound_sites)
    false_negatives = no_of_bound_sites - true_positives
    return tpr_at_fixed_fpr, true_positives, false_negatives


def plot_distributions(outpath, seq_probas, sc_probas, labels, fpr_thresh):
    fig, ax = plt.subplots()
    # Plotting results for M-SEQ
    tpr_at_fixed_fpr, true_pos, false_negs = identify_proba_thresh(seq_probas, labels, fpr_thresh=fpr_thresh)
    x = np.linspace(0, 1, 100)
    plt.plot(np.repeat(tpr_at_fixed_fpr, 100), np.linspace(0, 100, 100), lw=3, color='#F1C40F')
    plt.plot(x, beta.pdf(x, 1 + true_pos, 1 + false_negs), color='grey', lw=1.5)
    max_beta_seq = max(beta.pdf(x, 1 + true_pos, 1 + false_negs))

    # Plotting results for M-SC
    tpr_at_fixed_fpr, true_pos, false_negs = identify_proba_thresh(sc_probas, labels, fpr_thresh=fpr_thresh)
    plt.plot(np.repeat(tpr_at_fixed_fpr, 100), np.linspace(0, 100, 100), lw=3, color='#2471A3')
    plt.plot(x, beta.pdf(x, 1 + true_pos, 1 + false_negs), color='grey', lw=1.5)
    max_beta_sc = max(beta.pdf(x, 1 + true_pos, 1 + false_negs))

    plt.yticks([], [])
    # plt.yticks(fontsize=12)
    plt.xlim(0, 1)
    # Set the y-axis limit automatically:
    # plt.ylim(0, max(max_beta_sc, max_beta_seq) + 5)

    plt.ylim(0, 75)
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1], [0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=14)
    fig.set_size_inches(5, 1)
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(outpath + '.posterior_recall.pdf')