from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import seaborn as sns
import sys


def get_mode(confusion_matrix):
    """
    Get the mode of the recall using the confusion matrix

    Parameters:
        reduced confusion_matrix (ndarray): the confusion matrix at
        a probability threshold defined using a FPR=0.01
        # Ex: reduced confusion matrix
        # no_of_windows   y_hat   y
        # 151             0   1
        # 806             1   1
    Returns: Mode of the Model Recall.
    """
    TP = 0
    FN = 0
    for num_data_points, y_hat, y in confusion_matrix:
        # Defining the TPs, FPs, FNs and TNs:
        if y_hat == 0 and y == 1:
            FN = num_data_points
        elif y_hat == 1 and y ==1:
            TP = num_data_points
    mode = TP / (TP + FN)
    return mode, TP, FN


def plot_beta_uniform_prior(confusion_sequence, confusion_seq_chrom, row, nrows, factor, color):
    """
    # Let the prior be a beta(1,1) or a uniform prior
    # Let c, i be the number of correct & incorrect examples.
    # Then assuming that c and i are drawn from a bag, and the # of correct examples c follows a binomial distribution
    # Then, the posterior is a beta(1 + c, 1 + i)
    """

    row_idx = row + 1
    # Because my indices start from 0, and numpy row numbers start from 1. ~~\/~~
    plt.subplot(nrows, 1, row_idx)
    x = np.linspace(0, 1, 100)
    # Sequence network-only
    mode, tp, fn = get_mode(confusion_sequence)
    print mode
    plt.plot(np.repeat(mode, 100), np.linspace(0, 100, 100), lw=2,
             color=color, ls='--')
    plt.plot(x, beta.pdf(x, 1 + tp, 1 + fn), color='grey', lw=1)

    # Sequence + chromatin network
    mode, tp, fn = get_mode(confusion_seq_chrom)
    print mode
    plt.plot(x, beta.pdf(x, 1 + tp, 1 + fn), color='grey', lw=1)
    plt.plot(np.repeat(mode, 100), np.linspace(0, 100, 100), lw=2,
             color=color, ls='-')

    # Defining the model parameters
    plt.ylim(0, 75)
    plt.yticks([], [])
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1], [], fontsize=14)

    # plt.ylabel(factor, rotation=0)


def plot_figures(data_path):

    sns.set_style('whitegrid')
    # Change the factor list here based on what experiments you want
    # factor_list = ['DUXBL', 'CDX2', 'SOX2', 'RHOX11', 'SOX15',
    #                'FOXA1', 'BHLHB8', 'DLX6', 'SIX6', 'HLF']

    factor_list = ['Hoxc9', 'Hoxc8', 'Hoxc13', 'Hoxc10', 'Hoxa9', 'Hoxd9', 'Hoxc6']
    factor_list = ['Onecut2', 'Brn2', 'Ebf2']
    colors = ['#687466', '#cd8d7b', '#fbc490']
    fig, ax = plt.subplots()
    num_rows = len(factor_list)
    for idx, factor in enumerate(factor_list):
        print factor
        conf_mat_seq = np.loadtxt(data_path + factor + '.seq.chr10.confusion')
        conf_seq_chrom = np.loadtxt(data_path + factor + '.chrom.chr10.confusion')
        plot_beta_uniform_prior(conf_mat_seq, conf_seq_chrom, idx,
                                num_rows, factor, color=colors[idx])

    plt.yticks(fontsize=12)
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1], [0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=14)
    fig.set_size_inches(5.5, 2)
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(sys.argv[1] + 'recall.pdf')


def main():
    sns.set()
    data_path = sys.argv[1]
    plot_figures(data_path)

    # Run notes:
    # Let iTF_revision have all the input files:
    # For example:
    # CDX2.seq.confusion
    # CDX2.chrom.confusion
    # RUN
    # python metrics/posterior_accuracies.py ~/Desktop/iTF_revision/


if __name__ == "__main__":
    main()