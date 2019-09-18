from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import seaborn as sns
import sys


def get_mode(confusion_matrix):
    # Confusion matrix:
    # Num   y_hat   y
    # 255010    0   0
    # 151       0   1
    # 2576      1   0
    # 806       1   1
    TP = 0
    FN = 0
    # Set the TPs and FNs based on the confusion matrix:
    for num_data_points, y_hat, y in confusion_matrix:
        # Defining the TPs, FPs, FNs and TNs:
        if y_hat == 0 and y == 1:
            FN = num_data_points
        else:
            TP = num_data_points
    mode = TP / (TP + FN)
    return mode, TP, FN


def plot_beta_uniform_prior(confusion_sequence, confusion_seq_chrom, row, factor, nrows):
    # Let the prior be a beta(1,1) or a uniform prior
    # Let c, i be the number of correct & incorrect examples.
    # Then assuming that c and i are drawn from a bag, and the # of correct examples c follows a binomial distribution
    # Then, the posterior is a beta(1 + c, 1 + i)
    row_idx = row + 1
    # Because my indices start from 0, and numpy row numbers start from 1. ~~\/~~
    plt.subplot(12, 1, row_idx)
    x = np.linspace(0, 1, 100)
    # Sequence network-only
    mode, tp, fn = get_mode(confusion_sequence)
    plt.plot(np.repeat(mode, 50), np.linspace(0, 50, 50), lw=3, color='#F1C40F')
    plt.plot(x, beta.pdf(x, 1 + tp, 1 + fn), color='grey', lw=1, ls=':')

    # Sequence + chromatin network
    mode, tp, fn = get_mode(confusion_seq_chrom)
    plt.plot(x, beta.pdf(x, 1 + tp, 1 + fn), color='grey', lw=1, ls=':')
    plt.plot(np.repeat(mode, 50), np.linspace(0, 50, 50), lw=3, color='grey')

    # Defining the model parameters
    plt.ylim(0, 50)
    plt.yticks([], [])
    # plt.ylabel(factor, rotation=0)


def plot_figures(data_path):

    sns.set_style('whitegrid')
    # Change the factor list here based on what experiments you want
    factor_list = ['DUXBL', 'CDX2', 'SOX2', 'RHOX11', 'SOX15',
                   'FOXA1', 'BHLHB8', 'DLX6', 'SIX6', 'HLF']
    fig, ax = plt.subplots()
    num_rows = len(factor_list)
    for idx, factor in enumerate(factor_list):
        print factor
        conf_mat_seq = np.loadtxt(data_path + factor + '.seq.confusion')
        conf_seq_chrom = np.loadtxt(data_path + factor + '.chrom.confusion')
        plot_beta_uniform_prior(conf_mat_seq, conf_seq_chrom, idx, factor, num_rows)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 50)
    fig.set_size_inches(8, 8)
    plt.savefig(sys.argv[1] + 'recall.png')


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