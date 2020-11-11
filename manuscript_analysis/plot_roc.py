import numpy as np
import sys, os
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import seaborn as sns


def plot_fig_s1a():
    # supplementary figure 1A
    # change this based on file location
    labels_file = '/Users/asheesh/Desktop/iTF_revision/test.Ascl1/' \
                  'Ascl1_run/Ascl1.chr10.labels'
    seq_probas_file = '/Users/asheesh/Desktop/iTF_revision/test.Ascl1/' \
                      'Ascl1_run/Ascl1.chr10.seq.probs'
    chrom_probas_file = '/Users/asheesh/Desktop/iTF_revision/test.Ascl1/' \
                        'Ascl1_run/Ascl1.chr10.chrom.probs'

    outfile = '/Users/asheesh/Desktop/iTF_revision/iTF_figures/SuppFig1/'

    # Loading the data
    labels = np.loadtxt(labels_file)
    seq_probas = np.loadtxt(seq_probas_file)
    chrom_probas = np.loadtxt(chrom_probas_file)

    # Getting the FPR and TPRs
    fpr_seq, tpr_seq, _ = roc_curve(y_true=labels, y_score=seq_probas)
    fpr_chrom, tpr_chrom, _ = roc_curve(y_true=labels, y_score=chrom_probas)

    # Plotting the curves
    sns.set_style('white')
    fig, ax = plt.subplots()
    # ...plot ...
    plt.plot(fpr_seq, tpr_seq, color='#dcc57c', lw=2.5)
    plt.plot(fpr_chrom, tpr_chrom, color='#618681', lw=2.5)
    fig.set_size_inches(4, 4)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(outfile + '1A.pdf')


def plot_fig_s1b():
    # supplementary figure 1B
    run_metrics = '/Users/asheesh/Desktop/iTF_revision/test.Ascl1/' \
                  'Ascl1_run/ascl1_run.txt'
    outfile = '/Users/asheesh/Desktop/iTF_revision/iTF_figures/SuppFig1/'
    dat = np.loadtxt(run_metrics, dtype=str)
    print dat
    sns.set_style('ticks')
    fig, ax = plt.subplots()
    sns.stripplot(x=dat[:, 0], y=dat[:, 1].astype(float), color='#27496d',
                  size=2)
    sns.boxplot(x=dat[:, 0], y=dat[:, 1].astype(float), color='#dae1e7',
                linewidth=1, showfliers=False)
    plt.ylim(0, 1)
    fig.set_size_inches(4, 4)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    sns.despine()
    plt.savefig(outfile + '1B.pdf')


if __name__ == "__main__":
    plot_fig_s1a()
    plot_fig_s1b()
