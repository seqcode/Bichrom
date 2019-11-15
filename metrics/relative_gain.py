from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta
import sys

## PAPER
def metrics_boxplots(datapath):
    sns.set_style('ticks', {'axes.edgecolor': 'black'})
    print sns.axes_style()
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=.20, bottom=.10, right=.99, top=.90)
    # sns.set_style('ticks')
    dat = np.loadtxt(datapath + '.txt', delimiter=", ", dtype=str)
    # Plot 1: Using only the following four runs here: (not the ATAC)
    c0 = (dat[:, 0] == 'ES_chrom')
    c1 = (dat[:, 0] == 'ES_seq')
    c2 = (dat[:, 0] == 'ES')
    c3 = (dat[:, 0] == 'input')
    # Let's use these conditions to subset are data
    conditions = np.transpose(np.vstack((c0, c1, c2, c3)))
    dat = dat[np.any(conditions, axis=1)]
    width = 4
    height = 4.5
    # Box plot for gain in performance:
    sns.boxplot(dat[:, 0], dat[:, 1].astype(float), palette="Set2")
    plt.xticks(range(4), [r'$M_c$', r'$M_s$', r'$M_s$'+r'$_i$', r'$M_s$' + r'$_c$'], fontsize=12)
    plt.ylim(0, 1)
    plt.yticks(fontsize=12)
    plt.ylabel("AUC (Precision-Recall Curve)", fontsize=12)
    plt.title('Ascl1', fontsize=14)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
    # Setting figure size and saving figure.
    fig.set_size_inches(width, height)
    sns.despine()
    plt.savefig(datapath + '.results.pdf')
    plt.show()


def metrics_boxplots2(datapath, celltype, TF):
    sns.set_style('ticks', {'axes.edgecolor': 'black'})
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=.20, bottom=.10, right=.99, top=.90)
    # sns.set_style('ticks')
    dat = np.loadtxt(datapath + '.txt', delimiter=", ", dtype=str)
    print dat
    # Plot 1: Using only the following four runs here: (not the ATAC)
    s = celltype + "_" + TF
    c = celltype + "_chrom_" + TF
    c0 = (dat[:, 0] == celltype + "_" + TF)
    c1 = (dat[:, 0] == celltype + "_chrom_" + TF)
    # Let's use these conditions to subset are data
    conditions = np.transpose(np.vstack((c0, c1)))
    dat = dat[np.any(conditions, axis=1)]
    width = 4
    height = 4.5
    # Box plot for gain in performance:
    my_pal = {s: "#F4D03F", c: "#2471A3"}

    sns.boxplot(dat[:, 0], dat[:, 1].astype(float), palette=my_pal, order=[s, c])
    # plt.xticks(range(4), [r'$M_c$', r'$M_s$', r'$M_s$'+r'$_i$', r'$M_s$' + r'$_c$'], fontsize=12)
    plt.xticks(range(2), [r'$M_s$',r'$M_s$' + r'$_c$'], fontsize=16)
    plt.ylim(0, 1)
    plt.yticks(fontsize=14)
    plt.ylabel("AUC (Precision-Recall Curve)", fontsize=14)
    plt.title(TF, fontsize=18)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
    # Setting figure size and saving figure.
    fig.set_size_inches(width, height)
    sns.despine()
    plt.savefig(datapath + celltype + TF + '.results.pdf')


def subp(datapath):
    sns.set_style('whitegrid')
    # iA
    Brn2 = [300, 354, 385, 269]
    Ebf2 = [792, 1225, 1128, 889]
    Onecut2 = [1245, 1052, 1305, 992]
    Ascl1 = [585, 372, 675, 282, 699, 258]
    Ngn2 = [847, 528, 976, 399, 1042, 333]

    # iN
    # iNBrn2 = [288, 229, 293, 224]
    # iNEbf2 = [792, 1225, 1128, 889]
    # iNOnecut2 = [1245, 1052, 1305, 992]
    # Ascl1 = [585, 372, 675, 282]
    # Ngn2 = [847, 528, 976, 399]
    # Ascl1_HM = [585, 372, 699, 258]
    # Ngn2_HM = [847, 528, 1042, 333]

    tflist = [Ascl1, Ngn2, Brn2, Ebf2, Onecut2]
    tfnames = ['Ascl1', 'Ngn2', 'Brn2', 'Ebf2', 'Onecut2', 'Ascl1_HM', 'Ngn2_HM']

    fig, ax = plt.subplots()
    for idx, vals, name in zip(range(5), tflist, tfnames):

        plt.subplot(5,1, idx+1)
        seq_recall = vals[0]/(vals[0] + vals[1])
        plt.plot(np.repeat(seq_recall, 40), range(40), c='#F4D03F', lw=2)
        x = np.arange(0.01, 1, 0.01)
        y = beta.pdf(x, 1 + vals[0], 1 + vals[1])
        plt.plot(x, y,  c='#F4D03F', lw=1, ls='--', color='grey')

        chrom_recall = vals[2]/(vals[2] + vals[3])
        plt.plot(np.repeat(chrom_recall,40), range(40), c='#2471A3', lw=2)
        y = beta.pdf(x, 1 + vals[2], 1 + vals[3])
        plt.plot(x, y,  c='#2471A3', lw=1, ls='--', color='grey')

        if name == 'Ascl1' or name == 'Ngn2':
            print "here"
            HM_recall = vals[4]/(vals[4] + vals[5])
            plt.plot(np.repeat(HM_recall, 40), range(40), c='grey', lw=2)
            y = beta.pdf(x, 1 + vals[4], 1 + vals[5])
            plt.plot(x, y, c='#2471A3', lw=1, ls='--', color='grey')

        plt.ylim(0,40)
        plt.xlim(0,1)
        if idx < 4:
            plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1], [])
        else:
            plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1], [0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.yticks([], [])
        plt.ylabel(name, fontsize=12)
    fig.set_size_inches(8, 4)
    plt.savefig(datapath + 'recall.pdf')


datapath = sys.argv[1]

subp(datapath)

#for celltype in ['iA', 'iN']:
#    for tf in ['Ebf2', 'Onecut2', 'Brn2']:
#        metrics_boxplots2(datapath, celltype, tf)
