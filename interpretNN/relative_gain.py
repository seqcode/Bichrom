import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve


def metrics_boxplots(datapath):
    """
    Saves the chrom-only, sequence-only, sequence + control, sequence + chromatin boxplots to datapath.results.txt
    Parameters:
        datapath (str): Path the to ".txt" comma-separated run summary files (without the .txt suffix)
    Returns:
        None
    """
    # Loading the data:
    dat = np.loadtxt(datapath + '.txt', delimiter=',', dtype=str)
    # Extract the data that to be plotted:
    # Plot 1: Using only the following four runs here: (not the ATAC)
    chrom_only = (dat[:, 0] == 'ES_chrom')
    sequence_only = (dat[:, 0] == 'ES_seq')
    sequence_and_chromatin = (dat[:, 0] == 'ES')
    sequence_and_control = (dat[:, 0] == 'input')

    conditions = np.transpose(np.vstack((chrom_only, sequence_only, sequence_and_control, sequence_and_chromatin)))
    dat = dat[np.any(conditions, axis=1)]

    sns.set_style('ticks', {'axes.edgecolor': 'black'})
    fig, ax = plt.subplots()
    # Define the color palette:
    my_palette = {'ES_chrom': 'indianred', 'ES_seq': '#F1C40F', 'ES': '#1F618D', 'input': 'grey'}
    # Plotting
    sns.boxplot(dat[:, 0], dat[:, 1].astype(float), palette=my_palette)
    # Defining the plot attributes
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 1)
    plt.ylabel("", fontsize=12)
    # Defining the plot size & thickness etc.
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1)
    # Setting figure size and saving figure.
    fig.subplots_adjust(left=.20, bottom=.10, right=.95, top=.95)
    # Plot size:
    width = 4
    height = 4.5
    fig.set_size_inches(width, height)
    sns.despine()
    plt.savefig(datapath + 'boxplots.results.pdf')


def relative_prcs(datapath):
    """
    Plots the AUC (PR) for all the held-out chromosomes for both sequence, and the sequence+chromatin models.
    Parameters:
        datapath (str): Path the to ".probas" & ".labels" files for the held-out chromosomes (without the files suffix)
    Returns:
         None
    """
    # Note: Keeps colors consistent with the boxplots.
    # The files are currently here: ~/Documents/tempACI_files/Ascl1.probas/

    sns.set_style('ticks', {'axes.edgecolor': 'black'})
    fig, ax = plt.subplots()

    chromosomes = ['chr10', 'chr11', 'chr12', 'chr13', 'chr14']
    alpha = 0.2
    for chrom in chromosomes:
        if chrom == 'chr10':
            alpha = 1
        else:
            alpha = 0.2
        # Load the labels:
        labels = np.loadtxt(datapath + '.probas/Ascl1.' + chrom + '.labels')
        # Load the sequence probabilities:
        sequence_probas = np.loadtxt(datapath + '.probas/Ascl1.ES.' + chrom + '.seq.adam.probs')
        # Load the chromatin probabilities:
        sequence_and_chrom_probas = np.loadtxt(datapath + '.probas/Ascl1.ES.' + chrom + '.chrom.adam.probs')

        sequence_precision, sequence_recall, _ = precision_recall_curve(labels, sequence_probas)
        plt.step(sequence_recall, sequence_precision, color='#F1C40F', where='post', alpha=alpha)
        full_precision, full_recall, _ = precision_recall_curve(labels, sequence_and_chrom_probas)
        plt.step(full_recall, full_precision, color='#1F618D', where='post', alpha=alpha)

    # Defining the plot attributes
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 1)
    plt.ylabel("", fontsize=12)
    # Defining the plot size & thickness etc.
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1)
    # Setting figure size and saving figure.
    fig.subplots_adjust(left=.10, bottom=.10, right=.95, top=.95)
    # Plot size:
    width = 4
    height = 4.5
    fig.set_size_inches(width, height)
    sns.despine()
    plt.savefig(datapath + '.prc.results.pdf')

