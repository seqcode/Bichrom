import numpy as np
import sys
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Inputs:
# 1. Labels file
# 2. Probabilities file - sequence
# 3. Probabilities file - sequence & chromatin

# Output: An auPRC curve

sns.set_style('whitegrid')


def plot_pr_curve():
    """
    Plot the precision-recall curves.
    Parameters:
        sequence_net_probas

    :return:
    """

def plot_prc(labels, probas, plot_color):
    precision, recall, _ = precision_recall_curve(y_true=labels, probas_pred=probas)
    plt.plot(recall, precision, c=plot_color, lw=2.5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)


def plot_figure(sequence_file, bimodal_file, labels, TF):
    sequence_net_probas = np.loadtxt(sequence_file)
    bimodal_net_probas = np.loadtxt(bimodal_file)
    plot_prc(labels, sequence_net_probas, plot_color='#F1C40F')
    plot_prc(labels, bimodal_net_probas, plot_color='#2471A3')
    plt.savefig(TF + '.pdf')


if __name__ == "__main__":
    sequence_file = sys.argv[1]
    bimodal_file = sys.argv[2]
    labels_file = sys.argv[3]
    # Load the labels
    true_labels = np.loadtxt(labels_file)
    TF = sys.argv[4]
    plot_figure(sequence_file=sequence_file, bimodal_file=bimodal_file, labels=true_labels, TF=TF)
