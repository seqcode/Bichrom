from __future__ import division
import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
import seaborn as sns
import math
from pylab import *


def extraction_function(model):
    """
    Defines a Keras function to return network embeddings
    Parameters:
         model: the trained chromNN model in hdf5
    Returns:
        Keras function that returns network embeddings
    """
    seq_input = model.layers[0].input    # need to use layer names here
    # chrom_input = model.layers[3].input  # need to use layer names here. # why was this 5 some other time?
    chrom_input = model.layers[4].input
    network_embedding = model.layers[-1].input
    return K.function([seq_input, chrom_input, K.learning_phase()], [network_embedding])


def get_embeddings(model, input_data):
    """
    Get the network embeddings for the input dataset
    Parameters:
        model: A trained model in ".hdf5" format
        input_data: This should be a (seq, chrom) tuple
    Returns:
        Supervised embedding in the sequence-chromatin space.
        This embedding forms a 2 x n matrix, where n is the lenght of the
        input dataset, or the number of input data points
    """
    # get the sequence and chromatin inputs:
    seq_input, chrom_input = input_data
    # use the defined Keras function to get activations:
    extract_activations = extraction_function(model)
    # get and reshape the embedding!
    activations = np.array(extract_activations([seq_input, chrom_input, 0]))
    # reshape with 2 nodes in the pre-logistic (or output) layer
    activations_rs = np.reshape(activations, (activations.shape[1], 2))
    activations_rs = activations_rs.astype(np.float64)
    print activations_rs.shape
    # Now, we want to get the weights assigned to these activations.
    # CHECK: NEED TO MAKE SURE I'M USING THE CORRECT SEQ & CHROM EDGES
    w, b = model.layers[-1].get_weights()
    w = np.reshape(w, (2,))
    embedding = activations_rs * w
    return embedding


def get_embeddings_low_mem(model, input_data):
    """
    Identical in function to get_embeddings, however can work on lower memory
    by loading large datasets in batches. Preferable to get_embeddings() when
    working with the whole genome.
    """
    # get the sequence and chromatin inputs:
    seq_input, chrom_input = input_data
    # use the defined Keras function to get activations:
    extract_activations = extraction_function(model)
    batch_list = []  # contains arrays with embeddings for each batch
    # Iterating in batches till I hit the end of the list:
    for i in range(0, len(seq_input), 500):
        start = i
        stop = min(i + 500, len(seq_input))
        activations = np.array(extract_activations([seq_input[start:stop], chrom_input[start:stop], 0]))
        # reshape with 2 nodes in the pre-logistic (or output) layer
        activations_rs = np.reshape(activations, (activations.shape[1], 2))
        activations_rs = activations_rs.astype(np.float64)
        batch_list.append(activations_rs)
    activations_rs = np.vstack(batch_list)
    w, b = model.layers[-1].get_weights()
    w = np.reshape(w, (2,))
    embedding = activations_rs * w
    return embedding


def plot_1d_seq(out_path, embedding, neg_embedding):
    """
    Plot the 1-D density of sequence scores for bound vs. unbound loci.
    Parameters:
        out_path: Directory to store the output figures
        embedding: Embeddings for the positive or bound set
        neg_embedding: Embeddings for a randomly sampled unbound set
    Returns:
        None
    """
    sns.set_style('ticks')
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=.20, bottom=.20, right=.95, top=.95)
    sns.distplot(embedding[:, 0], bins=50, hist_kws={'alpha': 0.5}, norm_hist=True, color='#D68910', kde=False)
    sns.distplot(neg_embedding[:, 0], bins=50, hist_kws={'alpha': 0.5}, norm_hist=True, color='grey', kde=False)
    # Set axes
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlabel('Sequence sub-network activations', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    fig.set_size_inches(6, 4)
    sns.despine()
    plt.savefig(out_path + 'Sequence_scores.histogram.pdf')


def plot_1d_chrom(out_path, embedding, neg_embedding):
    """
    Plot the 1-D density of chromatin scores for bound vs. unbound loci.
    Parameters:
        out_path: Directory to store the output figures
        embedding: Embeddings for the positive or bound set
        neg_embedding: Embeddings for a randomly sampled unbound set
    Returns:
        None
    """
    # Plotting the sequence differences!
    sns.set_style('ticks')
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=.20, bottom=.20, right=.95, top=.95)
    sns.distplot(embedding[:, 1], bins=50, hist_kws={'alpha': 0.5}, norm_hist=True, color='#D68910', kde=False)
    sns.distplot(neg_embedding[:, 1], bins=50, hist_kws={'alpha': 0.5}, norm_hist=True, color='grey', kde=False)
    # Set axes
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlabel('Chromatin sub-network activations', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    fig.set_size_inches(6, 4)
    sns.despine()
    plt.savefig(out_path + 'Chromatin_scores.histogram.pdf')


def plot_embeddings(out_path, embedding, neg_embedding):
    """
    Plot the joint sequence and chromatin embeddings for bound vs. unbound loci.
    Parameters:
        out_path: Directory to store the output figures
        embedding: Embeddings for the positive or bound set
        neg_embedding: Embeddings for a randomly sampled unbound set
    Returns:
        None
    """
    sns.set_style('ticks')
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=.15, bottom=.15, right=.95, top=.95)
    plt.scatter(x=embedding[:, 0], y=embedding[:, 1], s=8, c='#D68910')
    plt.scatter(x=neg_embedding[:, 0], y=neg_embedding[:, 1], s=8, c='grey')
    # Set figure styles and size
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel('Sequence sub-network activations', fontsize=14)
    plt.ylabel('Chromatin sub-network activations', fontsize=14)
    fig.set_size_inches(6, 6)
    plt.savefig(out_path + "Joint_embedding.pdf")


def plot_split_embeddings(out_path, embedding):
    """
    Plot the joint sequence and chromatin embeddings for bound loci. These
    embeddings are split by the median sequence scores.
    Parameters:
        out_path: Directory to store the output figures
        embedding: Embeddings for the positive or bound set
        neg_embedding: Embeddings for a randomly sampled unbound set
    Returns:
        None
    """
    seq_score = embedding[:, 0]
    chromatin_score = embedding[:, 1]

    # Determine the color threshold automatically
    # print np.sum(seq_score > 4)/len(seq_score)
    q1 = np.quantile(seq_score, 0.25)
    q4 = np.quantile(seq_score, 0.75)
    print q1, q4
    # Figure 1
    sns.set_style('ticks')
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=.15, bottom=.15, right=.95, top=.95)
    # Top quartile
    plt.scatter(x=seq_score[seq_score >= q4], y=chromatin_score[seq_score >= q4], s=4, c='#1F618D', alpha=0.5)
    # Middle data
    plt.scatter(x=seq_score[(seq_score >= q1) & (seq_score < q4)],
                y=chromatin_score[(seq_score >= q1) & (seq_score < q4)], s=4, c='#AEB6BF', alpha=0.5)
    # Bottom quartile
    plt.scatter(x=seq_score[seq_score < q1], y=chromatin_score[seq_score < q1], s=4, c='#85C1E9', alpha=0.5)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel('Sequence sub-network activations', fontsize=14)
    plt.ylabel('Chromatin sub-network activations', fontsize=14)
    fig.set_size_inches(6, 6)
    plt.savefig(out_path + "Embeddings_split.pdf")

    # Figure 2: KDE PLOTS
    sns.set_style('ticks')
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=.15, bottom=.15, right=.95, top=.95)
    sns.kdeplot(chromatin_score[seq_score > 4], color='#D68910')
    sns.kdeplot(chromatin_score[seq_score < 4], color='#2471A3')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlabel('Chromatin sub-network activations', fontsize=12)
    plt.ylabel('Frequency', fontsize=14)
    fig.set_size_inches(6, 4)
    plt.savefig(out_path + "Embeddings_split_density.pdf")