from __future__ import division
import keras.backend as K
import seaborn as sns
from pylab import *


def extraction_function(model):
    """
    Defines a Keras function to return network embeddings
    Parameters:
         model: the trained chromNN model in hdf5
    Returns:
        Keras function that returns network embeddings
    """
    seq_input = model.get_layer('seq').input
    chrom_input = model.get_layer('chrom_input').input
    network_embedding = model.layers[-1].input
    return K.function([seq_input, chrom_input, K.learning_phase()], [network_embedding])


def get_embeddings(model, seq_input, chrom_input):
    """
    Get the network embeddings for the input dataset
    Parameters:
        model: A trained model in ".hdf5" format
        seq_input: This should be a onehot seq tensor
        chrom_input: This should be a chromatin vector
    Returns:
        Supervised embedding in the sequence-chromatin space.
        This embedding forms a 2 x n matrix, where n is the lenght of the
        input dataset, or the number of input data points
    """
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


def get_embeddings_low_mem(model, seq_input, chrom_input):
    """
    Identical in function to get_embeddings, however can work on lower memory
    by loading large datasets in batches. Preferable to get_embeddings() when
    working with the whole genome.
    """
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
    plt.scatter(x=embedding[:, 0], y=embedding[:, 1], c='#D68910',
                s=3, alpha=0.2)
    plt.scatter(x=neg_embedding[:, 0], y=neg_embedding[:, 1], s=3, alpha=0.2,
                c='grey')
    # Set figure styles and size
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel('Sequence sub-network activations', fontsize=14)
    plt.ylabel('Chromatin sub-network activations', fontsize=14)
    fig.set_size_inches(6, 6)
    plt.savefig(out_path + "Joint_embedding.png", dpi=960)


def plot_embeddings_bound_only(out_path, embedding, neg_embedding):
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

    # Set figure styles and size
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel('Sequence sub-network activations', fontsize=14)
    plt.ylabel('Chromatin sub-network activations', fontsize=14)
    fig.set_size_inches(6, 6)
    plt.savefig(out_path + "Joint_embedding_bound_only.pdf")

    # plot histogram: split embeddings into seq-low and seq-high & plot.
    median_seqscore = np.median(embedding[:,0])
    chrom_at_seq_low = embedding[:, 1][embedding[:, 0] <= median_seqscore]
    chrom_at_seq_high = embedding[:, 1][embedding[:, 0] > median_seqscore]