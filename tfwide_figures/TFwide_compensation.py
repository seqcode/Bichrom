from __future__ import division
import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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


def get_activations_low_mem(model, input_data):
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
    abs_w = w/abs(w)
    print abs_w[0]
    return abs_w * activations_rs  # Not returning the embeddings here.


def split_embeddings_by_domains(embeddings, domains, protein):
    # domains is a column vector with each row either 1/0 based on ATAC-seq.
    seq_embeddings = embeddings[:, 0]
    embeddings_at_acc = seq_embeddings[domains == 1]
    embeddings_at_inacc = seq_embeddings[domains == 0]

    # This treats the A/IA as an index
    dat_a = pd.DataFrame(embeddings_at_acc, columns=['score'])
    dat_a['chromatin'] = pd.DataFrame(np.asarray('A').repeat(len(embeddings_at_acc)))
    dat_ia = pd.DataFrame(embeddings_at_inacc, columns=['score'])
    dat_ia['chromatin'] = pd.DataFrame(np.asarray('IA').repeat(len(embeddings_at_inacc)))
    dat = pd.concat([dat_a, dat_ia])
    dat['protein'] = protein
    return dat
    # print dat
    # Plotting
    # sns.boxplot(x=dat.index.values, y=dat[0])
    # plt.savefig(outpath + 'split_sequenceScores.pdf')



