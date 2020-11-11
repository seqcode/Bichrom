"""
Functions to analyze preexisting mouse ES chromatin data at Ascl1 binding sites.
These functions are called by explain_variance.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joint_embeddings import get_embeddings_low_mem
import pandas as pd


def make_heatmap_per_quartile(datapath, out_path):
    """
    Figure 5B
    Heatmap of 13 rows * 4 columns.
    Each row is a histone modification, plotting median enrichment for each HM
    at the four quartiles based on the chromatin sub-network scores.

    Parameters:
        datapath: path + prefix to Ascl1 data
        out_path: output file prefix
    Returns: None
    """
    # Label order as in design file
    hm_labels = ['ATACSEQ: 0', 'H3K27ac: 1', 'H3K27me3: 2', 'H3K4me1: 3', 'H3K4me2: 4',
                 'H3K4me3: 5', 'H3K9ac: 6', 'H3K9me3: 7', 'H3K36me3: 8', 'H2AZ: 9', 'acH2AZ: 10',
                 'H3: 11', 'H4K20me3: 12']

    # load & reshape the chromatin data
    chromatin_data = np.load(datapath + '.bound.chromtracks.npy')
    chromatin_data = np.reshape(chromatin_data, (-1, 13, 10))
    # sum tags in each window.
    summarized_chrom_dat = np.sum(chromatin_data, axis=2)

    # load and sort the embeddings
    embedding = np.loadtxt(datapath + '.embedding.txt')
    chromatin_net_scores = embedding[:, 1]
    sorted_indices = np.argsort(chromatin_net_scores)
    chrom_data_sorted = summarized_chrom_dat[sorted_indices][::-1]

    # get data quartiles
    data_len = len(chrom_data_sorted)
    quartile_1 = chrom_data_sorted[:data_len/4]
    quartile_2 = chrom_data_sorted[data_len/4: data_len/2]
    quartile_3 = chrom_data_sorted[data_len/2: (3*data_len/4)]
    quartile_4 = chrom_data_sorted[(3*data_len/4):]

    # process data in each quartile: summing across instances
    heatmap_dat = []
    for q in [quartile_1, quartile_2, quartile_3, quartile_4]:
        mean_enrichment = np.sum(q, axis=0)/len(q)
        heatmap_dat.append(mean_enrichment)
    heatmap_dat = np.array(heatmap_dat)

    heatmap_dat = heatmap_dat.transpose()

    # reorder heatmap to match the order of the chromatin distribution plot
    # (Fig. 5A)
    # remove H3 as no domains are found here.
    heatmap_dat = heatmap_dat[[1, 9, 10, 3, 0, 4, 5, 6, 8, 2, 7, 12],:]

    # plotting the heatmap
    fig, ax = plt.subplots()
    sns.heatmap(heatmap_dat, cmap='copper', yticklabels=hm_labels,
                linewidths=0.5, linecolor='grey', cbar_kws={"shrink": 0.5})
    fig.set_size_inches(3, 6)
    fig.subplots_adjust(left=.30, bottom=.10, right=.90, top=.95)
    plt.savefig(out_path + '5B.pdf')


def plot_compensation(datapath, out_path):
    """
    Figure 5C and 5D
    Plotting the average BichromSEQ and BichromCHR scores at binding sites within
    each of the 12 chromHMM states called in mouse ES cells.
    Note: For this function, we require to already have extracted the latent 2-D embedding
    This can be done using joint_embeddings/get_latent_embeddings.py
    Parameters:
        datapath: path + prefix to input data
        out_path: output file prefix
    Note: The input data directory also must contain a genome annotation file
    derived from chrom
    Returns: None
    """
    # Loading the joint embeddings:
    embedding = np.loadtxt(datapath + '.embedding.txt')
    bichrom_seq_score = embedding[:, 0]
    bichrom_chr_score = embedding[:, 1]

    # loading the mES chromHMM annotations
    annotation = np.loadtxt(datapath + '.bound.chromHMM.annotation', dtype=str)
    hmm_states_at_ascl1sites = annotation[:, 0]
    state_labels = ['E1', 'E2', 'E3', 'E5', 'E4', 'E6', 'E7', 'E8', 'E9',
                    'E10', 'E11']

    state_terms = ['CTCF', 'Quiescent', 'Heterochromatin', 'Enhancer',
                   'Repressed Chromatin', 'Bivalent Promoters',
                   'Active Promoter', 'Strong Enhancer',
                   'Transcriptional Transition', 'Transcriptional Elongation',
                   'Weak/Poised Enhancers']

    data = np.vstack((bichrom_seq_score, bichrom_chr_score,
                      hmm_states_at_ascl1sites))
    data = data.transpose()
    state_labels_idx = [int(x[1:]) for x in state_labels]
    # getting the mean chromatin and sequence network scores at states.
    seq_mean_values = []
    chrom_mean_values = []
    sizes = []
    for hmm_state in state_labels:
        seq_subsetted_dat = data[data[:, 2] == hmm_state, 0].astype(float)
        chrom_subsetted_dat = data[data[:, 2] == hmm_state, 1].astype(float)
        # Appending means to list
        seq_mean_values.append(np.median(seq_subsetted_dat))
        chrom_mean_values.append(np.median(chrom_subsetted_dat))
        sizes.append(np.shape(seq_subsetted_dat)[0])

    chrom_mean_values = np.array(chrom_mean_values)[np.argsort(state_labels_idx)]
    seq_mean_values = np.array(seq_mean_values)[np.argsort(state_labels_idx)]
    # scaling sizes
    sizes = [x/10 for x in sizes]
    # Plotting the bubble plots
    fig, axs = plt.subplots(1, 2)
    # chromatin mean values
    axs[0].scatter(chrom_mean_values, state_labels_idx, s=sizes, color='#cd8d7b')
    for y_coordinate in range(1, 12):
       axs[0].axhline(y=y_coordinate, xmin=0, xmax=1,
                  ls='--', color='grey', lw=1)
    # sequence mean values
    axs[1].scatter(seq_mean_values, state_labels_idx, s=sizes, color='#084177')
    for y_coordinate in range(1, 12):
       axs[1].axhline(y=y_coordinate, xmin=0, xmax=1,
                  ls='--', color='grey', lw=1)
    plt.savefig(out_path + '5C.pdf')


def scores_at_domains(model, datapath, out_path):
    """
    Plot the distribution of preexisting chromatin scores at the bound
    TF sites.
    Parameters:
        model: A tf.Keras model.
        datapath: Path to input data directory
        out_path: Output file prefix.
    """
    # load the entire chromatin data
    chromatin = np.loadtxt(datapath + '.chromtracks')
    domains = np.loadtxt(datapath + '.domains')
    # Label order is from the chromatin design file
    hm_labels = ['ATACSEQ', 'H3K27ac', 'H3K27me3', 'H3K4me1', 'H3K4me2',
                 'H3K4me3', 'H3K9ac', 'H3K9me3', 'H3K36me3', 'H2AZ', 'acH2AZ',
                 'H3', 'H4K20me3']
    domain_scores = []
    for idx, label in enumerate(hm_labels):
        enrichment = domains[:, idx]
        curr_chrom = chromatin[enrichment == 1]

        # Take care of the experiments with NO domain calls
        if len(curr_chrom) == 0:
            chrom = np.zeros(shape=(1000, 130))
            seq = np.zeros(shape=(1000, 500, 4))
            raise Exception("Your experiment must have > 0 domain calls")
        else:
            seq = np.zeros(shape=(len(curr_chrom), 500, 4))
        # get embeddings
        embeddings = get_embeddings_low_mem(model, seq, curr_chrom)
        chrom_scores = embeddings[:, 1]
        curr = np.vstack((chrom_scores,
                          np.repeat(label, repeats=len(chrom_scores))))
        curr = np.transpose(curr)
        domain_scores.append(curr)

    dat = np.vstack(domain_scores)
    # Saving the data here in case we want to use it further:-
    np.savetxt(out_path + 'chrom_scores.txt', dat, fmt='%s')
    # Plotting
    dat = pd.read_csv(out_path + 'chrom_scores.txt', header=None,
                      sep=" ", names=['value', 'track'])
    sns.set_style('ticks')
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes', labelsize=12)
    fig, ax = plt.subplots()

    for idx, label in enumerate(hm_labels):
        # define a subplot
        plt.subplot(13, 1, idx+1)
        # subset data
        dat_at_label = dat[dat['track'] == label]
        sns.distplot(dat_at_label['value'], kde=False,
                     color='#ff1e56')
        plt.title(label, fontsize=10)
        plt.xlabel('')
        if idx < 11:
            plt.xticks([], [])
    fig.set_size_inches(4, 7)
    plt.subplots_adjust(hspace=0.75)
    fig.subplots_adjust(left=.20, bottom=.05, right=.99, top=.97)
    plt.savefig(out_path + '5A.pdf')
