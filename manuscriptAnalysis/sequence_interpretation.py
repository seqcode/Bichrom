import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from joint_embeddings import get_embeddings_low_mem

import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties
# from interpretNN.utils import make_onehot
import itertools
from collections import defaultdict
import pandas as pd
import re


def make_onehot(buf, seq_length):
    fd = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
    onehot = [fd[base] for seq in buf for base in seq]
    onehot_np = np.reshape(onehot, (-1, seq_length, 4))
    return onehot_np


def convert_to_dictionary(sequence):
    # Define a dictionary mapping the one-hot encoding back to letters
    l = []
    rev_dict = {0: 'A', 1: 'T', 2: 'G', 3: 'C'}
    for nucleotide in sequence:
        idx = np.argmax(nucleotide)
        l.append(rev_dict[idx])
    return l


def get_multiplicity_at_categories(seq_input, chromatin_input, motifs,
                                   model, outfile):
    """
    Divide sites into those with sequence scores less than and gt than the
    median sequence sub-network score.

    Save these sites,and calculates/prints the average occurence
    of the E-box k-mers.

    :param seq_input: seq_input
    :param chromatin_input: chromatin_input
    :param motifs: the set of k-mers to be scanned
    :param model: the trained model
    :param outfile: the prefix to the output path
    :return: None
    """

    # Get the bound embeddings:
    bound_embeddings = get_embeddings_low_mem(model, seq_input, chromatin_input)

    # get the median sequence score
    seq_scores = bound_embeddings[:, 0]
    median_bound_seqscore = np.median(seq_scores)

    # get the onehot vectors for the sequences in "low" and "high" categories
    onehot_high = seq_input[seq_scores > median_bound_seqscore]
    onehot_low = seq_input[seq_scores <= median_bound_seqscore]
    # convert the one-hot vectors to strings and save to files
    seq_low = [''.join(convert_to_dictionary(x)) for x in onehot_low]
    seq_high = [''.join(convert_to_dictionary(x)) for x in onehot_high]
    np.savetxt(outfile + 'seq_lessthan_median.txt', seq_low, fmt='%s')
    np.savetxt(outfile + 'seq_gtthan_median.txt', seq_high, fmt='%s')

    # calculate the average number of motifs in each sample

    def calc_no_of_motifs(sequence_set):
        no_of_motifs = []
        for vector in sequence_set:
            seq = convert_to_dictionary(vector)
            seq = ''.join(seq)
            match = []
            for motif in motifs:
                match.append(len(re.findall(motif, seq)))
            no_of_motifs.append(np.sum(match))
        print np.mean(no_of_motifs)

    calc_no_of_motifs(onehot_low)
    calc_no_of_motifs(onehot_high)


def plot_correlation(datapath, embedding, seq_input, out_a, out_b, motifs):
    """
    Plot correlation between the number of motifs/k-mers and sequence network
    scores

    :param datapath: path + prefix to input Ascl1 data
    :param embedding: the latent network embedding
    :param seq_input: seq_input
    :param out_a: outpath for figure A
    :param out_b: outpath for figure B
    :param motifs: set of motifs
    :return: None
    """

    num_of_motifs = []
    for vector in seq_input:
        seq = ''.join(convert_to_dictionary(vector))
        match = []
        for motif in motifs:
            match.append(len(re.findall(motif, seq)))
        num_of_motifs.append(np.sum(match))

    # Figure 1: Draw a histogram of counts
    # Treat all counts >4 as a single unit
    binned_dat = [x if x < 5 else 5 for x in num_of_motifs]
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=.20, bottom=.15, right=.95, top=.95)
    sns.distplot(binned_dat, bins=6, kde=False, norm_hist=True,
                 hist_kws={"rwidth": 0.75, 'edgecolor': 'black',
                           'color': '#2471A3',
                           'alpha': 1.0})
    plt.xticks(np.linspace(0.4, 4.6, 6), ['x=0', 'x=1', 'x=2', 'x=3', 'x=4', 'x>5'])
    plt.xlabel('Matches to CAGSTG kmers at\nAscl1 binding windows', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
    fig.set_size_inches(3.5, 4)
    sns.despine()
    plt.savefig(out_a)

    # Figure 2: Do boxplots correlating this with sequence sub-network scores:
    sequence_score = embedding[:, 0]
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=.15, bottom=.15, right=.95, top=.95)
    sns.boxplot(binned_dat, sequence_score, color='#D68910')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
    plt.xticks(range(6), ['x=0', 'x=1', 'x=2', 'x=3', 'x=4', 'x>5'])
    plt.xlabel('Matches to CAGSTG kmers\nat Ascl1 binding windows', fontsize=10)
    plt.ylabel('Sequence sub-network activations', fontsize=10)
    fig.set_size_inches(3.5, 4)
    plt.savefig(out_b)


def reverse_complement(kmer):
    rc_dict = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    reverse_kmer = [rc_dict[x] for x in kmer]
    comp_kmer = reverse_kmer[::-1]
    return ''.join(comp_kmer)


def random_sequences(num):
    idx = 0
    l = []
    while idx < num:
        idx = idx + 1
        indices = np.random.randint(0, 4, 500)
        seq_dict = {0: 'A', 1: 'T', 2: 'G', 3: 'C'}
        l.append([seq_dict[x] for x in indices])
    return l


def second_order_motifs(file_path, model, motif):
    # Define the motif
    # Create a dataset matching the following shape:
    # sequence : (1, 500, 4)
    # chromatin: (1, 140)
    simulated_chromatin = np.zeros((1, 130))
    # Start with creating a simple string sequence
    simulated_sequence = np.repeat('N', 500)
    simulated_sequence[250:256] = list(motif)
    # construct all possible 10 base-pair motifs
    # construct these motifs such that the 'CAGCTG' motif is flanked by all possible 2-mers on either side.
    bases = ['A', 'T', 'G', 'C']
    flanks = itertools.product(bases, repeat=4)
    score_list = []
    # kmer_dict is a default dict to store scores assigned to both f and rc sequences
    kmer_dict = defaultdict(list)
    # Score each flank
    for flank in flanks:
        word = flank[:2] + tuple(motif) + flank[2:]
        kmer = ''.join(word)
        # Embed motif in the 'zero' background word.
        simulated_sequence[245:255] = word
        simulated_sequence = list(simulated_sequence)
        sequence_onehot = make_onehot(simulated_sequence, 500)
        # Create a simulated input vector
        simulated_input = (sequence_onehot, simulated_chromatin)
        scores = get_embeddings_low_mem(model, sequence_onehot,
                                        simulated_chromatin)
        sequence_score = scores[0][0]
        score_list.append(sequence_score)
        if reverse_complement(kmer) in kmer_dict:
            kmer_dict[reverse_complement(kmer)].append(sequence_score)
        else:
            kmer_dict[kmer].append(sequence_score)
    # Aggregate scores based on first order k-mers
    k = []
    score_list = []
    for kmer, val in kmer_dict.iteritems():
        for scores in val:
            seq = list(kmer)
            seq[0] = 'N'
            seq[9] = 'N'
            k.append(''.join(seq))
            score_list.append(scores)
    # Save the data files..
    dat = np.transpose(np.vstack((k, score_list)))
    fp = open(file_path, "a")
    np.savetxt(fp, dat, fmt='%s')
    fp.close()
    return dat


def first_order_motifs(file_path, model, motif, num):
    # Goal: Embedding 8bp k-mers into a 1000 randomly generated sequences.
    # Unlike the second_order motifs, will do this in parallel because of the larger number of inputs
    # Constructing the flanks.
    bases = ['A', 'T', 'G', 'C']
    flanks = itertools.product(bases, repeat=2)
    seq_list = []
    kmer_list = []
    for flank in flanks:
        word = flank[:1] + tuple(motif) + flank[1:]
        # The kmer is the string version of the word
        kmer = ''.join(word)
        seq_generator = random_sequences(num)
        for sequence in seq_generator:
            sequence[246:254] = word
            simulated_sequence = list(sequence)
            sequence_onehot = make_onehot(simulated_sequence, 500)
            # creating a simulated input vector
            seq_list.append(sequence_onehot)
            kmer_list.append(kmer)
    X = np.reshape(seq_list, (16 * num, 500, 4))
    C = np.zeros(shape=(16 * num, 130))
    simulated_input = (X, C)
    score_list = get_embeddings_low_mem(model, simulated_input)
    # Now, I have scores for each embedding motif, as well as the motif list.
    # Now use a dictionary to get the reverse complements sorted
    kmer_dict = defaultdict(list)
    for kmer, score in zip(kmer_list, score_list):
        if reverse_complement(kmer) in kmer_dict:
            kmer_dict[reverse_complement(kmer)].append(score[0])
        else:
            kmer_dict[kmer].append(score[0])
    # Having dealt with the reverse complement, put this in lists
    k = []
    score_list = []
    for kmer, val in kmer_dict.iteritems():
        for scores in val:
            k.append(''.join(kmer))
            score_list.append(scores)
    # Saving the data files..
    dat = np.transpose(np.vstack((k, score_list)))
    fp = open(file_path, "a")
    np.savetxt(fp, dat, fmt='%s')
    fp.close()


def motifs_in_ns(file_path, model, motif):
    # Start with creating a simple string sequence
    simulated_sequence = np.repeat('N', 500)
    simulated_sequence[250:256] = list(motif)
    # construct all possible 8 base-pair motifs
    # construct these motifs such that the 'CAGCTG' motif is flanked by all possible 2-mers on either side.
    # 1. construct all possible 4-bp sequences
    bases = ['A', 'T', 'G', 'C']
    flanks = itertools.product(bases, repeat=2)
    seq_list = []
    kmer_list = []
    for flank in flanks:
        word = flank[:1] + tuple(motif) + flank[1:]
        kmer = ''.join(word)
        simulated_sequence[246:254] = word
        simulated_sequence = list(simulated_sequence)
        sequence_onehot = make_onehot(simulated_sequence, 500)
        seq_list.append(sequence_onehot)
        kmer_list.append(kmer)
    # Get scores
    X = np.reshape(seq_list, (16, 500, 4))
    C = np.zeros(shape=(16, 130))
    simulated_input = (X, C)
    score_list = get_embeddings_low_mem(model, simulated_input)

    # I have to consider reverse complements
    kmer_dict = defaultdict(list)
    for kmer, score in zip(kmer_list, score_list):
        if reverse_complement(kmer) in kmer_dict:
            kmer_dict[reverse_complement(kmer)].append(score[0])
        else:
            kmer_dict[kmer].append(score[0])
    # Putting it back into a list
    k = []
    score_list = []
    for kmer in kmer_dict.iterkeys():
        kmer_dict[kmer] = np.mean(kmer_dict[kmer])
    for kmer, val in kmer_dict.iteritems():
        k.append(''.join(kmer))
        score_list.append(val)
    # Saving the data files..
    dat = np.transpose(np.vstack((k, score_list)))
    fp = open(file_path, "a")
    np.savetxt(fp, dat, fmt='%s')
    fp.close()


def plot_kmer_scores(file_path, outfile):
    # plot boxplots for 2-D arrays of k-mers and scores
    dat = np.loadtxt(file_path, dtype=str)
    d = defaultdict(list)
    # Get all scores for a given k-mer
    for kmers, scores in dat:
        d[kmers].append(float(scores))
    # Sorting the k-mers based on their median scores
    l = []
    k = []
    for kmers in d:
        k.append(kmers)
        l.append(np.median(d[kmers]))
    order = np.argsort(l)
    kmers = np.array(k)[order]
    # Writing back to a list based on the above sort
    sorted_scores = []
    matched_kmers = []
    for kmer in kmers:
        for scores in d[kmer]:
            sorted_scores.append(scores)
            matched_kmers.append(kmer)
    dat = np.transpose(np.vstack((matched_kmers, sorted_scores)))
    # Plotting: Boxplots
    # Package: Seaborn

    fig, ax = plt.subplots()
    fig.subplots_adjust(left=.15, bottom=.35, right=.95, top=.90)
    # Plot
    ## CHANGING STUFF MARCH 2020
    sns.stripplot(dat[:, 0], dat[:, 1].astype(float), color='#cd8d7b')
    plt.xticks(range(len(kmers)), kmers, rotation=45, fontsize=8, ha='right')
    plt.yticks(fontsize=10)
    # Set axes
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
    plt.xlabel('All 8-mers embedded in \n 10k randomly generated 500bp windows',
               fontsize=12)
    plt.ylabel('Sequence sub-network activations', fontsize=12)
    fig.set_size_inches(8, 4)
    sns.despine()
    plt.savefig(outfile)


def plot_dots(file_path, outfile):
    dat = np.loadtxt(file_path, dtype=str)
    scores = dat[:, 1].astype(float)
    order = np.argsort(scores)
    scores = scores[order]
    kmers = dat[:, 0][order]
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=.15, bottom=.25, right=.95, top=.95)
    plt.plot(range(len(scores)), scores, "o", ms=4, color='#cd8d7b')
    plt.xticks(range(len(scores)), kmers, rotation=45, fontsize=8, ha='right')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
    plt.xlabel("8-mers", fontsize=12)
    plt.ylabel("Sequence sub-network activations", fontsize=12)
    fig.set_size_inches(8, 2.5)
    sns.despine()
    plt.ylim(-5.5, 3)
    plt.savefig(outfile)


def plot_multiplicity(model, motif, outfile, no_of_repeats):
    sequences = random_sequences(no_of_repeats)
    # Generate 10000 random sequences:
    C = np.zeros(shape=(1, 130))
    data = np.zeros(shape=(no_of_repeats, 6))
    for seq_id, sequence in enumerate(sequences):
        print seq_id
        # Iterate over the 10000 sequences.
        idx = 0
        locations = np.random.randint(5, 495, 1)  # Putting it in a single location for now
        for loc in locations:
            # Note: The 5 to 495 is to make sure that I don't go out
            # of sequence bounds while inserting sequence
            # Append 1, 2, 3, 4 and 5  motifs
            # Check the baseline score of the generated sequence
            sequence_onehot = make_onehot(sequence, 500)
            # 1. Check score with no motif
            data[seq_id, idx] = get_embeddings_low_mem(model, sequence_onehot, C)[0][0]
            idx = idx + 1
            sequence[loc - 3:loc + 3] = list(motif)
            sequence = list(sequence)
            sequence_onehot = make_onehot(sequence, 500)
            data[seq_id, idx] = get_embeddings_low_mem(model, sequence_onehot, C)[0][0]
            idx = idx + 1
            while True:
                if idx == 6:
                    break
                else:
                    # Add motifs and append to lists
                    offset = np.random.randint(5, 500) # Add atleast 5
                    # to the left edge of the motif
                    if 5 < idx + offset < 495:
                        curr = idx + offset
                        sequence[curr-3: curr+3] = list(motif)
                        sequence = list(sequence)
                        sequence_onehot = make_onehot(sequence, 500)
                        data[seq_id, idx] = get_embeddings_low_mem(model, sequence_onehot, C)[0][0]
                        idx += 1
                    else:
                        pass

    # Converting to a tidy data
    data = pd.DataFrame(data)
    data = pd.melt(data)
    # Plotting
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=.15, bottom=.15, right=.95, top=.95)
    sns.boxplot(data['variable'], data['value'], color='#D2B4DE')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
    plt.xticks(range(6), [0, 1, 2, 3, 4, 5])
    plt.xlabel('Number of embedded CAGSTG motifs in simulated data',
               fontsize=10)
    plt.ylabel('Sequence sub-network activations', fontsize=10)
    fig.set_size_inches(3.5, 4)
    plt.savefig(outfile)