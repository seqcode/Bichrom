import numpy as np
import pandas as pd

np.random.seed(1)


def make_onehot(buf, seq_length):
    fd = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
    onehot = [fd[base] for seq in buf for base in seq]
    onehot_np = np.reshape(onehot, (-1, seq_length, 4))
    return onehot_np


def load_data(datapath):
    # Load test data (without using a generator)
    # Note: Switching to using Pandas instead of numpy loadtxt due to
    # the faster pandas load time.
    # sequence
    seq = pd.read_csv(datapath + '.seq', header=None)
    seq = np.array(seq).reshape(-1,)
    seq_dat_onehot = make_onehot(seq, 500)
    # prior chromatin
    chrom_dat = pd.read_csv(datapath + ".chromtracks", header=None,
                            delim_whitespace=True)
    # labels
    labels = pd.read_csv(datapath + '.labels', header=None)
    labels = np.array(labels).reshape(-1,)
    # bedfile
    bed_array = pd.read_csv(datapath + '.bed', delimiter='\t', header=None)
    return seq_dat_onehot, chrom_dat.values, labels, bed_array


def get_bound_data(seq_data, chromatin_data, test_labels, bed_array):
    """
    Take as input the test data, and returns the bound subset.
    Parameters:
        seq_data (ndarray): shape: N * 500 (seq_len) * 4
        chromatin_data (ndarray): shape: N * 10 * no_of_Chrom_Tracks
        test_labels (ndarray): shape: N * 1
        bed_array (ndarray): shape: N * 3

    Returns:
        bound seq_data and bound chromatin_data
    """
    # identify bound sites.
    bound_indices = np.array(test_labels == 1)
    # subset data
    bound_seq_data = seq_data[bound_indices]
    bound_chromatin_data = chromatin_data[bound_indices]
    bound_bed_array = bed_array.loc[bound_indices]
    return bound_seq_data, bound_chromatin_data, bound_bed_array


def get_random_subset(seq_data, chromatin_data, test_labels, subset_size):
    """
    Take as input the test data, and returns the a randomly selected
    unbound subset.
    Parameters:
        seq_data (ndarray): shape: N * 500 (seq_len) * 4
        chromatin_data (ndarray): shape: N * 10 * no_of_Chrom_Tracks
        test_labels (ndarray): shape: N * 1

    Returns:
        a subset of unbound seq_data and unbound chromatin_data
    """
    # get unbound labels
    unbound_indices = np.array(test_labels == 0)
    #
    unbound_seq_data = seq_data[unbound_indices]
    unbound_chromatin_data = chromatin_data[unbound_indices]

    # randomly sample indexes used to sample
    upper_lim = int(np.sum(unbound_indices))
    selected_indices = np.random.randint(0, high=upper_lim, size=subset_size)
    return unbound_seq_data[selected_indices], \
           unbound_chromatin_data[selected_indices]