import numpy as np
import pandas as pd


def make_onehot(sequences, seq_length):
    """
    Converts a sequence string into a one-hot encoded array
    """
    fd = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0],
          'C': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
    onehot = [fd[base] for seq in sequences for base in seq]
    onehot_np = np.reshape(onehot, (-1, seq_length, 4))
    return onehot_np


def load_data(seq_data_file, chromatin_data_files, window_size):
    """
    Loads the input user-provided data.
    """
    # Load data (without using a generator)
    # Note: Switching to using Pandas instead of numpy loadtxt due to
    # the faster pandas load time.
    # sequence
    seq = pd.read_csv(seq_data_file, header=None,
                      names=['seq_str'])
    seq = seq['seq_str'].values
    seq_dat_onehot = make_onehot(list(seq), window_size)
    # prior chromatin (merge user provided chromatin data)
    chromatin_data = []
    for chromatin_data_file in chromatin_data_files:
        dat = pd.read_csv(chromatin_data_file, delim_whitespace=True,
                          header=None)
        chromatin_data.append(dat)
    merged_chromatin_data = pd.concat(chromatin_data, axis=1).values
    return seq_dat_onehot, merged_chromatin_data