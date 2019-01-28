# The idea here is to use mini-batches matched for accessibility status
""" Helper module with methods for one-hot sequence encoding and generators to
to enable whole genome iteration """

from __future__ import division
import numpy as np


class Sequence:
    """ Methods for manipulation of DNA Sequence """
    def __init__(self):
        pass

    @staticmethod
    def map(buf, seqlen):
        """ Converts a list of sequences to a one hot numpy array """
        fd = {'A' : [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G' : [0,0,1,0],'C': [0,0,0, 1], 'N': [0,0,0,0]}
        onehot = [fd[base] for seq in buf for base in seq]
        onehot_np = np.reshape(onehot, (-1,seqlen,4))
        return onehot_np

    @staticmethod
    def add_to_buffer(buf, line):
        buf.append(line.strip())


class Chromatin:
    """ Methods for manipulating discrete chromatin tag counts/ domain calls"""
    def __init__(self):
        pass

    @staticmethod
    def map(buf, seqlen):
        return np.array(buf)

    @staticmethod
    def add_to_buffer(buf, line):
        chrom = line.strip().split()
        val = [float(x) for x in chrom]
        buf.append(val)


def assign_handler(dtype):
    """ Choosing class based on input file type"""
    if dtype == str:
        # use Sequence methods
        handler = Sequence
    else:
        # use Chromatin methods
        handler = Chromatin
    return handler


def create_random_indices(batchsize, labels, domains):
    # Loading the labels and domains
    # Finding the total number of instances in each set ( for example. accessible and bound )
    set_ba = np.sum(np.logical_and(labels == 1, domains == 1))
    set_bia = np.sum(np.logical_and(labels == 1, domains == 0))
    set_ua = np.sum(np.logical_and(labels == 0, domains == 1))
    set_uia = np.sum(np.logical_and(labels == 0, domains == 0))
    # Using this length to create a vector of indices, which can then be used to choose my negative set.
    l = []
    s = []
    for vec in set_ba, set_bia, set_ua, set_uia:
        indices = np.random.choice(vec, size=int(batchsize/4))  # 200 each then in this case
        sup_idx = np.random.choice(vec, size=int(batchsize/2))
        l.append(indices)
        s.append(sup_idx)
    # Return two list that contains two different index sets to be used for all four subsets.
    return l, s


def create_batches(data_array, seqlen, dtype, index_list, perm, labels, domains, datflag):
    """ A generator to return a batch of training data, while iterating over the file in a loop. """
    # The accessibility and binding labels have already been loaded in the main script.
    # The idea is to not load any of the larger data-sets more than once.
    handler = assign_handler(dtype)
    # Randomly choosing a flag leads to not complete alternation between batches.
    flag = int(np.random.randint(0, 2, 1))
    # Creating sets from the data:-
    set_ba = data_array[np.logical_and(labels == 1, domains == 1)]
    set_bia = data_array[np.logical_and(labels == 1, domains == 0)]
    set_ua = data_array[np.logical_and(labels == 0, domains == 1)]
    set_uia = data_array[np.logical_and(labels == 0, domains == 0)]
    # Get the random index lists for each of the four vectors.
    # This is done outside this function because I need constant indices across sequence and acc generators.
    main, sup = index_list
    vec_ba, vec_bia, vec_ua, vec_uia = main
    s_vec_ba, s_vec_bia, s_vec_ua, s_vec_uia = sup
    # This is an extra list of vector indices that may be useful training set manipulation.
    # Creating mini-batches for training:
    # I need my IA set to have some representation from A regions.
    if flag == 0:
        # Initial strategy: IA (bound) + IA (unbound)
        # Modification: Adding 1 part more of A (unbound)
        # This is to account for any low ATAC regions that may have been called IA but really do have some A.
        # 1 (B_IA) + 2(UB_IA) + 1(UB_A)
        curr_b = set_bia[vec_bia] # 1/4
        curr_u = set_uia[s_vec_uia] # 2/4
        curr_ua = set_ua[vec_ua] # 1/4 (Total adds up to the batchsize)
        if datflag == 'chromtracks':
            batch = np.vstack((curr_b, curr_u, curr_ua))
        else:
            batch = np.hstack((curr_b, curr_u, curr_ua))
        buf = batch[perm]
        return handler.map(buf, seqlen)
    else:
        # Here , doubling the accessible negative set.
        # 1 (B_A) + 3 (UB_A)
        curr_b = set_ba[vec_ba] # 1/4
        curr_u = set_ua[vec_ua] # 1/4
        curr_ua = set_ua[s_vec_ua] # 2/4 (Total adds up to the batchsize)
        if datflag == 'chromtracks':
            batch = np.vstack((curr_b, curr_u, curr_ua))
        else:
            batch = np.hstack((curr_b, curr_u, curr_ua))
        buf = batch[perm]
        return handler.map(buf, seqlen)
