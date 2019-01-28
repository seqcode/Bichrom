import numpy as np


def make_onehot(buf, seq_length):
    fd = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
    onehot = [fd[base] for seq in buf for base in seq]
    onehot_np = np.reshape(onehot, (-1, seq_length, 4))
    return onehot_np


def load_data(datapath):
    # The idea is to load everything without using a generator
    sequence = np.loadtxt(datapath + ".seq", dtype=str)
    X = make_onehot(sequence, 500)
    # Loading the chromatin data
    C = np.loadtxt(datapath + ".chromtracks", dtype=float)
    # Loading the labels
    labels = np.loadtxt(datapath + ".labels", dtype=int)
    bedfile = np.loadtxt(datapath + '.bed', dtype=str)
    # Is there anything else I need now? No don't think so. # Cool!
    return X, C, labels, bedfile


def get_bound_data(datapath):
    loaded_data = load_data(datapath)
    X, C, labels, bedfile = loaded_data
    # Now I need to subset these guys into bound sites only
    bound_indices = np.array(labels == 1)
    print bound_indices
    # Subset data
    Xbound = X[bound_indices]
    Cbound = C[bound_indices]
    labelsbound = labels[bound_indices]  # This is primarily for code compatibility. I do not need to use this.
    bedBound = bedfile[bound_indices]
    # save these bound files as re-usable outputs:
    np.save(datapath + '.bound.seq.npy', Xbound)
    np.save(datapath + '.bound.chromtracks.npy', Cbound)
    np.save(datapath + '.bound.labels.npy', labelsbound)
    np.savetxt(datapath + '.bound.bed', bedBound, fmt='%s')
    # return
    return Xbound, Cbound


def get_random_sample(datapath):
    loaded_data = load_data(datapath)
    X, C, labels, bedfile = loaded_data
    # get unbound labels
    unbound_indices = np.array(labels == 0)
    X_unbound = X[unbound_indices]
    C_unbound = C[unbound_indices]
    # randomly sample from a uniform?
    high = int(np.sum(np.sum(unbound_indices)))  # choose from the entire set
    rint = np.random.randint(0, high=high, size=1500)
    # return & save the randomly selected unbound sets.
    np.save(datapath + '.unbound.random.seq.npy', X_unbound[rint])
    np.save(datapath + '.unbound.random.chrom.npy', C_unbound[rint])
    return X_unbound[rint], C_unbound[rint]


def load_bound_data(datapath):
    # The idea is to load everything without using a generator
    X = np.load(datapath + ".bound.seq.npy") # already in one hot
    # Loading the chromatin data
    C = np.load(datapath + ".bound.chromtracks.npy")
    # Loading the labels
    labels = np.load(datapath + ".bound.labels.npy")
    # Is there anything else I need now? No don't think so. # Cool!
    return X, C


def get_random_sample_shuffled(datapath_to_random_sample):
    sequence = np.loadtxt(datapath_to_random_sample + ".seq", dtype=str)
    X = make_onehot(sequence, 500)
    # Loading the chromatin data
    C = np.loadtxt(datapath_to_random_sample + ".chromtracks", dtype=float)
    # Is there anything else I need now? No don't think so. # Cool!
    return X, C


def load_network_probabilities(datapath):
    chrom_probs = np.loadtxt(datapath + ".bound.chrom.probs")
    return chrom_probs