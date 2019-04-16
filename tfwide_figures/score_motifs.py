import numpy as np
from keras.models import load_model
import keras.backend as K


def reverse_complement(kmer):
    rc_dict = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    reverse_kmer = [rc_dict[x] for x in kmer]
    comp_kmer = reverse_kmer[::-1]
    return ''.join(comp_kmer)


def make_onehot(buf, seq_length):
    fd = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
    onehot = [fd[base] for seq in buf for base in seq]
    onehot_np = np.reshape(onehot, (-1, seq_length, 4))
    return onehot_np


def random_sequences(num):
    idx = 0
    l = []
    while idx < num:
        idx = idx + 1
        indices = np.random.randint(0, 4, 500)
        seq_dict = {0: 'A', 1: 'T', 2: 'G', 3: 'C'}
        l.append([seq_dict[x] for x in indices])
    return l


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
    return abs_w * activations_rs


def calculate_motif_scores(model, motif_onehot, num, randomseqs, chromsize):
    # Goal: Embedding 8bp k-mers into a 1000 randomly generated sequences.
    # Unlike the second_order motifs, will do this in parallel because of the larger number of inputs
    # Constructing the flanks.
    seq_list = []
    kmer_list = []
    for sequence in randomseqs:
        simulated_sequence = list(sequence)
        sequence_onehot = make_onehot(simulated_sequence, 500)
        # print sequence_onehot[0, 0:15, :]
        # print motif_onehot.shape
        motif_onehot = motif_onehot[:, [0, 3, 2, 1]]
        sequence_onehot = sequence_onehot.astype(float)
        sequence_onehot = np.zeros_like(sequence_onehot)
        sequence_onehot[0, 250:250+motif_onehot.shape[0], :] = motif_onehot
        # creating a simulated input vector
        seq_list.append(sequence_onehot)
        kmer_list.append(motif_onehot)
    X = np.reshape(seq_list, (num, 500, 4))
    simulated_chromatin = np.zeros(shape=(num, chromsize * 10))
    simulated_input = (X, simulated_chromatin)
    scores = get_activations_low_mem(model, simulated_input)
    print np.mean(scores[:, 0])


def motif_scores(datapath):

    protein_list = ['Ascl1', 'Ngn2', 'CDX2', 'FOXA1', 'DUXBL', 'SOX15', 'SOX2', 'SIX6', 'DLX6', 'HLF', 'RHOX11']
    background = random_sequences(1000)
    for protein in protein_list:
        # model
        try:
            modelpath_curr = datapath + protein + '.NIH3T3.chrom.adam.10.hdf5'
            print modelpath_curr
            model = load_model(modelpath_curr)
            # get a list of random sequences
            print protein
            for innerprotein in protein_list:
                # Load motifs for all TFs
                motif = np.loadtxt(datapath + innerprotein + '.txt')
                calculate_motif_scores(model, motif, 1000, background, 1)
        except:
            modelpath_curr = datapath + protein + '.ES.chrom.adam.05.hdf5'
            print modelpath_curr
            model = load_model(modelpath_curr)
            # get a list of random sequences
            motif = np.loadtxt(datapath + 'Ascl1.txt')
            print protein
            for innerprotein in protein_list:
                # Load motifs for all TFs
                motif = np.loadtxt(datapath + innerprotein + '.txt')
                calculate_motif_scores(model, motif, 1000, background, 13)
