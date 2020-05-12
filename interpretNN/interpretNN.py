import argparse
import numpy as np
from subprocess import call

# Import from utils
from utils import get_bound_data
from utils import get_random_subset
from utils import load_data

# Importing from Keras
from keras.models import load_model

# joint embeddings
from joint_embeddings import get_embeddings, get_embeddings_low_mem
from joint_embeddings import plot_1d_chrom, plot_1d_seq
from joint_embeddings import plot_embeddings
from joint_embeddings import plot_embeddings_bound_only


def get_data(datapath):
    """
    Takes path to data directory, and writes bound + shuffled unbound files.
    Parameters:
        datapath: Path to the input test data, eg. X
            Files in the directory should include:
            - X.seq
            - X.chromtracks
            - X.labels
            - X.bed
    Returns:
        bound and unbound seq and chromatin data
    """
    # load data into memory
    seq_data, chromatin_data, labels, bed_array = load_data(datapath)
    # subset bound data
    b_seq_data, b_chromatin_data, b_bed_array = get_bound_data(seq_data=seq_data,
                                                               chromatin_data=chromatin_data,
                                                               bed_array=bed_array,
                                                               test_labels=labels)
    # save the bound files for any future use:
    np.save(datapath + '.bound.seq.npy', b_seq_data)
    np.save(datapath + '.bound.chromtracks.npy', b_chromatin_data)
    # using pandas here due to previous annoying fmt errors with np savetxt.
    b_bed_array.to_csv(datapath + '.bound.bed', sep='\t',
                       header=False, index=False)

    # select a random subset of unbound data (with a fixed seed)
    # i can change the subset size here, or define that separately.
    ub_seq_data, ub_chromatin_data = get_random_subset(seq_data=seq_data,
                                                       chromatin_data=chromatin_data,
                                                       test_labels=labels, subset_size=20000)
    return b_seq_data, b_chromatin_data, ub_seq_data, ub_chromatin_data


def embed(outdir, model, b_seq_data, b_chromatin_data, ub_seq_data,
          ub_chromatin_data):
    """
    Loads the bound data & extracts the joint embeddings
    This function takes as input path to the input data, as well as a loaded model.
    The function extracts the network embeddings into the final logistic node.

    Parameters:
        outpath (str): output directory
        model (Model): trained model
        b_seq_data (ndarray): onehot sequence tensor
        b_chromatin_data (ndarray): bound onehot chromatin tensor
        ub_seq_data (ndarray): unbound onehot sequence tensor
        ub_chromatin_data (ndarray): unbound chromatin tensor

    Returns: None
        Saves both the positive and negative embedding matrices
        and plots to the specified output directory.
    """
    # Extract and save the embeddings of bound and unbound sets to file.
    embeddings_bound = get_embeddings(model, b_seq_data, b_chromatin_data)
    # Extract and save the embeddings of a random negative set
    embeddings_unbound = get_embeddings_low_mem(model, ub_seq_data,
                                                ub_chromatin_data)

    # Creating the outfile
    call(['mkdir', outdir])
    out_path = outdir + '/embeddings/'
    call(['mkdir', out_path])
    # Saving the embeddings to outfile
    np.savetxt(out_path + "bound_embedding.txt", embeddings_bound)
    np.savetxt(out_path + 'unbound_embedding.txt', embeddings_unbound)

    # Plotting
    # Plot 2-D embeddings: Bound + Unbound Sites
    plot_embeddings(out_path, embeddings_bound, embeddings_unbound)
    # Plot 2-D embeddings: Bound only
    plot_embeddings_bound_only(out_path, embeddings_bound, embeddings_unbound)
    # Plot marginal 1D distributions:
    plot_1d_seq(out_path, embeddings_bound, embeddings_unbound)
    plot_1d_chrom(out_path, embeddings_bound, embeddings_unbound)


def main():
    # TO DO:
    # SET UP AN EXAMPLE RUN SCRIPT/README
    parser = argparse.ArgumentParser(description='Learn Latent Embeddings',
                                     prog='interpretNN')
    # adding parser arguments
    parser.add_argument('model', help='Path to trained M-SC models')
    parser.add_argument('datapath', help='Path to test data')
    parser.add_argument('out', help='outdir')

    args = parser.parse_args()

    # Identify the best model:
    model = load_model(args.model)

    # load bound & unbound data
    b_seq_data, b_chromatin_data, ub_seq_data, ub_chromatin_data = get_data(args.datapath)
    print b_seq_data
    print b_chromatin_data
    print ub_seq_data
    print ub_chromatin_data
    # extract, save and plot 2-D embeddings
    embed(outdir=args.out, model=model, b_chromatin_data=b_chromatin_data,
          b_seq_data=b_seq_data, ub_chromatin_data=ub_chromatin_data,
          ub_seq_data=ub_seq_data)


if __name__ == "__main__":
    main()
