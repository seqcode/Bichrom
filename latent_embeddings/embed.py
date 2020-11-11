"""
Use this script to embed a TF's bound sites into a 2-D plane.
This module requires that you have a trained Bichrom model (see README and trainNN).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pandas as pd
from subprocess import call
import get_latent_embeddings as le
from utils import load_data
from tensorflow.keras.models import load_model


def embed_data(model, seq_data, chromatin_data, outdir):
    """
    Loads the bound data & extracts the joint embeddings
    This function takes as the input sequence and chromatin data,
    as well as a loaded model.
    The function extracts the network embeddings prior to the final logistic node.

    Parameters:
        outpath (str): output directory
        model (Model): trained model
        seq_data (str): sequence
        chromatin_data (ndarray): bound chromatin tensor

    Returns: None
        Saves both the positive and negative embedding matrices
        and plots to the specified output directory.
    """
    # Extract and save the embeddings of bound and unbound sets to file.
    embeddings = le.get_embeddings_low_mem(model, seq_data, chromatin_data)
    # Creating the outfile
    call(['mkdir', outdir])
    out_path = outdir + '/embeddings/'
    call(['mkdir', out_path])
    # Saving the embeddings to outfile
    np.savetxt(out_path + "latent_embeddings.txt", embeddings)
    # Plotting
    plot_2d_embeddings(out_path, embeddings)


def plot_2d_embeddings(out_path, embedding):
    """
    Plot the joint sequence and chromatin embeddings for bound vs. unbound loci.
    Parameters:
        out_path: Directory to store the output figures
        embedding: Embeddings for the positive or bound set
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
    plt.savefig(out_path + "latent_embeddings.pdf")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=
                                     'Derive and plot 2-D embeddings')
    parser.add_argument('-model',
                        help='Trained bichrom model',
                        required=True)
    parser.add_argument('-seq',
                        help='Sequence file, each line contains sequences '
                             'associated with one genomic window',
                        required=True)
    parser.add_argument('-chrom',
                        nargs='+',
                        help='List of files, each containing chromatin features, '
                             'associated with one genomic window per line',
                        required=True)
    parser.add_argument('-length',
                        help='Length of training windows', type=int,
                        required=True)
    parser.add_argument('-outdir', help='Output directory',
                        required=True)

    args = parser.parse_args()

    seq_onehot, chromatin_features = load_data(args.seq,
                                               args.chrom,
                                               window_size=args.length)
    model = load_model(args.model)
    embed_data(model=model, seq_data=seq_onehot,
               chromatin_data=chromatin_features, outdir=args.outdir)