import numpy as np
import pandas as pd


def find_hills(grad_star_inp, datapath, out_path):
    """
    A simple peak finding function given a 1-D saliency signal.
    Parameters:
        grad_star_inp: A N * L * 4 matrix containing the an importance score per
        position per nucleotide. (This can be derived from integrated gradients,
        DeepLIFT etc.)
        datapath: Path to the bound data
        out_path: Output file path.
    """
    # Sum information at each nucleotide; this will result in a matrix
    # of dimension N * L.
    importance = np.sum(grad_star_inp, axis=2)
    # expected shape: N * L (500) (where N is number of bound sites)
    print(importance.shape)
    # using a straight-forward approach: assigning hills/peaks using the mode.
    peaks = np.argmax(importance, axis=1)
    bedfile = pd.read_csv(datapath + '.bound.bed', header=None, sep="\t",
                          names=['chr', 'start', 'stop'])
    # save the hills as an events file
    events_chr = bedfile['chr']
    position = bedfile['start'] + peaks
    hills_dat = pd.concat([events_chr, position], axis=1)
    hills_dat.to_csv(out_path + '.hills', sep=':', index=False, header=False)
