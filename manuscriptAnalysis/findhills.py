import numpy as np
import pandas as pd


def find_hills(grad_star_inp, datapath, out_path):

    # sum across nucleotides
    importance = np.sum(grad_star_inp, axis=2)
    # expected shape: N * 500 (where N is number of bound sites)
    print importance.shape

    # using a straight-forward approach: assigning max values to hills
    peaks = np.argmax(importance, axis=1)
    bedfile = pd.read_csv(datapath + '.bound.bed', header=None, sep="\t",
                          names=['chr', 'start', 'stop'])

    # make events file
    events_chr = bedfile['chr']
    position = bedfile['start'] + peaks
    hills_dat = pd.concat([events_chr, position], axis=1)
    hills_dat.to_csv(out_path + '.hills', sep=':', index=False, header=False)
