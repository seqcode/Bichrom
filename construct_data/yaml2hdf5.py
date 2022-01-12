description = """
Save bichrom training/validation/test datasets in HDF5 format to enable fast access during training

Usage:

    $ python yaml2hdf5.py bichrom.yaml bichrom.h5

"""

import os
import sys
import h5py
import numpy as np
import pandas as pd

from yaml import load, Loader

DNA2index = {
        "A": 0,
        "T": 1,
        "G": 2,
        "C": 3
}

def dna2onehot(dnaSeqs):
    numSeq = len(dnaSeqs)        
    seqLen = len(dnaSeqs[0])

    # initialize the matrix to seqlen x 4
    seqMatrixs = np.zeros((numSeq,seqLen,4), dtype=int)
    # change the value to matrix
    for i in range(0,numSeq):
        dnaSeq = dnaSeqs[i].upper()
        seqMatrix = seqMatrixs[i]
        for j in range(0,seqLen):
            try:
                seqMatrix[j, DNA2index[dnaSeq[j]]] = 1
            except KeyError:
                continue
    return seqMatrixs

def yaml2hdf5(yml, h5file):

    data = load(open(yml, 'r'), Loader = Loader)    # load yaml
    h5 = h5py.File(h5file, 'a', libver='latest')    

    for sets in ["train", "val", "test"]:

        # initiate HDF5 object
        h5.create_group(f"{sets}/chromatin_tracks/")

        # one-hot coding DNA sequence
        with open(data[f"{sets}"]["seq"], 'r') as f:
            seqs = []
            for line in f:
                seqs.append(line.strip())
            seqs_onehot = dna2onehot(seqs)
        seq_ds = h5[f"{sets}"].create_dataset("seq", data=seqs_onehot, chunks=True)  # save
        seq_ds.attrs.create("src", data[f"{sets}"]["seq"])  # keep track of source file

        # chromatin tracks
        for ct in data[f"{sets}"]["chromatin_tracks"]:
            ct_id = os.path.basename(ct).split(".")[1]  # get id
            f = np.loadtxt(ct)
            ct_ds = h5[f"{sets}/chromatin_tracks"].create_dataset(ct_id, data=f, chunks=True)
            ct_ds.attrs.create("src", ct)   # keep track of source file

        # labels
        labels = np.loadtxt(data[f"{sets}"]["labels"], dtype=int)
        labels_ds = h5[f"{sets}"].create_dataset("labels", data=labels)
        labels_ds.attrs.create("src", data[f"{sets}"]["seq"])   # keep track of source file

    h5.swmr_mode = True     # SWMR mode on
    h5.close()

def main():
    # load arguments
    if len(sys.argv[1:]) != 2:
        print(description)
        raise Exception("Require exactly 2 arguments!")
    else:
        yml, h5file = sys.argv[1:]

    yaml2hdf5(yml, h5file)

if __name__ == "__main__":
    main()

