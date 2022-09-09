#!python3

import os
import argparse
import numpy as np
import pandas as pd

import pyfasta
import pyBigWig
import logging

from tensorflow.keras.models import load_model

def dna2onehot(dnaSeq):
    DNA2index = {
        "A": 0,
        "T": 1,
        "G": 2,
        "C": 3
    }

    seqLen = len(dnaSeq)

    # initialize the matrix to seqlen x 4
    seqMatrixs = np.zeros((seqLen,4), dtype=int)
    # change the value to matrix
    dnaSeq = dnaSeq.upper()
    for j in range(0,seqLen):
        try:
            seqMatrixs[j, DNA2index[dnaSeq[j]]] = 1
        except KeyError as e:
            continue
    return seqMatrixs

def get_data(chunk, genome_pyfasta, bigwigs, nbins):
    seqs = []
    mss = []
    for item in chunk.itertuples():
        # get seq info
        seq = genome_pyfasta[item.chrom][int(item.start):int(item.end)]
        seq_onehot = dna2onehot(seq)

        # get chrom info
        ms = []
        try:
            for idx, bigwig in enumerate(bigwigs):
                m = (np.nan_to_num(bigwig.values(item.chrom, item.start, item.end))
                                        .reshape((nbins, -1))
                                        .mean(axis=1, dtype=float))
                ms.append(m)
            ms = np.concatenate(ms)
        except RuntimeError as e:
            logging.warning(e)
            logging.warning(f"Chromatin track doesn't have information in {item}")
            raise e

        # store
        seqs.append(seq_onehot); mss.append(ms)

    seqs = np.stack(seqs); mss = np.stack(mss)
    return {"seq": seqs, "chrom_input": mss}

def predict_generator(bed_file, fasta, bigwig_files, nbins, batchsize=128):
    """
    Generator that iterate through the bed file until the end
    """
    bed_chunks = pd.read_table(bed_file, header=None, usecols=[0, 1, 2], names=["chrom", "start", "end"], chunksize=batchsize)
    genome_pyfasta = pyfasta.Fasta(fasta)
    bigwigs = [pyBigWig.open(bw) for bw in bigwig_files]

    for chunk in bed_chunks:
        try:
            input = get_data(chunk, genome_pyfasta, bigwigs, nbins)
        except RuntimeError as e:
            continue
        yield input

def main():
    parser = argparse.ArgumentParser(description='Use Bichrom model for prediction given bed file')
    parser.add_argument('-mseq', required=True,
                        help='Sequence Model')
    parser.add_argument('-msc', required=True,
                        help='Bichrom Model')
    parser.add_argument('-fa', help='The fasta file for the genome of interest', required=True)
    parser.add_argument('-chromtracks', nargs='+', help='A list of BigWig files for all input chromatin experiments, please follow the same order of training data', required=True)
    parser.add_argument('-nbins', type=int, help='Number of bins for chromatin tracks', required=True)
    parser.add_argument('-prefix', required=True, help='Output prefix')
    parser.add_argument('-bed', required=True, help='bed file describing region used for prediction')
    args = parser.parse_args()

    mseq = load_model(args.mseq)
    msc = load_model(args.msc)
    pred_dataset = predict_generator(args.bed, args.fa, args.chromtracks, args.nbins)

    # get predictions
    mseq_probs = []; msc_probs = []
    for input in pred_dataset:
        mseq_prob = mseq(input, training=False)
        msc_prob = msc(input, training=False)
        mseq_probs.append(mseq_prob)
        msc_probs.append(msc_prob)
    mseq_probs = np.concatenate(mseq_probs)
    msc_probs = np.concatenate(msc_probs)
    
    # save to file
    with open(args.prefix + "mseq_prob.txt", "w") as fmseq, open(args.prefix + "msc_prob.txt", "w") as fmsc:
        np.savetxt(fmseq, mseq_probs)
        np.savetxt(fmsc, msc_probs)


if __name__ == "__main__":
    main()