""" Helper module with methods for one-hot sequence encoding and generators to
to enable whole genome iteration """

import numpy as np


class Sequence:
    """ Methods for manipulation of DNA Sequence """
    def __init__(self):
        pass

    @staticmethod
    def map(buf, seqlen):
        """ Converts a list of sequences to a one hot numpy array """
        fd = {'A' : [1, 0, 0, 0], 'T': [0,1,0,0], 'G' : [0,0,1,0],'C': [0,0,0,1], 'N': [0,0,0,0],
              'a' : [1, 0, 0, 0], 't': [0,1,0,0], 'g': [0,0,1,0], 'c': [0,0,0,1],
              'n': [0,0,0,0]}
        onehot = [fd[base] for seq in buf for base in seq]
        onehot_np = np.reshape(onehot,(-1,seqlen,4))
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
    if dtype == "seq":
        # use Sequence methods
        handler = Sequence
    else:
        # use Chromatin methods
        handler = Chromatin
    return handler


def train_generator(filename, batchsize, seqlen, dtype, iterflag):
    """ A generator to return a batch of training data, while iterating over the file in a loop. """
    handler = assign_handler(dtype)
    with open(filename, "r") as fp:
        line_index = 0
        buf = [] # buf is my feature buffer
        while True:
            for line in fp:
                if line_index < batchsize:
                        handler.add_to_buffer(buf, line)
                        line_index += 1
                else:
                    yield handler.map(buf, seqlen)
                    buf = [] # clean buffer
                    handler.add_to_buffer(buf, line)
                    line_index = 1 # reset line index
            if iterflag == "repeat":
                # reset file pointer
                fp.seek(0)
            else:
                yield handler.map(buf, seqlen)
                break
