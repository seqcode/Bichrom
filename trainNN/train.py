"""
Top-level function to train, test and plot performance metrics for:
- sequence-only network
- bimodal sequence-chromatin network

The module performs the following functions in order:

1. Train a sequence-only network.
2. Train a sequence and prior chromatin network.
3. Test both networks and plot metrics.
"""

import argparse


def run_seq_network():
    pass


def run_bimodal_network():
    pass


def main():
    parser = argparse.ArgumentParser(description='Train M-SEQ and M-SC')

    # Adding the required parser arguments (DATA)
    parser.add_argument('train_path', help='Filepath + prefix to the training data')
    parser.add_argument('val_path', help='Filepath + prefix to the validation data')
    parser.add_argument('test_path', help='Filepath + prefix to the test data')
    # Note: In future iterations, build this separation into python itself !
    parser.add_argument('noOfChrom', help='number of input chromatin datasets')

    # I'm going to change structure such that I have the models & metric out-files at the same place.
    # Leaving this be for now though!
    parser.add_argument("outfile", help="Filepath or prefix for storing the training metrics")
    parser.add_argument("basemodel", help="Base sequence model used to train this network")

    # Adding optional parser arguments

    args = parser.parse_args()

    train_path = args.datapath
    val_path = args.val_path
    test_path = args.test_path
    metrics = args.metrics_file
    basemodel = args.basemodel
    filelen = len(train_path + '.labels')
    chromsize = args.chromsize

    # Other Parameters
    batchsize = 400
    seqlen = 500
    convfilters = 240
    strides = 15
    pool_size = 15
    lstmnodes = 32
    dl1nodes = 1024
    dl2nodes = 512


if __name__ == "__main__":
    main()



