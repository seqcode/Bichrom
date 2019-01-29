from __future__ import division
import argparse
import sys
import numpy as np
from sklearn.metrics import average_precision_score as auprc
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, concatenate, Input, LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping
import iterutils_mm as  mm
import iterutils as iu
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from subprocess import call


def merge_generators(filename, batchsize, seqlen):
    # need to pass these variables in order to pick mini-batches
    print 'loading the datasets..'
    A = np.loadtxt(filename + ".chrom")
    y = np.loadtxt(filename + ".labels")
    X = np.loadtxt(filename + ".seq", dtype='str')
    # C = np.loadtxt(filename + '.chromtracks')
    C = np.loadtxt(filename + '.chromtracks')
    print X.shape
    # from here, everything is in memory/need no I/O
    print 'done loading..'
    # pass all these datasets, as they are big, and it is a huge overhead to want to load them more than once.
    # the generator functionality is already here. I do not need another generator.
    while True:
        # creating the random vectors:
        perm = np.random.permutation(batchsize)
        idx_list = mm.create_random_indices(batchsize, y, A)
        # The supplementary idx_list is so we can create non-exact matched sets.
        # under the above knowledge, pick a mini-batch for all these three variables.
        # choose a permutation for every batch. Handle this together?
        sub_X = mm.create_batches(X, seqlen, str, idx_list, perm, y, A, 'seq')
        sub_y = mm.create_batches(y, seqlen, float, idx_list, perm, y, A, 'labels')
        yield sub_X, sub_y

def val_generator(filename, batchsize, seqlen):
    X = iu.train_generator(filename + ".seq", batchsize, seqlen, "seq", "repeat")
    y = iu.train_generator(filename + ".labels", batchsize, seqlen, "labels", "repeat")
    while True:
        yield X.next(), y.next()


class PR_metrics(Callback):

    def on_train_begin(self, logs=None):
        self.val_auprc = []
        self.train_auprc = []

    def on_epoch_end(self, epoch, logs=None):
        """ monitor PR """
        # validation data
        x_val, y_val = self.validation_data[0], self.validation_data[1]
        predictions = self.model.predict(x_val)
        aupr = auprc(y_val,predictions)
        self.val_auprc.append(aupr)


def keras_graphmodel(convfilters, strides, pool_size, lstmnodes, dl1nodes, dl2nodes, seqlen):
    """
    Build the NN architecture
    inputs: paramters for the architecture
    outputs: a build model architecture
    """
    seq_input = Input(shape=(seqlen, 4,), name='seq')
    xs = Conv1D(convfilters, 20, padding="same")(seq_input)
    xs = Activation('relu')(xs)
    xs = MaxPooling1D(padding="same", strides=strides, pool_size=pool_size)(xs)
    xs = LSTM(lstmnodes)(xs)
    # fully connected dense layers
    xs = Dense(dl1nodes, activation='relu')(xs)
    xs = Dropout(0.5)(xs)
    xs = Dense(dl2nodes, activation='sigmoid')(xs)
    result = Dense(1, activation='sigmoid')(xs)
    model = Model(inputs=seq_input, outputs=result)
    return model

def train(model, filename, batchsize, seqlen, filelen):
    """ 
    Train the NN using Adam 
    inputs: NN architecture
    outputs: trained model
    """
    adam = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=adam)
    # caculate the number of steps per epoch:
    steps_per_epoch = filelen/batchsize
    mg = merge_generators(filename, batchsize, seqlen)
    mgVal = val_generator(filename_val, 500000, seqlen)
    validation_data = mgVal.next()
    PR_history = PR_metrics()
    # adding checkpointing..
    filepath=sys.argv[2]
    checkpointer = ModelCheckpoint(filepath + '.{epoch:02d}.hdf5', verbose=1, save_best_only=False)
    # training the model..
    hist = model.fit_generator(epochs=1, steps_per_epoch=steps_per_epoch, generator = mg, validation_data=validation_data, callbacks=[PR_history, checkpointer])
    # consolidating the training metrics! (should probably be another function!)
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    val_pr = PR_history.val_auprc
    L = loss, val_loss, val_pr
    print L # safety net! 
    return model, L


def plot_network_outputs(L, metrics):
    loss, val_loss, val_pr = L
    np.savetxt(metrics + 'loss', loss, fmt='%1.2f')
    np.savetxt(metrics + 'val.loss', val_loss, fmt='%1.2f')
    np.savetxt(metrics + 'valpr', val_pr, fmt='%1.2f')
    # plot the losses
    # plt.figure()
    # plt.plot(loss) 
    # plt.plot(val_loss)
    # plt.savefig(metrics + '.loss.png')
    # plot the PRCs
    # plt.figure()
    # plt.plot(val_pr)
    # plt.savefig(metrics + '.pr.png')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Characterize the sequence and chromatin\
                                             predictors of induced TF binding")
    # adding the required parser arguments
    parser.add_argument("datapath", help="Filepath or prefix to the data files")
    parser.add_argument("val_datapath", help="Input data to be used for validation")
    parser.add_argument("outfile", help="Filepath or prefix for storing the metrics")
 
    # adding optional parser arguments
    parser.add_argument("--batchsize", help="batchsize used for training", default=512)
    parser.add_argument("--seqlen", help="input sequence length", default=500)
    parser.add_argument("--chromsize", help="input sequence length", default=12)

    args = parser.parse_args()
     
    filename = args.datapath
    filename_val = args.val_datapath
    metrics = args.metrics_file 
    filelen = len(filename + '.labels')
    
    # Optional Arguments:
    chromsize = args.chromsize
    seqlen = args.seqlen
    batchsize = args.batchsize

    # Other Default Parameters:
    convfilters = 240
    strides = 15
    pool_size = 15
    lstmnodes = 64
    dl1nodes = 1024
    dl2nodes = 512
    
    # Defining the network architecture
    # model = keras_graphmodel(convfilters, strides, pool_size, lstmnodes, dl1nodes, dl2nodes, seqlen)

    # Training the model    
    # trained_model, L = train(model, filename, batchsize, seqlen, filelen)
    # plotting the training losses/prcs
    # plot_network_outputs(L, metrics)
