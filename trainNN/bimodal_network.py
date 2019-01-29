from __future__ import division
import sys
import numpy as np
import sklearn.metrics
import sklearn.model_selection as ms
from sklearn.metrics import average_precision_score as auprc
# user defined module
import iterutils as iu
# keras imports
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.models import Model,load_model
from keras.layers import Dense, Dropout, Activation, Flatten, concatenate, Input, LSTM, Bidirectional
from keras.layers import Conv1D, MaxPooling1D, Reshape, Lambda
from keras.optimizers import SGD, Adam
import keras.backend as K

def merge_generators_seq(filename, batchsize, seqlen):
    X = iu.train_generator(filename + ".seq", batchsize, seqlen, "seq", "repeat")
    # merge the marks here, and the split them up in training.
    C = iu.train_generator(filename + ".chromtracks", batchsize, seqlen, "chrom", "repeat")
    y = iu.train_generator(filename + ".labels", batchsize, seqlen, "labels", "repeat")
    while True:
        print X.next,
        yield [X.next()], y.next()


def merge_generators(filename, batchsize, seqlen):
    X = iu.train_generator(filename + ".seq", batchsize, seqlen, "seq", "repeat")
    # merge the marks here, and the split them up in training.
    C = iu.train_generator(filename + ".chromtracks", batchsize, seqlen, "chrom", "repeat")
    y = iu.train_generator(filename + ".labels", batchsize, seqlen, "labels", "repeat")
    while True:
        print X.next,
        yield [X.next(), C.next()], y.next()


def add_new_layers(basemodel, chromsize):
    curr_layer = basemodel.get_layer(name='dense_2') # CHANGE THIS HERE!
    curr_tensor = curr_layer.output
    print curr_tensor.shape
    # So x is the 'dense2' layer, has weights mapping from the previous 4 nodes to new 2 nodes.
    xs = Dense(1, name='dense_3_new', activation='tanh')(curr_tensor)

    # creating a chromatin structure.
    print "chromatin net..."
    chrom_input = Input(shape=(chromsize * 10,), name='chrom_input')
    print chrom_input.shape
    ci = Reshape((chromsize, 10), input_shape=(chromsize * 10,))(chrom_input)
    print ci.shape
    def permute(x):
        return K.permute_dimensions(x, (0,2,1))
    LP = Lambda(permute)
    ci = LP(ci)
    print "Shape after transpose"
    print ci.shape
    # padding = same makes not as much sense here as you'd think. you only have 10 values. You're effectively zero padding to the same size. gah!
    xc = Conv1D(15, 1, padding="valid", activation='relu', name="conv1d_chrom1")(ci)
    # xc = Flatten()(xc)
    xc = LSTM(5, activation='relu', name='lstm_chrom')(xc)
    xc = Dense(1, activation='tanh', name="dense_chrom")(xc)
    # merging sequence and chromatin networks
    merge = concatenate([xs, xc])
    print merge.shape
    result = Dense(1, activation='sigmoid', name="Dense_merged")(merge)
    model = Model(inputs=[basemodel.input, chrom_input], outputs=result)
    return model
    # great! so this is the transferred model. Yups!! And it works yo. :D 

class PR_metrics(Callback):

    def on_train_begin(self, logs=None):
        self.val_auprc = []
        self.train_auprc = []

    def on_epoch_end(self, epoch, logs=None):
        """ monitor PR """
        # validation data
        x_val, c_val, y_val = self.validation_data[0], self.validation_data[1], self.validation_data[2]
        print x_val.shape
        print c_val.shape
        print y_val.shape
        predictions = self.model.predict([x_val, c_val])
        aupr = auprc(y_val,predictions)
        self.val_auprc.append(aupr)


def transfer(filename, filename_val, basemodel, model, filelen, batchsize):
    # making the base model layers non-trainable:
    for layer in basemodel.layers:
        layer.trainable = False
    # training rest of the model.
    # optimizer 
    # adam = Adam(lr=0.001)
    # model.compile(loss='binary_crossentropy', optimizer=adam)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd)
    # caculate the number of steps per epoch
    steps_per_epoch = filelen/batchsize
    # data generator (train)
    mg = merge_generators(filename, batchsize, seqlen)
    # data validation (validation)
    mgVal = merge_generators(filename_val, 200000, seqlen)
    validation_data = mgVal.next()
    PR_history = PR_metrics()

    # adding checkpointing..
    filepath=sys.argv[2]
    checkpointer = ModelCheckpoint(filepath + '.{epoch:02d}.hdf5', verbose=1, save_best_only=False)
    # training the model..
    hist = model.fit_generator(epochs=2, steps_per_epoch=steps_per_epoch, generator = mg, validation_data=validation_data, callbacks=[PR_history, checkpointer])
    # consolidating the training metrics! (should probably be another function!)
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    val_pr = PR_history.val_auprc
    # L = list with all the losses
    L = loss, val_loss, val_pr
    print L # safety net! 
    return model, L


def plot_network_outputs(L, metrics):
    loss, val_loss, val_pr = L
    np.savetxt(metrics + '.loss', loss, fmt='%1.2f')
    np.savetxt(metrics + '.val.loss', val_loss, fmt='%1.2f')
    np.savetxt(metrics + '.valpr', val_pr, fmt='%1.2f')
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
    
    parser = argparse.ArgumentParser(description="Train a bimodal sequence and chromatin network")
    # adding the required parser arguments
    parser.add_argument("datapath", help="Filepath or prefix to the data files")
    parser.add_argument("val_datapath", help="Input data to be used for validation")
    parser.add_argument("outfile", help="Filepath or prefix for storing the metrics")
    parser.add_argument("basemodel", help="Base sequence model used to train this network")

    # adding optional parser arguments
    parser.add_argument("--batchsize", help="batchsize used for training", default=512)
    parser.add_argument("--seqlen", help="input sequence length", default=500)
    parser.add_argument("--chromsize", help="input sequence length", default=12)

    args = parser.parse_args()

    filename = args.datapath
    filename_val = args.val_datapath
    metrics = args.metrics_file
    basemodel = args.basemodel
    filelen = len(filename + '.labels')

    # Optional Arguments:
    chromsize = args.chromsize
    seqlen = args.seqlen
    batchsize = args.batchsize
 
    # Other Parameters
    batchsize = 400
    seqlen = 500
    convfilters = 240
    strides = 15
    pool_size = 15
    lstmnodes = 32
    dl1nodes = 1024
    dl2nodes = 512
    
    # model = add_new_layers(basemodel, chromsize=chromsize)
    # trained_model, L = transfer(filename, filename_val, basemodel, model, filelen, batchsize)
    # plot_network_outputs(L, metrics)
