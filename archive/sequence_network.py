from __future__ import division
import argparse
import sys
import numpy as np
try:
    from sklearn.metrics import average_precision_score as auprc
    from keras.models import Model
    from keras.layers import Dense, Dropout, Flatten, Activation, concatenate, Input, LSTM
    from keras.layers import Conv1D, MaxPooling1D, BatchNormalization
    from keras.optimizers import SGD, Adam
    from keras.callbacks import EarlyStopping
    import trainNN.iterutils_mm as mm
    import trainNN.iterutils as iu
    from keras.callbacks import Callback
    from keras.callbacks import ModelCheckpoint
    from subprocess import call
except Exception as e:
    print e
    print "Please make sure the module is installed"
    print "Exiting."
    exit()


class Params():
    def __init__(self):
        self.batchsize = [500]  # [1024, 4096]
        self.dense_layers = range(1, 10)
        self.filters = [16, 32, 64, 128, 256]
        self.filter_size = [6, 15, 24, 32]
        self.pooling = [(2, 2), (4, 2), (6,2), (8,2), (10,2), (15,2),
                        (4, 4), (6, 4), (8, 4), (10, 4), (15, 4),
                        (6, 6), (8, 6), (10, 6), (15, 6),
                        (8, 8), (10, 8), (15, 8),
                        (10, 10), (15, 10), (15, 15)]
        self.dropout = [0.5, 0.5, 0.75]
        self.dense_layers_size = [128, 512, 1024, 2048]


def choose_params():
    # instantiate a class object
    params = Params()
    choice = []
    for values in [params.batchsize, params.dense_layers, params.filters, params.filter_size, params.pooling,
        params.dropout, params.dense_layers_size]:
        size = len(values)
        rnum = np.random.choice(size)
        choice.append(values[rnum])
    return choice


def merge_training_generators(filename, batchsize, seqlen):
    X = iu.train_generator(filename + ".seq", batchsize, seqlen, "seq", "repeat")
    y = iu.train_generator(filename + ".labels", batchsize, seqlen, "labels", "repeat")
    while True:
        yield X.next(), y.next()


def val_generator(filename_val, batchsize, seqlen):
    X = iu.train_generator(filename_val + ".seq", batchsize, seqlen, "seq", "repeat")
    y = iu.train_generator(filename_val + ".labels", batchsize, seqlen, "labels", "repeat")
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
        aupr = auprc(y_val, predictions)
        self.val_auprc.append(aupr)


def build_model(hyperparameters, seq_length):
    """ Define a Keras graph model with sequence as input """

    batch_size, n_layers, n_filters, filter_size, (pooling_size, pooling_strides), dropout, layer_size = hyperparameters
    print hyperparameters
    # Defining the model
    seq_input = Input(shape=(seq_length, 4,), name='seq')
    xs = Conv1D(filter_size, n_filters, activation='relu')(seq_input)
    xs = BatchNormalization()(xs)
    xs = MaxPooling1D(padding="same", strides=pooling_strides, pool_size=pooling_size)(xs)
    xs = Flatten()(xs)
    # Adding a specified number of dense layers
    for idx in range(n_layers):
        xs = Dense(layer_size, activation='relu')(xs)
        xs = Dropout(dropout)(xs)
    # Sigmoid output
    result = Dense(1, activation='sigmoid')(xs)
    model = Model(inputs=seq_input, outputs=result)
    return model


def train(model, filename, modelpath, batchsize, seqlen, steps_per_epoch):
    """ 
    Train the NN using Adam 
    inputs: NN architecture
    outputs: trained model
    """
    model.compile(loss='binary_crossentropy', optimizer='adam')
    # Calculate the number of steps per epoch:
    mg = merge_training_generators(filename, batchsize, seqlen)
    mgVal = val_generator(filename_val, 500000, seqlen)
    validation_data = mgVal.next()
    PR_history = PR_metrics()
    # Adding check-pointing
    checkpointer = ModelCheckpoint(filepath + '.{epoch:02d}.hdf5', verbose=1, save_best_only=False)
    # Parameters for early stopping
    earlystop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    # Training the model..
    hist = model.fit_generator(epochs=10, steps_per_epoch=steps_per_epoch, generator=mg,
                               validation_data=validation_data, callbacks=[PR_history, checkpointer, earlystop])
    # consolidating the training metrics! (should probably be another function!)
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    val_pr = PR_history.val_auprc
    L = loss, val_loss, val_pr
    return model, L


def save_network_outputs(L, metrics):
    loss, val_loss, val_pr = L
    np.savetxt(metrics + '.loss', loss, fmt='%1.2f')
    np.savetxt(metrics + '.val.loss', val_loss, fmt='%1.2f')
    np.savetxt(metrics + '.valpr', val_pr, fmt='%1.2f')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Characterize the sequence and chromatin\
                                             predictors of induced TF binding")
    # adding the required parser arguments
    parser.add_argument("datapath", help="/Path/to/prefix for the training data")
    parser.add_argument("val_datapath", help="/Path/to/prefix for the validation data")
    parser.add_argument("outfile", help="/Path/to/prefix for storing training metrics")
 
    # adding optional parser arguments
    parser.add_argument("--seqlen", help="input sequence length", default=500)
    parser.add_argument("--data_size", help="input training data size", default=1000)

    args = parser.parse_args()
     
    filename = args.datapath
    filename_val = args.val_datapath

    # Optional Arguments
    if args.seqlen:
        # using user-specified seq length
        seqlen = int(args.seqlen)
    else:
        seqlen = 500

    # Choosing the parameters from a hyper-parameter search space:
    params = choose_params()
    batchsize = params[0]  # The batch-size needs to be passed to the the train module.
    metrics = args.outfile
    for val in params:
        metrics = metrics + '.' + str(val)
    modelpath = metrics + '.model'

    # Defining the steps per epoch
    steps = args.data_size / batchsize

    # Defining the network architecture
    model = build_model(params, seq_length=seqlen)

    # Training the model    
    trained_model, L = train(model, filename, modelpath, batchsize, seqlen, steps)
    # Saving the training loss and auPRC.
    save_network_outputs(L, metrics)
