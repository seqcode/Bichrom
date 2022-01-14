import h5py
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import average_precision_score as auprc
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, concatenate, Input, LSTM
from tensorflow.keras.layers import Conv1D, Reshape, Lambda
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K

from tensorflow.distribute import MirroredStrategy

import iterutils


def data_generator(h5file, path, batchsize, seqlen, bin_size):
    if h5file:
        train_generator = iterutils.train_generator_h5
    else:
        train_generator = iterutils.train_generator
        
    dat_seq = train_generator(h5file, path['seq'], batchsize, seqlen, 'seq', 'repeat')
    dat_chromatin = []
    for chromatin_track in path['chromatin_tracks']:
        dat_chromatin.append(
            train_generator(h5file, chromatin_track, batchsize, seqlen, 'chrom', 'repeat'))
    y = train_generator(h5file, path['labels'], batchsize, seqlen, 'labels', 'repeat')
    while True:
        combined_chrom_data = []
        for chromatin_track_generators in dat_chromatin:
            curr_chromatin_mark = next(chromatin_track_generators)
            mark_resolution = curr_chromatin_mark.shape
            assert (mark_resolution == (batchsize, seqlen/bin_size)),\
                "Please check binning, specified bin size=50"
            combined_chrom_data.append(pd.DataFrame(curr_chromatin_mark))
        chromatin_features = pd.concat(combined_chrom_data, axis=1).values
        print(chromatin_features.shape)
        sequence_features = next(dat_seq)
        labels = next(y)
        yield [sequence_features, chromatin_features], labels


def add_new_layers(base_model_path, seq_len, no_of_chromatin_tracks, bin_size):
    """
    Takes a pre-existing M-SEQ (Definition in README) & adds structure to \
    use it as part of a bimodal DNA sequence + prior chromatin network
    Parameters:
        base_model (keras Model): A pre-trained sequence-only (M-SEQ) model
        chrom_size (int) : The expected number of chromatin tracks
    Returns:
        model: a Keras Model
    """

    def permute(x):
        return K.permute_dimensions(x, (0, 2, 1))

    mirrored_strategy = MirroredStrategy()
    with mirrored_strategy.scope():
        # load basemodel
        base_model = load_model(base_model_path)
        # Transfer from a pre-trained M-SEQ
        curr_layer = base_model.get_layer(name='dense_2')
        curr_tensor = curr_layer.output
        xs = Dense(1, name='MSEQ-dense-new', activation='tanh')(curr_tensor)

        # Defining a M-C sub-network
        chrom_input = Input(shape=(no_of_chromatin_tracks * int(seq_len/bin_size),), name='chrom_input')
        ci = Reshape((no_of_chromatin_tracks, int(seq_len/bin_size)),
                    input_shape=(no_of_chromatin_tracks * int(seq_len/bin_size),))(chrom_input)
        # Permuting the input dimensions to match Keras input requirements:
        permute_func = Lambda(permute)
        ci = permute_func(ci)
        xc = Conv1D(15, 1, padding='valid', activation='relu', name='MC-conv1d')(ci)
        xc = LSTM(5, activation='relu', name='MC-lstm')(xc)
        xc = Dense(1, activation='tanh', name='MC-dense')(xc)

        # Concatenating sequence (MSEQ) and chromatin (MC) networks:
        merged_layer = concatenate([xs, xc])
        result = Dense(1, activation='sigmoid', name='MSC-dense')(merged_layer)
        model = Model(inputs=[base_model.input, chrom_input], outputs=result)
    return model, base_model


class PrecisionRecall(Callback):

    def __init__(self, val_data):
        super().__init__()
        self.validation_data = val_data

    def on_train_begin(self, logs=None):
        self.val_auprc = []
        self.train_auprc = []

    def on_epoch_end(self, epoch, logs=None):
        (x_val, c_val), y_val = self.validation_data
        predictions = self.model.predict([x_val, c_val])
        aupr = auprc(y_val, predictions)
        self.val_auprc.append(aupr)


def save_metrics(hist_object, pr_history, records_path):
    loss = hist_object.history['loss']
    val_loss = hist_object.history['val_loss']
    val_pr = pr_history.val_auprc
    # Saving the training metrics
    np.savetxt(records_path + 'trainingLoss.txt', loss, fmt='%1.2f')
    np.savetxt(records_path + 'valLoss.txt', val_loss, fmt='%1.2f')
    np.savetxt(records_path + 'valPRC.txt', val_pr, fmt='%1.2f')
    return loss, val_pr


def transfer(h5file, train_path, val_path, basemodel, model, steps_per_epoch,
             batchsize, records_path, bin_size, seq_len):
    """
    Trains the M-SC, transferring weights from the pre-trained M-SEQ.
    The M-SEQ weights are kept fixed except for the final layer.

    Parameters:
        train_path (str): Path + prefix to training data
        val_path (str): Path + prefix to the validation data
        basemodel (Model): Pre-trained keras M-SEQ model
        model (Model): Defined bimodal network
        steps_per_epoch (int): Len(training_data/batchsize)
        batchsize (int): Batch size used in SGD
        records_path (str): Path + prefix to output directory

    Returns:
        loss (ndarray): An array with the validation loss at each epoch
    """

    # Making the base model layers non-trainable:
    for layer in basemodel.layers:
        layer.trainable = False
    # Training rest of the model.
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd)

    # Get train and validation data
    train_data_generator = data_generator(h5file, train_path, batchsize, seqlen=seq_len, bin_size=bin_size)
    val_data_generator = data_generator(h5file, val_path, 20000, seqlen=seq_len, bin_size=bin_size)
    validation_data = next(val_data_generator)
    precision_recall_history = PrecisionRecall(validation_data)
    checkpointer = ModelCheckpoint(records_path + 'model_epoch{epoch}.hdf5',
                                   verbose=1, save_best_only=False)

    hist = model.fit_generator(epochs=15, steps_per_epoch=steps_per_epoch,
                               generator=train_data_generator,
                               validation_data=validation_data,
                               callbacks=[precision_recall_history,
                                          checkpointer])

    loss, val_pr = save_metrics(hist_object=hist, pr_history=precision_recall_history,
                                records_path=records_path)
    return loss, val_pr


def transfer_and_train_msc(h5file, train_path, val_path, base_model_path,
                           batch_size, records_path, bin_size, seq_len):

     # Calculate size of training set
    if not h5file:
        training_set_size = len(np.loadtxt(train_path['labels']))
    else:
        with h5py.File(h5file, 'r', libver='latest', swmr=True) as temp:
            training_set_size = temp[train_path['labels']].shape[0]
    # Calculate the steps per epoch
    steps_per_epoch = training_set_size / batch_size
    # Calculate number of chromatin tracks
    no_of_chrom_tracks = len(train_path['chromatin_tracks'])
    model, basemodel = add_new_layers(base_model_path, seq_len, no_of_chrom_tracks, bin_size)
    loss, val_pr = transfer(h5file, train_path, val_path, basemodel, model, steps_per_epoch,
                    batch_size, records_path, bin_size, seq_len)
    return loss, val_pr
