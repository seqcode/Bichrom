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

import tensorflow as tf

import iterutils


def TFdataset(path, batchsize, dataflag):

    TFdataset_batched = iterutils.train_TFRecord_dataset(path, batchsize, dataflag)

    return TFdataset_batched

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

    mirrored_strategy = tf.distribute.MirroredStrategy()
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
        """ monitor PR """
        predictions = np.array([])
        labels = np.array([])
        # TODO: How to simplify this part
        for x_vals, y_val in self.validation_data:
            ds = []
            for key, val in x_vals.items():
                val = tf.data.Dataset.from_tensors(val)
                options = tf.data.Options()
                options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE
                val = val.with_options(options)
                ds.append(val)
            ds_zip = tf.data.Dataset.zip((tuple(ds),))
            prediction = self.model.predict(ds_zip)
            predictions = np.concatenate([predictions, prediction.flatten()])
            labels = np.concatenate([labels, y_val])
        aupr = auprc(labels, predictions)
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


def transfer(train_path, val_path, basemodel, model,
             batchsize, records_path):
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
    train_dataset = TFdataset(train_path, batchsize, "all").prefetch(tf.data.AUTOTUNE)
    val_dataset = TFdataset(val_path, batchsize, "all").prefetch(tf.data.AUTOTUNE)
    precision_recall_history = PrecisionRecall(val_dataset)
    checkpointer = ModelCheckpoint(records_path + 'model_epoch{epoch}.hdf5',
                                   verbose=1, save_best_only=False)

    hist = model.fit_generator(train_dataset, epochs=15,
                               validation_data=val_dataset,
                               callbacks=[precision_recall_history,
                                          checkpointer])

    loss, val_pr = save_metrics(hist_object=hist, pr_history=precision_recall_history,
                                records_path=records_path)
    return loss, val_pr


def transfer_and_train_msc(train_path, val_path, base_model_path,
                           batch_size, records_path, bin_size, seq_len):

    # Calculate number of chromatin tracks
    no_of_chrom_tracks = len(train_path['chromatin_tracks'])
    model, basemodel = add_new_layers(base_model_path, seq_len, no_of_chrom_tracks, bin_size)
    loss, val_pr = transfer(train_path, val_path, basemodel, model,
                    batch_size, records_path)
    return loss, val_pr
