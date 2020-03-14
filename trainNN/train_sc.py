from __future__ import division
import numpy as np

from sklearn.metrics import average_precision_score as auprc
from keras.models import Model
from keras.layers import Dense, concatenate, Input, LSTM
from keras.layers import Conv1D, Reshape, Lambda
from keras.optimizers import SGD
import keras.backend as K
import iterutils as iu
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint


def data_generator(filename, batchsize, seqlen):
    X = iu.train_generator(filename + '.seq', batchsize,
                           seqlen, 'seq', 'repeat')
    C = iu.train_generator(filename + '.chromtracks', batchsize,
                           seqlen, 'chrom', 'repeat')
    y = iu.train_generator(filename + '.labels', batchsize,
                           seqlen, 'labels', 'repeat')
    while True:
        yield [X.next(), C.next()], y.next()


def add_new_layers(base_model, chrom_size):
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

    # Transfer from a pre-trained M-SEQ
    curr_layer = base_model.get_layer(name='dense_2')
    curr_tensor = curr_layer.output
    xs = Dense(1, name='MSEQ-dense-new', activation='tanh')(curr_tensor)

    # Defining a M-C sub-network
    chrom_input = Input(shape=(chrom_size * 10,), name='chrom_input')
    ci = Reshape((chrom_size, 10), input_shape=(chrom_size * 10,))(chrom_input)
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
    return model


class PrecisionRecall(Callback):
    def on_train_begin(self, logs=None):
        self.val_auprc = []
        self.train_auprc = []

    def on_epoch_end(self, epoch, logs=None):
        x_val, c_val, y_val = self.validation_data[0], self.validation_data[1],\
                              self.validation_data[2]
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
    return loss


def transfer(train_path, val_path, basemodel, model, steps_per_epoch,
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
    train_data_generator = data_generator(train_path, batchsize, seqlen=500)
    val_data_generator = data_generator(val_path, 200000, seqlen=500)
    validation_data = val_data_generator.next()
    precision_recall_history = PrecisionRecall()
    checkpointer = ModelCheckpoint(records_path + 'model_epoch{epoch}.hdf5',
                                   verbose=1, save_best_only=False)

    hist = model.fit_generator(epochs=2, steps_per_epoch=steps_per_epoch,
                               generator=train_data_generator,
                               validation_data=validation_data,
                               callbacks=[precision_recall_history,
                                          checkpointer])

    loss = save_metrics(hist_object=hist, pr_history=precision_recall_history,
                        records_path=records_path)
    return loss


def transfer_and_train_msc(train_path, val_path, no_of_chrom_tracks, basemodel,
                           batch_size, records_path):

    # Calculate size of the training set:
    training_set_size = len(np.loadtxt(train_path + '.labels'))
    # Calculate the steps per epoch
    steps_per_epoch = training_set_size / batch_size

    model = add_new_layers(basemodel, chrom_size=no_of_chrom_tracks)
    loss = transfer(train_path, val_path, basemodel, model, steps_per_epoch,
                    batch_size, records_path)
    return loss
