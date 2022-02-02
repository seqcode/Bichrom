import h5py
import numpy as np

from sklearn.metrics import average_precision_score as auprc
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

import tensorflow as tf

# local imports
import iterutils

def TFdataset(path, batchsize, dataflag):

    TFdataset_batched = iterutils.train_TFRecord_dataset(path, batchsize, dataflag)

    return TFdataset_batched

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
        for x_val, y_val in self.validation_data:
            x_val = tf.data.Dataset.from_tensors(x_val)
            options = tf.data.Options()
            options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE
            x_val = x_val.with_options(options)
            prediction = self.model.predict(x_val)
            predictions = np.concatenate([predictions, prediction.flatten()])
            labels = np.concatenate([labels, y_val])
        aupr = auprc(labels, predictions)
        self.val_auprc.append(aupr)


def build_model(params, seq_length):
    """
    Define a Keras graph model with DNA sequence as input.
    Parameters:
        params (class): A class with a set of hyper-parameters
        seq_length (int): Length of input sequences
    Returns
        model (keras model): A keras model
    """
    # MirroredStrategy to employ all available GPUs
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        seq_input = Input(shape=(seq_length, 4,), name='seq')
        xs = Conv1D(filters=params.n_filters,
                    kernel_size=params.filter_size,
                    activation='relu')(seq_input)
        xs = BatchNormalization()(xs)
        xs = MaxPooling1D(padding="same", strides=params.pooling_stride,
                        pool_size=params.pooling_size)(xs)
        xs = LSTM(32)(xs)
        # adding a specified number of dense layers
        for idx in range(params.dense_layers):
            xs = Dense(params.dense_layer_size, activation='relu')(xs)
            xs = Dropout(params.dropout)(xs)
        result = Dense(1, activation='sigmoid')(xs)
        model = Model(inputs=seq_input, outputs=result)
    return model


def save_metrics(hist_object, pr_history, records_path):
    loss = hist_object.history['loss']
    val_loss = hist_object.history['val_loss']
    val_pr = pr_history.val_auprc

    # Saving the training metrics
    np.savetxt(records_path + 'trainingLoss.txt', loss, fmt='%1.2f')
    np.savetxt(records_path + 'valLoss.txt', val_loss, fmt='%1.2f')
    np.savetxt(records_path + 'valPRC.txt', val_pr, fmt='%1.2f')
    return loss, val_pr


# NOTE: ADDING A RECORDS PATH HERE!
def train(model, train_path, val_path, batch_size,
          records_path):
    """
    Train the Keras graph model
    Parameters:
        model (keras Model): The Model defined in build_model
        train_path (str): Path to training data
        val_path (str): Path to validation data
        steps_per_epoch (int): Len(training_data)/batch_size
        batch_size (int): Size of mini-batches used during training
        records_path (str): Path + prefix to output directory
    Returns:
        loss (ndarray): An array with the validation loss at each epoch
    """
    
    adam = Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=adam)
    train_dataset = TFdataset(train_path, batch_size, "seqonly").prefetch(tf.data.AUTOTUNE)
    val_dataset = TFdataset(val_path, batch_size, "seqonly").prefetch(tf.data.AUTOTUNE)

    precision_recall_history = PrecisionRecall(val_dataset)
    # adding check-pointing
    checkpointer = ModelCheckpoint(records_path + 'model_epoch{epoch}.hdf5',
                                   verbose=1, save_best_only=False)
    # defining parameters for early stopping
    # earlystop = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
    #                           patience=5)
    # training the model..
    hist = model.fit(train_dataset, epochs=15,
                        validation_data=val_dataset,
                        callbacks=[precision_recall_history,
                                    checkpointer])

    loss, val_pr = save_metrics(hist, precision_recall_history,
                                records_path=records_path)
    return loss, val_pr


def build_and_train_net(hyperparams, train_path, val_path, batch_size,
                        records_path, seq_len):

    model = build_model(params=hyperparams, seq_length=seq_len)

    loss, val_pr = train(model, train_path=train_path, val_path=val_path,
                        batch_size=batch_size, records_path=records_path)

    return loss, val_pr
