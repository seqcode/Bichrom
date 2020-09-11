import numpy as np

from sklearn.metrics import average_precision_score as auprc
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# local imports
from iterutils import train_generator


def data_generator(path, batchsize, seqlen):
    X = train_generator(path['seq'], batchsize, seqlen, 'seq', 'repeat')
    y = train_generator(path['labels'], batchsize, seqlen, 'labels', 'repeat')
    while True:
        yield next(X), next(y)


class PrecisionRecall(Callback):

    def __init__(self, val_data):
        super().__init__()
        self.validation_data = val_data

    def on_train_begin(self, logs=None):
        self.val_auprc = []
        self.train_auprc = []

    def on_epoch_end(self, epoch, logs=None):
        """ monitor PR """
        x_val, y_val = self.validation_data
        predictions = self.model.predict(x_val)
        aupr = auprc(y_val, predictions)
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
def train(model, train_path, val_path, steps_per_epoch, batch_size,
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
    adam = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=adam)
    train_generator = data_generator(train_path, batch_size, seqlen=500)
    val_generator = data_generator(val_path, 200000, seqlen=500)

    validation_data = next(val_generator)
    precision_recall_history = PrecisionRecall(validation_data)
    # adding check-pointing
    checkpointer = ModelCheckpoint(records_path + 'model_epoch{epoch}.hdf5',
                                   verbose=1, save_best_only=False)
    # defining parameters for early stopping
    # earlystop = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
    #                           patience=5)
    # training the model..
    hist = model.fit_generator(epochs=1, steps_per_epoch=steps_per_epoch,
                               generator=train_generator,
                               validation_data=validation_data,
                               callbacks=[precision_recall_history,
                                          checkpointer])

    loss, val_pr = save_metrics(hist, precision_recall_history,
                                records_path=records_path)
    return loss, val_pr


def build_and_train_net(hyperparams, train_path, val_path, batch_size,
                        records_path):

    # Calculate size of training set
    training_set_size = len(np.loadtxt(train_path['labels']))
    # Calculate the steps per epoch
    steps = training_set_size/batch_size
    model = build_model(params=hyperparams, seq_length=500)

    loss, val_pr = train(model, train_path=train_path, val_path=val_path,
                         steps_per_epoch=steps, batch_size=batch_size,
                         records_path=records_path)
    return loss, val_pr
