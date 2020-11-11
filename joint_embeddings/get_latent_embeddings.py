import numpy as np
import tensorflow.keras.backend as K
from tensorflow.python.keras.backend import eager_learning_phase_scope


def keras_extract_fn(model):
    seq_input = model.get_layer('seq').input
    chrom_input = model.get_layer('chrom_input').input
    emb = model.layers[-1].input
    print(emb.shape)
    return K.function([seq_input, chrom_input], [emb])


def get_embeddings_low_mem(model, seq_input, chrom_input):
    f = keras_extract_fn(model)

    embedding_list_by_batch = []
    # iterate in batches for processing large datasets.
    for batch_start_idx in range(0, len(seq_input), 500):
        batch_end_idx = min(batch_start_idx + 500, len(seq_input))
        current_batch_seq = seq_input[batch_start_idx:batch_end_idx]
        current_batch_chrom = chrom_input[batch_start_idx:batch_end_idx]
        print(current_batch_chrom)
        with eager_learning_phase_scope(value=0):
            sn_activations = np.array(f([current_batch_seq,
                                         current_batch_chrom]))
        activations_rs = np.reshape(sn_activations, (sn_activations.shape[1], 2))
        activations_rs = activations_rs.astype(np.float64)
        embedding_list_by_batch.append(activations_rs)

    activations = np.vstack(embedding_list_by_batch)
    w, b = model.layers[-1].get_weights()
    w = np.reshape(w, (2,))
    weighted_embeddings = activations * w
    return weighted_embeddings