from keras.models import Sequential
from keras.layers import Dense, Dropout, CuDNNLSTM, Embedding


# Hyperparams
HIDDEN_SIZE = 256  # LSTM/GRU units number of units
DROPOUT     = 0.25  # Dropout layers prob


def load_model(batch_size, n_chars, sequence_len, training=False):
    '''Create a multi-layer LSTM based neural network.

    Args:
        batch_size (int): Batch size should be specified correcly during training
                          because LSTMs are set to stateful
        n_chars (int): Number of unique charactes in training text corpus
        sequence_len (int): Length of sequences - number of chars preceding target chars
        training (bool): Training mode specifier. Need to adjust statefulness of LSTMs
    '''

    # Specify batch_input_shape during training because of stateful LSTM units
    batch_input_shape = (batch_size, sequence_len) if training else None

    model = Sequential()
    model.add(Embedding(n_chars, HIDDEN_SIZE,
                        input_length=sequence_len,
                        batch_input_shape=batch_input_shape))
    model.add(CuDNNLSTM(HIDDEN_SIZE, return_sequences=True, stateful=training))
    model.add(Dropout(DROPOUT))
    model.add(CuDNNLSTM(HIDDEN_SIZE, return_sequences=True, stateful=training))
    model.add(Dropout(DROPOUT))
    model.add(CuDNNLSTM(HIDDEN_SIZE, stateful=training))
    model.add(Dense(n_chars, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='nadam')

    return model
