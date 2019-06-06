from keras.models import Sequential
from keras.layers import Dense, Dropout, CuDNNLSTM, Embedding


# Hyperparams
HIDDEN_SIZE = 256
DROPOUT     = 0.25


def load_model(batch_size, n_chars, sequence_len, training=False):
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
