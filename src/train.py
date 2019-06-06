from keras.callbacks import ModelCheckpoint
from model import load_model
from dataset import get_javascript_dataset
from constants import MODELS_DIR, FINAL_MODEL_PATH, TIMESTEPS, BATCH_SIZE


def train(model, X, y):
    callbacks = [
        ModelCheckpoint(MODELS_DIR + 'jscode-weights-{epoch:02d}-{loss:.3f}.h5',
                        verbose=1,
                        save_weights_only=True)
    ]
    divisible_len = len(X) - len(X) % BATCH_SIZE
    model.fit(X[:divisible_len], y[:divisible_len],
              batch_size=BATCH_SIZE,
              epochs=3,
              callbacks=callbacks)


if __name__ == '__main__':
    X, y, n_vocab, chars = get_javascript_dataset()
    model = load_model(BATCH_SIZE, len(chars), TIMESTEPS, training=True)
    train(model, X, y)
    prod_model = load_model(BATCH_SIZE, len(chars), TIMESTEPS, training=False)
    prod_model.set_weights(model.get_weights())
    prod_model.save(FINAL_MODEL_PATH)
