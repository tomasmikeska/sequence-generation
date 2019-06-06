import numpy as np
from random import randint
from keras.models import load_model
from dataset import get_javascript_dataset
from constants import TEXT_CORPUS_PATH, TIMESTEPS, FINAL_MODEL_PATH, SAMPLE_ITERS


def sample(model, seeding_text, iterations):
    X, y, n_vocab, chars = get_javascript_dataset()
    char_to_int = { char: index for index, char in enumerate(chars) }
    int_to_char = { index: char for index, char in enumerate(chars) }

    result = ''
    pattern = [char_to_int[char] for char in seeding_text]

    for i in range(iterations):
        x = np.reshape(pattern, (1, len(pattern)))
        prediction = model.predict(x, verbose=0)[0]
        index = np.random.choice(len(prediction), 1, p=prediction)[0]
        result += int_to_char[index]
        pattern = pattern[1:] + [index]

    return result


if __name__ == '__main__':
    model = load_model(FINAL_MODEL_PATH)

    with open(TEXT_CORPUS_PATH) as f:
        corpus = f.read()
        random_start = randint(TIMESTEPS, len(corpus)-TIMESTEPS)
        seeding_text = corpus[random_start:random_start+TIMESTEPS]

    result = sample(model, seeding_text, SAMPLE_ITERS)

    print(result)
