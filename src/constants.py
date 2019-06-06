from utils import relative_path


TIMESTEPS          = 100  # Number of timesteps feeded to model
BATCH_SIZE         = 128
TEXT_CORPUS_PATH   = relative_path('../data/javascript_codes.js')
MODELS_DIR         = relative_path('../model/')  # Model save dir
FINAL_MODEL_PATH   = relative_path('../model/jscode-final-model.h5')  # Last learned model - arch+weights
SAMPLE_ITERS       = 2000  # Number of chars generated when sampling from model
