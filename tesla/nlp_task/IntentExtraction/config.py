# cofing:utf-8
# configuration for the Intent Extraction

import sys
from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent.parent.parent.parent
sys.path.append(str(MAIN_PATH))

from tesla.utils.functools import NoNewAttrs

BATCH_SIZE = 32
SAVE_STEPS = 10
LEARNING_RATE = 1e-4
LEARNING_RATE_LIMIT = 1e-4
TRAIN_STEPS = 100

SAVE_MODEL_PATH = './models'
PB_MODEL_PATH = './pb_models'

class Config(NoNewAttrs):
  vocab_size = 14352
  char_size = 74

  w_embedding_size = 320
  c_embedding_size = 16

  cell_type = 'gru'
  hidden_size = 320
  num_layers = 2
  forget_bias = True

  num_intents = 9
  num_tags = 54

  initialize_range = 0.01
  dropout = 0.2

model_config = Config()