# cofing:utf-8
# configuration for the NER

import sys
from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent.parent.parent.parent
sys.path.append(str(MAIN_PATH))

from tesla.utils.functools import NoNewAttrs

SAVE_MODEL_PATH = './models'
PB_MODEL_PATH = './pb_models'
LEARNING_RATE = 1e-4
LEARNING_RATE_LIMIT = 1e-4
TRAIN_STEPS = 100

SAVE_STEPS = 10
BATCH_SZIE = 30

DATA_PATH = MAIN_PATH / 'datasets/CONLL2000/train.bin'

class Config(NoNewAttrs):
  enable_char_embedding=True
  char_size=28
  char_embedding_size=16
  padding_seq_length=50
  vocab_size=19124
  embedding_size=320
  window_size=[2, 3, 4]
  pool_size=[14, 13, 12]  # the padding char length is equal to 15, which is hard-coding in data_pipeline.py
  filter_number=2
  num_layers=4
  cell_type='gru'
  forget_bias=True
  hidden_size=320
  num_classes=46
  initialize_range=1e-2
  dropout=0.1

model_config = Config()