# coding:utf-8
# Produced by Jiaxin Zhang
# Start Data: 4_Aug_2020
# TensorFlow Version for CNN-LSTM-CRF.
#
# For GOD I Trust.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import sys
import copy
import tensorflow as tf
from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent.parent.parent
sys.path.append(str(MAIN_PATH))
from tesla.models import BaseModel
from tesla.models.model_lego import embedding_lookup

class TaggerModel(BaseModel):
  def __init__(self,
               config,
               is_training,
               input_x,
               input_length,
               input_char=None):
    """Constructor for CNN-LSTM-CRF Model.
    
    Args:
      config: configuration.
      is_training: Boolean, control whether train or not.
      input_x: tf.int32 Tensor with shape [batch_size, seq_length].
      input_length: tf.int32 Tensor with shape [batch_size].
    """
    config = copy.deepcopy(config)
    self.is_training = is_training

    self.enable_char_embedding = config.enable_char_embedding
    if self.enable_char_embedding:
      assert input_char is not None
      self.char_size = config.char_size
      self.char_embedding_size = config.char_embedding_size

    self.vocab_size = config.vocab_size
    self.embedding_size = config.embedding_size
    self.window_size = config.window_size
    self.pool_size = config.pool_size
    self.filter_number = config.filter_number
    
    self.hidden_size = config.hidden_size
    self.num_classes = config.num_classes
    
    self.initialize_range = config.initialize_range
    self.dropout = config.dropout if self.is_training else 0.0
  
  def buildGraph(self, input_x, input_length, input_char=None):
    with tf.variable_scope('cnn_lstm_crf'):
      with tf.variable_scope('vocab_embedding'):
        embedding_output, self.embedding_table = embedding_lookup(
          input_ids=input_x,
          vocab_size=self.vocab_size,
          embedding_size=self.embedding_size,
          initializer_range=self.initialize_range)


if __name__ == '_main__':
  pass