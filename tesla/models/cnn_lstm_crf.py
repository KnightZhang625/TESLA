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
import numpy as np
import tensorflow as tf
from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent.parent.parent
sys.path.append(str(MAIN_PATH))
from tesla.models import BaseModel
from tesla.models.model_lego import embedding_lookup, textCNN, createMultiRNNCells, crfEncode, crfDecode
from tesla.models.utils import create_initializer

class TaggerModel(BaseModel):
  def __init__(self,
               config,
               is_training,
               input_x,
               input_length,
               input_char=None,
               golden_labels=None):
    """Constructor for CNN-LSTM-CRF Model.
    
    Args:
      config: configuration.
      is_training: Boolean, control whether train or not.
      input_x: tf.int32 Tensor with shape [batch_size, seq_length].
      input_length: tf.int32 Tensor with shape [batch_size].
      input_char: tf.int32 Tensor with shape [batch, seq_length, each_vocab_size].
    """
    config = copy.deepcopy(config)
    self.is_training = is_training

    self.enable_char_embedding = config.enable_char_embedding
    if self.enable_char_embedding:
      assert input_char is not None
      self.char_size = config.char_size
      self.char_embedding_size = config.char_embedding_size
      self.padding_seq_length = config.padding_seq_length

    self.vocab_size = config.vocab_size
    self.embedding_size = config.embedding_size
    self.window_size = config.window_size
    self.pool_size = config.pool_size
    self.filter_number = config.filter_number
    self.num_layers = config.num_layers
    self.cell_type = config.cell_type
    self.forget_bias = config.forget_bias

    self.hidden_size = config.hidden_size
    self.num_classes = config.num_classes
    
    self.initialize_range = config.initialize_range
    self.dropout = config.dropout if self.is_training else 0.0

    self.results = {}
    self.buildGraph(input_x, input_length, input_char, golden_labels)
  
  def buildGraph(self, 
                 input_x, 
                 input_length,
                 input_char,
                 golden_labels):
    with tf.variable_scope('cnn_lstm_crf'):
      # word embedding
      with tf.variable_scope('vocab_embedding'):
        embedding_output, self.embedding_table = embedding_lookup(
          input_ids=input_x,
          vocab_size=self.vocab_size,
          embedding_size=self.embedding_size,
          initializer_range=self.initialize_range)
      
      # character embedding
      if self.enable_char_embedding:
        with tf.variable_scope('char_embedding', reuse=tf.AUTO_REUSE):
          # char_embedding_output -> [b, s, v_s, e]
          char_embedding_output, self.char_embedding_table = embedding_lookup(
            input_ids=input_char,
            vocab_size=self.char_size,
            embedding_size=self.char_embedding_size,
            initializer_range=self.initialize_range,
            word_embedding_name='char_embedding')

          char_cnn_embeddings = []
          for i in range(self.padding_seq_length):
            char_embedding = textCNN(
              embedding=char_embedding_output[:, i, :, :],
              window_size=self.window_size,
              filter_number=self.filter_number,
              pool_size=self.pool_size,
              dropout_prob=self.dropout)
            char_cnn_embeddings.append(tf.expand_dims(char_embedding, 1))    
          char_cnn_embeddings = tf.concat(char_cnn_embeddings, 1)

        embedding_output = tf.concat((embedding_output, char_cnn_embeddings), -1)
        # this step is crucial, where the dynamic_rnn needs the exact shape
        embedding_output = tf.reshape(embedding_output, (-1, self.padding_seq_length, self.embedding_size + self.filter_number * 3))

      with tf.variable_scope('rnn'):
        assert_op = tf.assert_equal(self.num_layers % 2, 0)
        with tf.control_dependencies([assert_op]):
          num_bi_layers = int(self.num_layers / 2)
          num_bi_residual_layers =  num_bi_layers - 1

          fw_cell = createMultiRNNCells(cell_type=self.cell_type,
                                        hidden_size=self.hidden_size,
                                        num_layers=num_bi_layers,
                                        dropout_prob=self.dropout,
                                        num_residual_layers=num_bi_residual_layers,
                                        forget_bias=self.forget_bias)
          
          bw_cell = createMultiRNNCells(cell_type=self.cell_type,
                                        hidden_size=self.hidden_size,
                                        num_layers=num_bi_layers,
                                        dropout_prob=self.dropout,
                                        num_residual_layers=num_bi_residual_layers,
                                        forget_bias=self.forget_bias)

          # bi_output: [output_fw, output_bw]
          # bi_states: [state_fw * num_bi_layers, state_bw * num_bi_layers]
          bi_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            fw_cell, bw_cell, embedding_output, dtype=tf.float32,
            sequence_length=input_length)
          encoder_outputs = tf.concat(bi_outputs, -1)
      
      with tf.variable_scope('linear_transformer'):
        outputs = tf.layers.dense(
          encoder_outputs,
          self.num_classes,
          activation=tf.nn.relu,
          name='linear_transformer',
          kernel_initializer=create_initializer())
      self.results['outputs'] = outputs
      
      with tf.variable_scope('crf', reuse=tf.AUTO_REUSE):
        self.transition_params = tf.Variable(tf.truncated_normal([self.num_classes, self.num_classes], stddev=0.1), name='transition_params')
        self.results['transition_params'] = self.transition_params
        if self.is_training:
          self.log_likelihood, _ = crfEncode(
            logits=outputs,
            labels=golden_labels,
            sequence_lengths=input_length,
            transition_params=self.transition_params)
        
          self.results['log_likelihood'] = self.log_likelihood

  def decode(self, logit, sequence_lengths):
    """Viterbi Decode.
    
    Args:
      logit: tf.float32 Tensor with shape [seq_length, num_classes].
    
    Return:
      viterbi_sequence: a list of predicted indices.
		  viterbi_score: the log-likelihood score.
    """
    return crfDecode(logit, self.transition_params, sequence_lengths)
  
  def getResults(self, name):
    """Return the results.
    
    Args:
      name: name for the result, choose from ['log_likelihood', 'transition_params'].
    """
    if name in self.results:
      return self.results[name]
    print('Cannot find `{}` in results.'.format(name))
    raise ValueError

if __name__ == '__main__':
  ### Example for the Model ###
  import collections
  
  Config = collections.namedtuple('Config',
    'enable_char_embedding char_size char_embedding_size padding_seq_length \
      vocab_size embedding_size window_size pool_size filter_number num_layers \
      cell_type forget_bias hidden_size num_classes initialize_range dropout')
  config = Config(enable_char_embedding=True,
                  char_size=26,
                  char_embedding_size=30,
                  padding_seq_length=10,
                  vocab_size=120,
                  embedding_size=60,
                  window_size=[2, 3, 4],
                  pool_size=[4, 3, 2],
                  filter_number=2,
                  num_layers=4,
                  cell_type='gru',
                  forget_bias=True,
                  hidden_size=16,
                  num_classes=8,
                  initialize_range=1e-2,
                  dropout=0.1)
  
  input_x = tf.ones((5, 10), dtype=tf.int32)
  golden_labels = tf.ones((5, 10), dtype=tf.int32)
  input_char = tf.ones((5, 10, 5), dtype=tf.int32)
  input_length = tf.constant([5, 7, 8, 10, 10], dtype=tf.int32)
  toy_model = TaggerModel(config,
                          True,
                          input_x=input_x,
                          golden_labels=golden_labels,
                          input_length=input_length,
                          input_char=input_char)

  print(toy_model.log_likelihood)
  print(toy_model.transition_params)