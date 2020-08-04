# coding:utf-8
# Produced by Jiaxin Zhang
# Start Data: 4_Aug_2020
# TensorFlow Version for CRF Module.
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
import logging
import tensorflow as tf
# tf.enable_eager_execution()

from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent.parent.parent
sys.path.append(str(MAIN_PATH))

from tesla.models.utils import assert_rank, get_shape_list, create_initializer

def crfEncode(logits, labels, sequence_lengths):
  """CRF forward step, Please specify the scope outside.
  
  Args:
    logits: tf.float32 Tensor with shape [batch_size, seq_length, num_tags].
    labels: tf.int32 Tensor with shape [batch_size, seq_length].
    sequence_lengths: tf.int32 Tensor with shape [batch_size],

  Returns:
    log_likelihood: tf.float32 Tensor with shape [batch_size].
    transition_params: tf.float32 Tensor with shape [num_tags, num_tags].
  """
  assert_rank(logits, 3)
  assert_rank(labels, 2)
  assert_rank(sequence_lengths, 1)
  
  return tf.contrib.crf.crf_log_likelihood(
    logits, labels, sequence_lengths)

def crfDecode(logit, transition_params):
  """CRF Decode step. Only support single prediction.
  
  Args:
    logit: tf.float32 Tensor with shape [seq_length, num_tags].
    transition_params: tf.float32 Tensor with shape [num_tags, num_tags].
  
  Return:
    viterbi_sequence: a list of predicted indices.
    viterbi_score: the log-likelihood score.
  """
  assert_rank(logit, 2)
  assert_rank(transition_params, 2)
  return tf.contrib.crf.viterbi_decode(
    logit, transition_params)

def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size,
                     initializer_range,
                     word_embedding_name='word_embeddings',
                     use_one_hot_embeddings=False):
  """Looks up words embeddings for id tensor.

  Args:
    input_ids: int32 Tensor of shape [batch_size, seq_length] containing word ids.
    vocab_size: int. Size of the embedding vocabulary.
    embedding_size: int. Width of the word embeddings.
    initializer_range: float. Embedding initialation range.
    word_embedding_name: string. Name of the embedding table.
    use_one_hot_embeddings: bool. If True. use one-hot method for word embedding.
      If False, use 'tf.gather()'.
  
  Returns:
    float Tensor of shape [batch_size, seq_length, embedding_size].
  """
  
  embedding_table = tf.get_variable(
    name=word_embedding_name,
    shape=[vocab_size, embedding_size],
    initializer= create_initializer(initializer_range=initializer_range))
  
  if use_one_hot_embeddings:
    input_shape = get_shape_list(input_ids, expected_rank=2)
    input_ids_squeeze = tf.reshape(input_ids, [-1])
    one_hot_input_ids = tf.one_hot(input_ids_squeeze, depth=vocab_size)
    output = tf.matmul(one_hot_input_ids, embedding_table)
    output = tf.reshape(output, [input_shape[0], input_shape[1], -1])
  else:
    output = tf.nn.embedding_lookup(embedding_table, input_ids)
  
  return output, embedding_table

def textCNN(embedding,
            window_size,
            filter_number,
            pool_size,
            dropout_prob):
  """Apply textCNN on the embeddings.
    The code here is revised from the below url:
      https://github.com/dennybritz/cnn-text-classification-tf/blob/master/text_cnn.py
    Double Salute !

    ATTENTION: If use textCNN, please padding the whole data to the same length.
  """
  input_shape = get_shape_list(embedding)
  batch_size = input_shape[0]
  embedding_size = input_shape[2]
  # expand the embedding from [batch_size, seq_length, embedding_size]
  # to [batch_size, seq_length, embedding_size, 1]
  embedding_expanded = tf.expand_dims(embedding, -1)

  pooled_output = []
  for i, ws in enumerate(window_size):
    with tf.variable_scope('conv_{}'.format(i)):
      # [window_size, kernel_size, stride_size, filter_number]
      kernel_shape = [ws, embedding_size, 1, filter_number]
      W = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.1), name='w')
      # b must hold the same size as the filter number
      b = tf.Variable(tf.constant(0.1, shape=[filter_number]), name='b')
      conv = tf.nn.conv2d(embedding_expanded,
                          W,
                          strides=[1, 1, 1, 1],
                          padding='VALID',
                          name='conv')
      # [batch_size, seq_len-ws+1, 1, filter_number]
      h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

      # MaxPool
      # [batch_size, 1, 1, filter_number]
      # pool_size == seq_len-ws+1
      pooled = tf.nn.max_pool(h,
                             ksize=[1, pool_size[i], 1, 1],
                             strides=[1, 1, 1, 1],
                             padding='VALID',
                             name='pool')
      pooled = tf.reshape(pooled, [batch_size, -1])
      pooled_output.append(pooled)

  # [batch_size, len(window_size)*filter_number]
  output = tf.concat(pooled_output, -1)  

  # dropout
  with tf.variable_scope('dropout'):
    output = tf.nn.dropout(output, keep_prob=(1-dropout_prob))
  
  return output
    
if __name__ == '__main__':
  ### Example for CRF ###
  # gold_labels = tf.constant([[1, 2, 3], [5, 6, 7]], dtype=tf.int32)
  # scores = tf.random.normal((2, 3, 10), dtype=tf.float32) 
  # sequence_lengths = tf.constant([3, 2], dtype=tf.int32)  

  # with tf.variable_scope('test'):
  #   log_likelihood, transition_params = crfEncode(scores, gold_labels, sequence_lengths)
  
  # pred_scores = tf.random.normal([3, 10], dtype=tf.float32)
  # viterbi_sequence, viterbi_score = crfDecode(pred_scores, transition_params)
  # print(viterbi_sequence)
  # print(viterbi_score)

  ### Example for textCNN ###
  embedding = tf.random.normal((10, 20, 30))
  window_size = [2, 3, 4]
  filter_number = 2
  pool_size = [19, 18, 17]
  dropout_prob = 0.1
  for _ in range(10):
    output = textCNN(embedding,
                    window_size,
                    filter_number,
                    pool_size,
                    dropout_prob)
    print(output.shape)