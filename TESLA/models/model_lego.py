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

import logging
import tensorflow as tf
# tf.enable_eager_execution()

from utils import assert_rank

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

if __name__ == '__main__':
  ### Example for CRF ###
  gold_labels = tf.constant([[1, 2, 3], [5, 6, 7]], dtype=tf.int32)
  scores = tf.random.normal((2, 3, 10), dtype=tf.float32) 
  sequence_lengths = tf.constant([3, 2], dtype=tf.int32)  

  with tf.variable_scope('test'):
    log_likelihood, transition_params = crfEncode(scores, gold_labels, sequence_lengths)
  
  pred_scores = tf.random.normal([3, 10], dtype=tf.float32)
  viterbi_sequence, viterbi_score = crfDecode(pred_scores, transition_params)
  print(viterbi_sequence)
  print(viterbi_score)