# coding:utf-8
# Produced by Jiaxin Zhang
# Start Data: 24_Aug_2020
# TensorFlow Version for Intent Extraction train pipeline.
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
import tensorflow as tf
from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent.parent.parent.parent
sys.path.append(str(MAIN_PATH))

import tesla.nlp_task.IntentExtraction.config as cg
from tesla.models.multi_task_intent import MultiTaskIntentModel
from tesla.utils.intent_datasets import SNIPS
from tesla.utils.functools import Setup

setup = Setup(path='log', log_name='tensorflow')

def modelFnBuilder(config):
  """Returns 'model_fn' closure for Estimator."""
  def model_fn(features, labels, mode, params):
    print('*** Features ***')
    for name in sorted(features.keys()):
      tf.logging.info(' name = {}, shape = {}'.format(name, features[name].shape))
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # get the data
    input_texts = features['input_texts']
    input_texts_length = features['input_texts_length']
    input_chars = features['input_chars']
    input_chars_length = features['input_chars_length']
    output_tags = labels['output_tags'] if is_training else None

    # build the model
    model = MultiTaskIntentModel(config,
                                 cg.BATCH_SIZE,
                                 is_training,
                                 input_texts=input_texts,
                                 input_texts_length=input_texts_length,
                                 input_chars=input_chars,
                                 input_chars_length=input_chars_length,
                                 output_tags=output_tags)
    
    # predict
    if  mode == tf.estimator.ModeKeys.PREDICT:
      intent_logits = model.getResults('intent_logits')
      intent_probs = tf.nn.softmax(intent_logits, axis=-1)
      intent_labels = tf.nn.argmax(intent_probs, axis=-1)

      tag_logits = model.getResults('tag_logits')
      viterbi_sequence, viterbi_score = model.decode(logit=tag_logits, sequence_lengths=input_texts_length)
      
      predictions = {'intent_labels': intent_labels,
                     'viterbi_sequence': viterbi_sequence,
                     'viterbi_score': viterbi_score}
      output_spec = tf.estimator.EstimatorSpec(mode, predictions)
    elif mode == tf.estimator.ModeKeys.TRAIN:
      gold_intent_labels = labels['output_indents']
      intent_logits = model.getResults('intent_logits')
      max_time = tf.shape(gold_intent_labels)[1]
      target_weights = tf.sequence_mask(input_texts_length, max_time, dtype=intent_logits.dtype)
      batch_size = tf.cast(cg.BATCH_SIZE, dtype=tf.float32)

      intent_loss = tf.reduce_sum(
          tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=gold_intent_labels, logits=intent_logits) * target_weights) / batch_size

      tag_log_likelihood = model.getResults('log_likelihood')
      tag_loss = tf.reduce_mean(-tag_log_likelihood)

      loss = intent_loss + tag_loss
      tvars = tf.trainable_variables()
      l2_loss = 1e-2 * (tf.reduce_mean([tf.nn.l2_loss(v) for v in tvars]))
      loss += l2_loss

      lr = tf.train.polynomial_decay(
        cg.LEARNING_RATE,
        tf.train.get_or_create_global_step(),
        cg.TRAIN_STEPS)
      lr = tf.maximum(tf.constant(cg.LEARNING_RATE_LIMIT), lr)

      # create optimizer and update
      optimizer = tf.train.AdamOptimizer(learning_rate=lr)
      gradients = tf.gradients(loss, tvars, colocate_gradients_with_ops=True)
      clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
      train_op = optimizer.apply_gradients(zip(clipped_gradients, tvars), global_step=tf.train.get_global_step())

      logging_hook = tf.train.LoggingTensorHook({'step': tf.train.get_global_step(),
                                                 'loss': loss,
                                                 'l2_loss': l2_loss,
                                                 'lr': lr,
                                                 'intent_loss': intent_loss,
                                                 'tag_loss': tag_loss}, every_n_iter=1)
      output_spec = tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])
    else:
      raise NotImplementedError
    
    return output_spec
  
  return model_fn

def main(**kwargs):
  save_model_path = kwargs['save_path']
  Path(save_model_path).mkdir(exist_ok=True)




