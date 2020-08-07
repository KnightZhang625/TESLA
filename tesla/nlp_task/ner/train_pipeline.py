# coding:utf-8
# Produced by Jiaxin Zhang
# Start Data: 6_Aug_2020
# TensorFlow Version for NER train pipeline.
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

import functools
import tensorflow as tf
from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent.parent.parent.parent

import config as cg
from data_pipeline import inputFn
from tesla.models.cnn_lstm_crf import TaggerModel
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
    input_x = features['input_x']
    input_length = features['input_length']
    input_char = features['input_char']
    golden_labels = labels['golden_labels'] if is_training else None

    # build the model
    model = TaggerModel(config,
                        is_training=is_training,
                        input_x=input_x,
                        input_length=input_length,
                        input_char=input_char,
                        golden_labels=golden_labels)
    log_likelihood = model.getResults('log_likelihood')
    transition_params = model.getResults('transition_params')
    
    # perdict
    if mode == tf.estimator.ModeKeys.PREDICT:
      viterbi_sequence, viterbi_score = model.decode(logit=log_likelihood, transition_params=transition_params)
      predicitions = {'viterbi_sequence': viterbi_sequence,
                     'viterbi_score': viterbi_score}
      output_spec = tf.estimator.EstimatorSpec(mode, predictions=predicitions)
    elif mode == tf.estimator.ModeKeys.TRAIN:
      loss = tf.reduce_mean(-log_likelihood)
      # add l2 loss
      tvars = tf.trainable_variables()
      l2_loss = 1e-2 * (tf.reduce_mean([tf.nn.l2_loss(v) for v in tvars]))
      loss += l2_loss
      
      # learning rate
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
                                                 'lr': lr}, every_n_iter=1)
      output_spec = tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])
    else:
      raise NotImplementedError
    
    return output_spec
  
  return model_fn

def main():
  Path(cg.SAVE_MODEL_PATH).mkdir(exist_ok=True)

  model_fn = modelFnBuilder(cg.model_config)

  gpu_config = tf.ConfigProto()
  gpu_config.gpu_options.allow_growth = True
  gpu_config.intra_op_parallelism_threads = 2
  gpu_config.inter_op_parallelism_threads = 2

  run_config = tf.estimator.RunConfig(
    session_config=gpu_config,
    keep_checkpoint_max=1,
    save_checkpoints_steps=cg.SAVE_STEPS,
    model_dir=cg.SAVE_MODEL_PATH)

	# # For TPU		
	# run_config = tf.contrib.tpu.RunConfig(
	# 	session_config=gpu_config,
	# 	keep_checkpoint_max=1,
	# 	save_checkpoints_steps=10,
	# 	model_dir=_cg.SAVE_MODEL_PATH)

  estimator = tf.estimator.Estimator(model_fn, config=run_config)
  input_fn = functools.partial(inputFn, data_path=cg.DATA_PATH, steps=cg.TRAIN_STEPS, batch_size=cg.BATCH_SZIE)
  estimator.train(input_fn)

if __name__ == '__main__':
  main()