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
import argparse
import functools
import tensorflow as tf
from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent.parent.parent.parent
sys.path.append(str(MAIN_PATH))

import tesla.nlp_task.IntentExtraction.config as cg
from tesla.models.multi_task_intent import MultiTaskIntentModel
from tesla.utils.intent_datasets import SNIPS
from tesla.utils.functools import Setup

setup = Setup(path='log', log_name='tensorflow')

def createArgumentParser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', type=str, required=True, 
    help='Choose \'train\' or \'package\' the model.')
  parser.add_argument('--intra', type=int, default=0,
    help='Better equals to the physical cpu numbers.')
  parser.add_argument('--inter', type=int, default=0,
    help='Experiment starts from 2.')
  parser.add_argument('--ckpt', type=str, default=cg.SAVE_MODEL_PATH,
    help='The path where the model saved.')
  parser.add_argument('--pb', type=str, default=cg.PB_MODEL_PATH,
    help='The path where the pb model to saved.')

  return parser

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
      intent_labels = tf.math.argmax(intent_probs, axis=-1)

      tag_logits = model.getResults('tag_logits')
      viterbi_sequence, viterbi_score = model.decode(logit=tag_logits, sequence_lengths=input_texts_length)
      
      predictions = {'intent_labels': intent_labels,
                     'viterbi_sequence': viterbi_sequence,
                     'viterbi_score': viterbi_score}
      output_spec = tf.estimator.EstimatorSpec(mode, predictions)
    elif mode == tf.estimator.ModeKeys.TRAIN:
      gold_intent_labels = labels['output_indents']
      intent_logits = model.getResults('intent_logits')

      # max_time = tf.shape(gold_intent_labels)[1]
      # target_weights = tf.sequence_mask(input_texts_length, max_time, dtype=intent_logits.dtype)
      batch_size = tf.cast(cg.BATCH_SIZE, dtype=tf.float32)

      intent_loss = tf.reduce_sum(
          tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=gold_intent_labels, logits=intent_logits)) / batch_size

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

  model_fn = modelFnBuilder(cg.model_config)

  gpu_config = tf.ConfigProto()
  gpu_config.gpu_options.allow_growth = True
  intra = kwargs['intra']
  inter = kwargs['inter']
  gpu_config.intra_op_parallelism_threads = intra
  gpu_config.inter_op_parallelism_threads = inter

  run_config = tf.estimator.RunConfig(
    session_config=gpu_config,
    keep_checkpoint_max=1,
    save_checkpoints_steps=cg.SAVE_STEPS,
    model_dir=save_model_path)

	# # For TPU		
	# run_config = tf.contrib.tpu.RunConfig(
	# 	session_config=gpu_config,
	# 	keep_checkpoint_max=1,
	# 	save_checkpoints_steps=10,
	# 	model_dir=_cg.SAVE_MODEL_PATH)

  estimator = tf.estimator.Estimator(model_fn, config=run_config)
  snips = SNIPS()
  
  input_fn = functools.partial(snips.trainInputFn, is_train=True, batch_size=cg.BATCH_SIZE, steps=cg.TRAIN_STEPS)
  estimator.train(input_fn)

def package(**kwargs):
  ckpt_path = kwargs['ckpt_path']
  pb_path = kwargs['pb_path']

  model_fn = modelFnBuilder(cg.model_config)
  estimator = tf.estimator.Estimator(model_fn, ckpt_path)
  Path(pb_path).mkdir(exist_ok=True)

  snips = SNIPS()
  estimator.export_saved_model(pb_path, snips.serverInputFn)

if __name__ == '__main__':
  parser = createArgumentParser()
  args = parser.parse_args()

  mode = args.mode
  intra = args.intra
  inter = args.inter
  save_path = args.ckpt
  pb_path = args.pb

  if mode == 'train':
    main(save_path=save_path, intra=intra, inter=inter)
  elif mode == 'package':
    package(ckpt_path=save_path, pb_path=pb_path)