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

import tensorflow as tf
from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent.parent.parent.parent

import config as cg
from tesla.models.cnn_lstm_crf import TaggerModel

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
    golden_labels = features['golden_labels'] if is_training else None

    # build the model
    model = TaggerModel(config,
                        is_training=is_training,
                        input_x=input_x,
                        input_length=input_length,
                        input_char=input_char,
                        golden_labels=golden_labels)
    log_likelihood = model.getResults('log_likelihood')
    transition_params = model.getResults('transition_params')
    

def main():
  Path(cg.SAVE_MODEL_PATH).mkdir(exist_ok=True)