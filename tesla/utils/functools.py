# coding:utf-8
# Produced by Jiaxin Zhang
# Start Data: 7_Aug_2020
# TensorFlow Version for helpful functions.
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
import functools
import tensorflow as tf
from pathlib import Path
from tensorflow.contrib import predictor

class Setup(object):
    """Setup logging"""
    def __init__(self, path, log_name=None, file_name='main.log', verbosity=logging.INFO):
        Path(path).mkdir(exist_ok=True)
        tf.compat.v1.logging.set_verbosity(verbosity)
        handlers = [logging.FileHandler(str(path) + '/' + file_name),
                    logging.StreamHandler(sys.stdout)]
        logging.getLogger(log_name).handlers = handlers
        self.logger = logging

def forbid_new_attributes(wrapped_setatrr):
  def __setattr__(self, name, value):
      if hasattr(self, name):
          wrapped_setatrr(self, name, value)
      else:
          print('Add new {} is forbidden'.format(name))
          raise AttributeError
  return __setattr__

class NoNewAttrs(object):
  """forbid to add new attributes"""
  __setattr__ = forbid_new_attributes(object.__setattr__)
  class __metaclass__(type):
      __setattr__ = forbid_new_attributes(type.__setattr__)

def restorePath(pb_path):
  """Restore the latest model from the given path."""
  subdirs = [x for x in Path(pb_path).iterdir()
             if x.is_dir() and 'temp' not in str(x)]
  latest_model = str(sorted(subdirs)[-1])
  predict_fn = predictor.from_saved_model(latest_model)

  return predict_fn

def convertType(flag):
  def convertInputToPredict(func):
    @functools.wraps(func)
    def convertInputToPredictWrapper(*args, **kwargs):
      model, input_data = args[0], args[1]
      if flag == 'nested_list':
        if type(input_data) is str:
          return func(model, [input_data])
        else:
          return func(model, input_data)
      else:
        raise ValueError('Not support {}.'.format(flag))

    return convertInputToPredictWrapper
  return convertInputToPredict