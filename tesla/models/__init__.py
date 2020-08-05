# coding:utf-8
# Produced by Jiaxin Zhang
# Start Data: 4_Aug_2020
# TensorFlow Version for Base Model.
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
from pathlib import Path
from abc import ABC

class Setup(object):
    """Setup logging"""
    def __init__(self, path, log_name=None, file_name='main.log', verbosity=logging.INFO):
        Path(path).mkdir(exist_ok=True)
        tf.compat.v1.logging.set_verbosity(verbosity)
        handlers = [logging.FileHandler(str(path) + '/' + file_name),
                    logging.StreamHandler(sys.stdout)]
        logging.getLogger(log_name).handlers = handlers
        self.logger = logging

# setup = Setup(path='./log', log_name='tensorflow')

class BaseModel(ABC):
  """Base Model to inherit."""

  def buildGraph(self, *args, **kwargs):
    raise NotImplementedError
  
  def getResults(self, *args, **kwargs):
    raise NotImplementedError