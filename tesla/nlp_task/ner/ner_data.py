# coding:utf-8
# Produced by Jiaxin Zhang
# Start Data: 5_Aug_2020
# TensorFlow Version for NER data pipeline.
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
import codecs
import random
import tensorflow as tf
from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent.parent.parent
sys.path.insert(0, str(MAIN_PATH))
from tesla.utils.data_pipeline import createBatchIndex

def data_generator(data_path, batch_size):
  """This is the generator for the CONLL2000 dataset."""
  with codecs.open(data_path, 'r', 'utf-8') as file:
    data = file.read().split('\n')
  
  data = copy.deepcopy(data)
  random.shuffle(data)

  for (start, end) in createBatchIndex(len(data), batch_size):
    data_batch = data[start : end]



def input_fn(data_path, steps, batch_size):
  """
    Input function for the Estimator.

    Args:
      data_path: absolute path, the data should be saved as the txt format.
      steps: train steps.
  """
  output_types = {'input_x': tf.int32,
                  'input_length': tf.int32,
                  'input_char': tf.int32}
  output_shapes = {'input_x': [None, None],
                   'input_length': [None],
                   'input_char': [None, None, None]}
  label_types = {'golden_labels': tf.int32}
  label_shapes = {'golden_labels': [None, None]}