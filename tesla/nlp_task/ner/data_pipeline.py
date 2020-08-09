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
import pickle
import random
import functools
import tensorflow as tf
tf.enable_eager_execution()
from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent.parent.parent.parent
sys.path.append(str(MAIN_PATH))
from tesla.utils.data_pipeline import createBatchIndex, convertStrToIdx, padding_func
from config import model_config

def createDictionary(data_path):
  vocab_idx = {}
  vocab_c = 0
  tag_idx = {}
  tag_c = 0
  with codecs.open(data_path, 'rb') as file:
    datas = pickle.load(file)
    for line in datas:
      for entry in line:
        vocab = entry.split(' ')[0]
        tag = entry.split(' ')[1]
        if vocab not in vocab_idx:
          vocab_idx[vocab] = vocab_c
          vocab_c += 1
        if tag not in tag_idx:
          tag_idx[tag] = tag_c
          tag_c += 1
  vocab_idx['<unk>'] = vocab_c
  vocab_idx['<padding>'] = vocab_c + 1
  tag_idx['<unk>'] = tag_c
  tag_idx['<padding>'] = tag_c + 1
  return vocab_idx, tag_idx

def loadDict():
  global vocab_idx
  global tag_idx
  global char_idx
  with codecs.open(MAIN_PATH / 'datasets/CONLL2000/vocab_idx.bin', 'rb') as file:
    vocab_idx = pickle.load(file)
  with codecs.open(MAIN_PATH / 'datasets/CONLL2000/tag_idx.bin', 'rb') as file:
    tag_idx = pickle.load(file)
  char_idx = {chr(i) : i-97 for i in range(97, 123)}
  char_idx['<unk>'] = 26
  char_idx['<padding>'] = 27
loadDict()
convertStrToIdx_vocab = functools.partial(convertStrToIdx, dic=vocab_idx)
convertStrToIdx_char = functools.partial(convertStrToIdx, dic=char_idx)
convertStrToIdx_tag = functools.partial(convertStrToIdx, dic=tag_idx)

def data_generator(data_path, batch_size):
  """This is the generator for the CONLL2000 dataset."""
  with codecs.open(data_path, 'rb') as file:
    datas = pickle.load(file)
  
  datas = copy.deepcopy(datas)
  random.shuffle(datas)

  for (start, end) in createBatchIndex(len(datas), batch_size):
    # get data, label with batch size
    data_batch = datas[start : end]
    input_x = [[entry.split(' ')[0]for entry in data] for data in data_batch]
    input_char = [[[char.lower() for char in vocab] for vocab in x] for x in input_x]
    golden_labels = [[entry.split(' ')[1] if entry.split(' ')[2] != 'O' else 'O' 
                  for entry in data] for data in data_batch]
    input_x_max_length = model_config.padding_seq_length 
    input_length = [len(x) if len(x) <= input_x_max_length else input_x_max_length for x in input_x]
    
    # convert str to idx
    input_x = list(map(convertStrToIdx_vocab, input_x))
    input_char = [list(map(convertStrToIdx_char, line)) for line in input_char]
    golden_labels = list(map(convertStrToIdx_tag, golden_labels))

    # padding
    # ATTENTION: As the length of char should be equal to the length of the input,
    # the model has a loop for textCNN, so fixed max length is required
    padding_func_input_x = functools.partial(padding_func, max_length=input_x_max_length, pad_idx=vocab_idx['<padding>'])
    input_x_padded = list(map(padding_func_input_x, input_x))
  
    padding_func_input_char = functools.partial(padding_func, max_length=15, pad_idx=char_idx['<padding>'])
    input_char_padded = [list(map(padding_func_input_char, line)) for line in input_char]
    # use [char_idx['<padding>'] * 15] to pad the last part
    padding_func_input_char_mid = functools.partial(
      padding_func, max_length=input_x_max_length, pad_idx=[char_idx['<padding>'] 
        for _ in range(15)])
    input_char_padded = list(map(padding_func_input_char_mid, input_char_padded))

    padding_func_golden_labels = functools.partial(padding_func, max_length=input_x_max_length, pad_idx=tag_idx['<padding>'])
    golden_labels_padded = list(map(padding_func_golden_labels, golden_labels))

    # send the data
    features = {'input_x': input_x_padded,
                'input_char': input_char_padded,
                'input_length': input_length}
    tags = {'golden_labels': golden_labels_padded}
    yield(features, tags)

def inputFn(data_path, steps, batch_size):
  """
    Input function for the Estimator.

    Args:
      data_path: absolute path, the data should be saved as the binary format.
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

  data_generator_with_args = functools.partial(data_generator,
                                                data_path=data_path,
                                                batch_size=batch_size)
  dataset = tf.data.Dataset.from_generator(
    data_generator_with_args,
    output_types=(output_types, label_types),
    output_shapes=(output_shapes, label_shapes))
  
  dataset = dataset.repeat(steps)
  return dataset

def serverInputFunc():
  input_x = tf.placeholder(tf.int32, shape=[None, None], name='input_x')
  input_length = tf.placeholder(tf.int32, shape=[None], name='input_length')
  input_char = tf.placeholder(tf.int32, shape=[None, None, None], name='input_char')

  receiver_tensors = {'input_x': input_x,
                      'input_length': input_length,
                      'input_char': input_char}
  features = {'input_x': input_x,
              'input_length': input_length,
              'input_char': input_char}
  
  return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

if __name__ == '__main__':
  ### Test input_fn ###
  conll2000_data_path = MAIN_PATH / 'datasets/CONLL2000/train.bin'
  for data in inputFn(conll2000_data_path, 1, 100):
    print(data)
    input()

  ### create the dictionary ###
  # vocab_idx, tag_idx = createDictionary(conll2000_data_path)
  # with codecs.open(MAIN_PATH / 'datasets/CONLL2000/vocab_idx.bin', 'wb') as file:
  #   pickle.dump(vocab_idx, file)
  # with codecs.open(MAIN_PATH / 'datasets/CONLL2000/tag_idx.bin', 'wb') as file:
  #   pickle.dump(tag_idx, file)