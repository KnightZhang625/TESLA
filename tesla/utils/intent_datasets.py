# coding:utf-8
# Produced by Jiaxin Zhang
# Start Data: 19_Aug_2020
# TensorFlow Version for parsing the Intent Datasets, i.e., SNIPS.
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

import gc
import sys
import copy
import json
import codecs
import random
import functools
import tensorflow as tf
tf.enable_eager_execution()
from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent.parent.parent
sys.path.append(str(MAIN_PATH))

from tesla.utils.data_pipeline import convertStrToIdx, createBatchIndex, padding_func

class IntentText(object):
  """Base Object for Intent Extraction Dataset."""
  cutFunc = lambda _, i: i if i <= 25 else i - 6

  def __init__(self, sentence_length, word_length):
    self.sentence_length = sentence_length
    self.word_length = word_length
    self.word_idx = {}
    self.tag_idx = {}
    self.idx_tag = {}
    self.intent_idx = {}
    self.idx_intent = {}
    self.char_idx = {chr(i) : self.cutFunc(i-65) for i in range(65, 123) 
                      if i not in [i for i in range(91, 97)]}
    self.char_idx['<unk>'] = 52
    self.char_idx['<pad>'] = 53
  
  def processData(self, train_data, test_data):
    """Interface for the inheriting Class."""
    train_size = len(train_data)
    test_size = len(test_data)
    texts, tags, intents = list(zip(*train_data + test_data))

    # create dictionary
    self.word_idx, _ = self.makeWordDict(texts)
    self.tag_idx, self.idx_tag = self.makeWordDict(tags)
    self.intent_idx, self.idx_intent = self.makeWordDict([set(intents)])

    # convert str to idx
    convertStrToIdx_word = functools.partial(convertStrToIdx, dic=self.word_idx)
    texts_idx = list(map(convertStrToIdx_word, texts))

    convertStrToIdx_tag = functools.partial(convertStrToIdx, dic=self.tag_idx)
    tags_idx = list(map(convertStrToIdx_tag, tags))
    
    convertStrToIdx_indent = functools.partial(convertStrToIdx, dic=self.intent_idx)
    indents_idx = list(convertStrToIdx_indent(intents))
    
    convertStrToIdx_char = functools.partial(convertStrToIdx, dic=self.char_idx)
    chars_idx = [list(map(convertStrToIdx_char, sent)) for sent in texts]

    assert len(texts_idx) == len(tags_idx) == len(indents_idx) == len(chars_idx)

    train_texts_idx, train_tags_idx, train_indents_idx, train_chars_idx = \
      texts_idx[:train_size], tags_idx[:train_size], indents_idx[:train_size], chars_idx[:train_size]
    test_texts_idx, test_tags_idx, test_indents_idx, test_chars_idx = \
      texts_idx[-test_size:], tags_idx[-test_size:], indents_idx[-test_size:], chars_idx[-test_size:]

    return zip(train_texts_idx, train_tags_idx, train_indents_idx, train_chars_idx), \
            zip(test_texts_idx, test_tags_idx, test_indents_idx, test_chars_idx)

  def makeWordDict(self, data):
    vocab_idx = {}
    idx_vocab = {}
    c = 0
    for sent in data:
      for w in sent:
        if w not in vocab_idx:
          vocab_idx[w] = c
          idx_vocab[c] = w
          c +=1
    vocab_idx['<unk>'] = c
    vocab_idx['<pad>'] = c + 1
    
    return vocab_idx, idx_vocab
  
  def generateData(self, datasets, batch_size=32):
    """Generator for Tensorflow Dataset"""
    # shuffle the data
    datasets_copy = list(copy.deepcopy(datasets))
    random.shuffle(datasets_copy)

    texts, tags, indents, chars = zip(*datasets_copy)
    data_length = len(texts)
    for s, e in createBatchIndex(data_length, batch_size):
      batch_texts = texts[s : e]
      input_texts_length = [len(sent) for sent in batch_texts]
      max_input_texts_length = max(input_texts_length)
      padding_func_texts = functools.partial(
        padding_func, max_length=max_input_texts_length, pad_idx=self.word_idx['<pad>'])
      input_texts = list(map(padding_func_texts, batch_texts))

      batch_tags = tags[s : e]
      batch_tags_length = [len(tag) for tag in batch_tags]
      padding_func_tag = functools.partial(
        padding_func, max_length=max(batch_tags_length), pad_idx=self.tag_idx['<pad>'])
      output_tags = list(map(padding_func_tag, batch_tags))
 
      output_indents = indents[s : e]
      
      batch_chars = chars[s : e]
      input_chars_length = [[len(char) for char in sent] for sent in batch_chars]
      padding_func_charLength = functools.partial(
        padding_func, max_length=max_input_texts_length, pad_idx=0)
      input_chars_length = list(map(padding_func_charLength, input_chars_length))

      max_char_length = max([max(sent_length) for sent_length in input_chars_length])
      padding_func_char = functools.partial(
        padding_func, max_length=max_char_length, pad_idx=self.char_idx['<pad>'])
      input_chars = [list(map(padding_func_char, sent)) for sent in batch_chars]

      padding_func_char_sent = functools.partial(
        padding_func, max_length=max_input_texts_length,
        pad_idx=[self.char_idx['<pad>'] for _ in range(max_char_length)])
      input_chars = list(map(padding_func_char_sent, input_chars))
      
      input_data = {'input_texts': input_texts,
                    'input_texts_length': input_texts_length,
                    'input_chars': input_chars,
                    'input_chars_length': input_chars_length}
      output_data = {'output_tags': output_tags,
                     'output_indents': output_indents}
      yield (input_data, output_data)

    del datasets_copy
    gc.collect()

class SNIPS(IntentText):
  """
  SNIPS dataset class

  Args:
          path (str): dataset path
          sentence_length (int, optional): max sentence length
          word_length (int, optional): max word length
  """
  train_files = [
      "AddToPlaylist/train_AddToPlaylist_full.json",
      "BookRestaurant/train_BookRestaurant_full.json",
      "GetWeather/train_GetWeather_full.json",
      "PlayMusic/train_PlayMusic_full.json",
      "RateBook/train_RateBook_full.json",
      "SearchCreativeWork/train_SearchCreativeWork_full.json",
      "SearchScreeningEvent/train_SearchScreeningEvent_full.json",
  ]
  test_files = [
      "AddToPlaylist/validate_AddToPlaylist.json",
      "BookRestaurant/validate_BookRestaurant.json",
      "GetWeather/validate_GetWeather.json",
      "PlayMusic/validate_PlayMusic.json",
      "RateBook/validate_RateBook.json",
      "SearchCreativeWork/validate_SearchCreativeWork.json",
      "SearchScreeningEvent/validate_SearchScreeningEvent.json",
  ]

  def __init__(self, 
               dir_path=MAIN_PATH / 'datasets/SNIPS',
               sentence_length=30,
               word_length=12):
    """Constructor for the object.

    Args:
      dir_path: Path type, the default directory path for the SNIPS datasets.
      sentence_length: maximum padding length for the input sentence.
      word_length: maxumum padding length for the word.
    """
    self.dir_path = dir_path
    train_data_raw, test_data_raw = self._loadDatasets()
    super(SNIPS, self).__init__(sentence_length=sentence_length, word_length=word_length)
    self.train_datas, self.test_datas = self.processData(train_data_raw, test_data_raw)

  def _loadDatasets(self):
    train_data = self._loadIntents(self.train_files)
    test_data = self._loadIntents(self.test_files)
    train_data = [(sent, tags, indent) for indent, indent_data in train_data.items() 
                          for sent, tags in indent_data]
    test_data = [(sent, tags, indent) for indent, indent_data in test_data.items()
                          for sent, tags in indent_data]
    return train_data, test_data

  def _loadIntents(self, files):
    datas = {}
    for path in files:
      abs_path = self.dir_path / path
      intent_name = path.split('/')[0]
      with codecs.open(abs_path, 'r', 'utf-8', errors='ignore') as file:
        data = json.load(file)
      # the data contains multiple data with the main key 'data'
      datas[intent_name] = self._parseJson([d['data'] for d in data[intent_name]])
    return datas

  def _parseJson(self, data):
    sentences = []
    for sent in data:
      tokens_list = []
      tags_list = []
      for te in sent:
        token = te['text'].strip().split(' ')
        # ignore the null
        if len(token[0]) == 0:
          continue
        tokens_list += token
        entity = te.get('entity', None)
        if entity is not None:
          tags_list += self._createTags(entity, len(token))
        else:
          tags_list += ['O'] * len(token)
      sentences.append((tokens_list, tags_list))
    return sentences
  
  def trainInputFn(self, is_train, batch_size, steps):
    output_types = {'input_texts': tf.int32,
                    'input_texts_length': tf.int32,
                    'input_chars': tf.int32,
                    'input_chars_length': tf.int32}
    output_shapes = {'input_texts': [None, None],
                     'input_texts_length': [None],
                     'input_chars': [None, None, None],
                     'input_chars_length': [None, None]}
    label_types = {'output_tags': tf.int32,
                    'output_indents': tf.int32}
    label_shapes = {'output_tags': [None, None],
                    'output_indents': [None]}
    
    data_generator = functools.partial(self.generateData, 
                        datasets=self.train_datas if is_train else self.test_datas,
                        batch_size=batch_size)
    dataset = tf.data.Dataset.from_generator(
      data_generator,
      output_types=(output_types, label_types),
      output_shapes=(output_shapes, label_shapes))
    dataset = dataset.repeat(steps)

    return dataset
  
  @staticmethod
  def _createTags(tag, length):
    """Return labels like ['B-tag', 'I-tag']."""
    labels = ['B-' + tag]
    for _ in range(1, length):
      labels.append('I-' + tag)
    return labels

if __name__ == '__main__':
  snips = SNIPS()
  
  # for data in snips.generateData(snips.train_datas, batch_size=2):
  #   print(data)
  #   input()

  for data in snips.trainInputFn(True, 100, 2):
    print(data)