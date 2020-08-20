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

import sys
import json
import codecs
import functools
from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent.parent.parent
sys.path.append(str(MAIN_PATH))

class IntentText(object):
  """Base Object for Intent Extraction Dataset."""
  cutFunc = lambda _, i: i if i <= 25 else i - 6

  def __init__(self, sentence_length, word_length):
    self.sentence_length = sentence_length
    self.word_length = word_length
    self.word_idx = {}
    self.tag_idx = {}
    # ignore the error report from the IDE
    self.char_idx = {chr(i) : self.cutFunc(i-65) for i in range(65, 123) 
                      if i not in [i for i in range(91, 97)]}
  
  def processData(self, train_data, test_data):
    

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
        tokens_list += token
        entity = te.get('entity', None)
        if entity is not None:
          tags_list += self._createTags(entity, len(token))
        else:
          tags_list += ['O'] * len(token)
      sentences.append((tokens_list, tags_list))
    return sentences
  
  @staticmethod
  def _createTags(tag, length):
    """Return labels like ['B-tag', 'I-tag']."""
    labels = ['B-' + tag]
    for _ in range(1, length):
      labels.append('I-' + tag)
    return labels

if __name__ == '__main__':
  snips = SNIPS()