# coding:utf-8
# Produced by Jiaxin Zhang
# Start Data: 7_Aug_2020
# TensorFlow Version for NER predict pipeline.
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

import copy
import argparse
import functools
import config as cg
from config import model_config
from data_pipeline import vocab_idx, char_idx, tag_idx
from data_pipeline import convertStrToIdx_vocab, convertStrToIdx_char, convertStrToIdx_tag
from tesla.utils.functools import restorePath, convertType
from tesla.utils.data_pipeline import padding_func

idx_tag = {}
for key, value in tag_idx.items():
  idx_tag[value] = key

def createArgumentParser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--pb', type=str, default=cg.PB_MODEL_PATH,
    help='The path where the pb model saved.')

  return parser

@convertType(flag='nested_list')
def predict(model, sentence):
  input_x = [line.split(' ') for line in sentence]
  input_x_back = copy.deepcopy(input_x)
  input_char = [[[char.lower() for char in vocab] for vocab in x] for x in input_x]
  input_length = [len(x) for x in input_x]

  # convert str to idx
  input_x = list(map(convertStrToIdx_vocab, input_x))
  input_char = [list(map(convertStrToIdx_char, line)) for line in input_char]

  # padding
  input_x_max_length = model_config.padding_seq_length 
  padding_func_input_x = functools.partial(padding_func, max_length=input_x_max_length, pad_idx=vocab_idx['<padding>'])
  input_x_padded = list(map(padding_func_input_x, input_x))

  padding_func_input_char = functools.partial(padding_func, max_length=15, pad_idx=char_idx['<padding>'])
  input_char_padded = [list(map(padding_func_input_char, line)) for line in input_char]
  padding_func_input_char_mid = functools.partial(
      padding_func, max_length=input_x_max_length, pad_idx=[char_idx['<padding>'] 
        for _ in range(15)])
  input_char_padded = list(map(padding_func_input_char_mid, input_char_padded))
  
  features = {'input_x': input_x_padded,
              'input_char': input_char_padded,
              'input_length': input_length}
  predicitons = model(features)
  viterbi_sequence, viterbi_score = predicitons['viterbi_sequence'], predicitons['viterbi_score']

  results = []
  for i, res in enumerate(viterbi_sequence):
    actucal_res = res[: input_length[i]]
    single_res = []
    single_sentence = input_x_back[i]
    for j, t_i in  enumerate(actucal_res):
      single_res.append((single_sentence[j], idx_tag[t_i]))
    results.append(single_res)

  return results

if __name__ == '__main__':
  parser = createArgumentParser()
  args = parser.parse_args()
  model_path = args.pb

  model = restorePath(model_path)
  toy_data = ['Confidence in the pouond is widely expected', 'Chancellor of the Exchequer Nigel Lawson']
  results = predict(model, toy_data)
  print(results)