# coding:utf-8
# Produced by Jiaxin Zhang
# Start Data: 5_Aug_2020
# TensorFlow Version for data pipeline.
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

import codecs
import pickle
import functools
from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent.parent.parent

convertStrToIdx = lambda entry, dic : [dic[vocab] if vocab in dic else dic['<unk>'] 
                                        for vocab in entry]
padding_func = lambda line, max_length, pad_idx : line + [pad_idx  for _ in range(max_length - len(line))] if max_length >= len(line) else line[:max_length]

def createBatchIndex(total_length, batch_size):
  """Provide indices for the batch data."""
  batch_number = total_length // batch_size
  batch_number = batch_number if total_length % batch_size == 0 else batch_number + 1

  for i in range(batch_number):
    yield (i*batch_size, i*batch_size+batch_size)

def _createDefaultSavePath(func):
  """Decorator for creating the binary file name."""
  @functools.wraps(func)
  def _createDefaultSavePathWrapper(dataset_name, data_path, save_path=None):
    if save_path is not None:
      func(dataset_name, data_path, save_path)
    else:
      data_path = Path(data_path)
      path_parent = data_path.parent
      data_name = data_path.stem + '.bin'
      save_path = path_parent / data_name
      func(dataset_name, data_path, save_path)
    print('"{}" has been converted successfully to "{}".'.format(data_path.name, save_path))
  
  return _createDefaultSavePathWrapper

@_createDefaultSavePath
def convertToBinaryData(dataset_name, data_path, save_path=None):
  """Convert the txt format file to the binary file."""
  if dataset_name == 'conll2000':
    with codecs.open(data_path, 'r', 'utf-8') as file:
      datas = []
      one_data = []
      for line in file:
        line = line.strip()
        if len(line) != 0:
          one_data.append(line)
        else:
          if len(one_data) != 0:
            datas.append(one_data)
            one_data = []
  else:
    print('Do not support this dataset: "{}".'.format(dataset_name))
    raise ValueError
  
  print('Data total length: {}.'.format(len(datas)))
  with codecs.open(save_path, 'wb') as file:
    pickle.dump(datas, file)

if __name__ == '__main__':
  ### Convert CONLL2000 Dataset ###
  conll2000_data_path = MAIN_PATH / 'datasets/CONLL2000/train.txt'
  convertToBinaryData('conll2000', conll2000_data_path)