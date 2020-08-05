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

def createBatchIndex(total_length, batch_size):
  """Provide indices for the batch data."""
  batch_number = total_length // batch_size
  batch_number = batch_number if total_length % batch_size == 0 else batch_number + 1

  for i in range(batch_number):
    yield (i*batch_number, i*batch_number+batch_number)