# coding:utf-8
# Produced by Jiaxin Zhang
# Start Data: 4_Aug_2020
# TensorFlow Version for Utilization Module.
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

import six
# import logging
import tensorflow as tf

# logger = logging.getLogger()
# logging.set_verbosity(logging.ERROR)

def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of shape of tensor, preferring static dimensions,
      sometimes, the dimension is None.
            
  Args:
      tensor: a tf.Tensor which needs to find the shape.
      expected_rank: (optional) int. The expected rank of 'tensor'. If this is
          specified and the 'tensor' has a different rank, an error will be thrown.
      name: (optional) name of the 'tensor' when throwing the error.

  Returns:
      Dimensions as list type of the tensor shape.
      All static dimension will be returned as python integers,
      and dynamic dimensions will be returned as tf.Tensor scalars.
  """
  if name is None:
      name = tensor.name

  if expected_rank is not None:
      assert_rank(tensor, expected_rank, name)
  
  shape = tensor.shape.as_list()

  # save the dimension which is None
  dynamic_indices = []
  for (index, dim) in enumerate(shape):
      if dim is None:
          dynamic_indices.append(index)
  
  # dynamic_indices list is None
  if not dynamic_indices:
      return shape
  
  # replace the dynamic dimensions
  dynamic_shape = tf.shape(tensor)
  for index in dynamic_indices:
      shape[index] = dynamic_shape[index]
  return shape

def assert_rank(tensor, expected_rank, name=None):
  """Check whether the rank of the 'tensor' matches the expecetd_rank.
      Remember rank is the number of the total dimensions.
      
  Args:
      tensor: A tf.tensor to check.
      expected_rank: Python integer or list of integers.
      name: (optional) name of the 'tensor' when throwing the error.    
  """
  if name is None:
      name = tensor.name
  
  expected_rank_dict = {}
  # save the given rank into the dictionary, 
  # given rank could be either an integer or a list.
  if isinstance(expected_rank, six.integer_types):
      expected_rank_dict[expected_rank] = True
  else:
      for rank in expected_rank:
          expected_rank_dict[rank] = True

  tensor_rank = tensor.shape.ndims
  if tensor_rank not in expected_rank_dict:
      scope_name = tf.get_variable_scope().name
      print('For the tensor {} in scope {}, the tensor rank {} \
              (shape = {}) is not equal to the expected_rank {}'.format(
          name, scope_name, tensor_rank, str(tensor.shape), str(expected_rank)))
      raise ValueError

def create_initializer(init_type='trunc', initialize_range=0.02):
  if init_type is 'trunc':
    return tf.truncated_normal_initializer(stddev=initialize_range)
  else:
    raise NotImplementedError('Initialize Type: `{}` not implemented.'.format(init_type))