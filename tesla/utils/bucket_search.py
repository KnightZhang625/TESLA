# coding:utf-8
# Produced by Jiaxin Zhang
# Start Data: 8_Aug_2020
# TensorFlow Version for Bucket Search.
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

import bisect
import numpy as np
from numpy.linalg import norm

def cosineDistance(vector, matrix):
  """
    Calculate the cosine distance between the vector and the matrix.
  Args:
    vector: numpy array with shape (1, X).
    matrix: numpy array with shape (N, X), where N refers to the number of the data.
  """
  return np.dot(vector, matrix.T) / (norm(vector) * norm(matrix, axis=1))

def sanityCheck(shape, bucket_size):
  if len(shape) != 2:
    raise ValueError('Array dimension should be 2, got \'{}\' instead.'.format(len(shape)))
  if shape[0] <= 1:
    raise ValueError('There is no need to use this method for one data.')
  if shape[0] < bucket_size:
    raise ValueError('Bucket size: \'{}\' should be smaller than the data number: \'{}\'.'
                      .format(bucket_size, shape[0]))

def getBucketNumber(bucket_size, total_number):
  bucket_number = total_number // bucket_size
  bucket_number = bucket_number if total_number % bucket_size == 0 else bucket_number + 1
  for idx in range(bucket_number):
    yield (idx * bucket_size, idx * bucket_size + bucket_size)

def buildSearchData(array, bucket_size=10):
  """Split the array and create the search data.
  
  Args:
    array: the data array, np.float32.
    bucket_size: the number of the data in each bucket.

  Returns:
    distances_sorted_indices: sorted indices according to the cosine distances with the base vector.
    baseDist_bucketArray: a dictionary, where key is the distance with the base vector, value is the corresponding arrays.
    baseDist_cache: cache the reference point of each bucket.
    sorted_array: the sorted array according to the cosine distances with the base vector.
    base_vector: the base vector.
  """
  # get the array shape and make sanity check
  shape = array.shape
  sanityCheck(shape, bucket_size)

  # select the first vector as the base vector
  # example shape, (1, 128)
  base_vector = np.reshape(array[0, :], (1, -1))

  # calculate the cosine distances with the base vector
  # example shape, (1, 128), (10, 128)
  distances = cosineDistance(base_vector, array)

  # sort the distances
  distances_sorted = np.sort(distances)[0][::-1]
  # SAVE OBJECT -> to find the actual index in the array
  distances_sorted_indices = np.argsort(distances)[0][::-1]

  # split the distances_sorted
  # the following dictionary use base distance as key, 
  # the bucket matrix and the distances indices as value
  baseDist_bucketArray = {}
  baseDist_cache = []
  sorted_array = []
  for (start, end) in getBucketNumber(bucket_size, shape[0]):
    base_distance = distances_sorted[start]

    # split the array according to the sorted indices
    split_array = []
    for true_idx in distances_sorted_indices[start:end]:
      split_array.append(np.reshape(array[true_idx, :], (1, -1)))
    split_array = np.concatenate(split_array, axis=0)
    sorted_array.append(split_array)

    # check whether this distance is already in the baseDist_bucketArray or not
    while base_distance in baseDist_bucketArray:
      base_distance += 1e-10
    baseDist_cache.append(base_distance)
    baseDist_bucketArray[base_distance] = split_array
  sorted_array = np.concatenate(sorted_array, axis=0)
  
  return distances_sorted_indices, baseDist_bucketArray, baseDist_cache, sorted_array, base_vector

class BucketSearch(object):
  """A object to search is benefical for the data load."""
  def __init__(self, search_data):
    self.distances_sorted_indices = search_data['distances_sorted_indices']
    self.baseDist_bucketArray = search_data['baseDist_bucketArray']
    self.baseDist_cache = search_data['baseDist_cache']
    self.sorted_array = search_data['sorted_array']
    self.base_vector = search_data['base_vector']
  
  def search(self, query_vector, search_scope):
    # calculate the cosine distance between the query_vector and the base_vector
    base_distance = cosineDistance(query_vector, self.base_vector)[0][0]

    # do not let one search destroy the baseDist_cache
    # as the insort() handles the ascending sort array, reverses it
    baseDist_cache_subs = self.baseDist_cache[:][::-1]
    bisect.insort(baseDist_cache_subs, base_distance)
    baseDist_cache_subs = baseDist_cache_subs[::-1]
    insert_idx = baseDist_cache_subs.index(base_distance)

    # find the margin indices
    if insert_idx == 0:
      margin_idx = 0
    elif insert_idx == len(baseDist_cache_subs) - 1:
      margin_idx = -1
    else:
      p_left = baseDist_cache_subs[insert_idx-1]
      # the list contains the base_distance, so plus one
      p_right = baseDist_cache_subs[insert_idx+1]
      absDist_left = abs(base_distance - p_left)
      absDist_right = abs(base_distance - p_right)
      # for original list, no base_distance, no need to plus one
      margin_idx = (insert_idx - 1) if absDist_left < absDist_right else insert_idx

    margin_distance = self.baseDist_cache[margin_idx]
    margin_array = self.baseDist_bucketArray[margin_distance]
    margin_query_distance = cosineDistance(query_vector, margin_array)
    margin_sorted = np.argsort(margin_query_distance)[0][::-1]
    margin_start = margin_idx * margin_array.shape[0]
    abs_index = margin_start + margin_sorted[0]

    # determine the search scope
    left_scope = abs_index - search_scope
    right_scope = abs_index + search_scope
    data_size = self.sorted_array.shape[0]
    left_scope = left_scope if left_scope > 0 else 0
    right_scope = right_scope if right_scope < data_size else -1

    # determine the search array, the search index
    search_array = self.sorted_array[left_scope:right_scope, :]
    search_indices = self.distances_sorted_indices[left_scope:right_scope]
    search_distances = cosineDistance(query_vector, search_array)[0]
    search_distances_sorted = np.argsort(search_distances)[::-1]
    # search_distances = np.sort(search_distances)[0][::-1]

    result_indices = []
    result_distances = []
    for idx in search_distances_sorted:
      result_indices.append(search_indices[idx])
      result_distances.append(search_distances[idx])

    return result_indices, result_distances

if __name__ == '__main__':
  toy_array = np.random.rand(120, 128)

  distances_sorted_indices,\
  baseDist_bucketArray,\
  baseDist_cache,\
  sorted_array,\
  base_vector = buildSearchData(toy_array)

  search_data = {'distances_sorted_indices': distances_sorted_indices,
                 'baseDist_bucketArray': baseDist_bucketArray,
                 'baseDist_cache': baseDist_cache,
                 'sorted_array': sorted_array,
                 'base_vector': base_vector}

  bucket_search = BucketSearch(search_data)
  print(bucket_search.search(np.reshape(toy_array[52, :], (1, -1)), 10))