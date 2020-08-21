# coding:utf-8
# Produced by Jiaxin Zhang
# Start Data: 21_Aug_2020
# TensorFlow Version for Multi Task Intent Model.
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
import tensorflow as tf
from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent.parent.parent
sys.path.append(str(MAIN_PATH))
from tesla.models import BaseModel


class MultiTaskIntentModel(BaseModel):
	def __init__(self,
							 config,
							 is_training,
							 input_texts,
							 input_texts_length,
							 input_chars,
							 input_chars_length,
							 output_tags=None):
		"""Constructor for MultiTaskIntentModel.

		Args:
			config: configuration.
			is_training: Boolean, control whether train or not.
			input_texts: tf.int32 Tensor with shape [batch_size, seq_length].
			input_texts_length: tf.int32 Tensor with shape [batch_size].
			input_chars: tf.int32 Tensor with shape [batch_size, seq_length, word_length].
			input_chars_length: tf.int32 Tensor with shape  [batch_size, seq_length].
			output_tags: tf.int32 Tensor with shape [batch_size, num_tags], 
				default None when not training.
		"""
		config = copy.deepcopy(config)
		self.is_training = is_training

if __name__ == '__main__':
	import random
	import tensorflow as tf
	tf.enable_eager_execution()

	from model_lego import createMultiRNNCells

	input_char = tf.random.normal((10, 30, 15, 128))
	input_length = tf.constant([15 - random.randint(0, 10)  for _ in range(30)], dtype=tf.int32)
	print(input_char.shape)
	print(input_length)

	char_embeddings = tf.keras.layers.TimeDistributed(
		tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))(input_char)
	print(char_embeddings.shape)  # (10, 30, 128)

	fw_cell = createMultiRNNCells(cell_type='lstm',
																hidden_size=64,
																num_layers=1,
																dropout_prob=0.01,
																num_residual_layers=0,
																forget_bias=True)

	bw_cell = createMultiRNNCells(cell_type='lstm',
																hidden_size=64,
																num_layers=1,
																dropout_prob=0.01,
																num_residual_layers=0,
																forget_bias=True)

	results = []
	for b_idx in range(10):
		single_sentence = input_char[b_idx, :]
		bi_outputs, bi_states = tf.nn.bidirectional_dynamic_rnn(
			fw_cell, bw_cell, single_sentence, dtype=tf.float32,
			sequence_length=input_length)
		results.append(tf.expand_dims(tf.concat((bi_states[0][-1], bi_states[1][-1]), axis=-1), axis=0))
	print(tf.concat(results, axis=0).shape)

	input_word = tf.random.normal((10, 30, 128))
	output = tf.keras.layers.Bidirectional(
						tf.keras.layers.LSTM(64, return_sequences=True, return_state=True))(input_word)
	print(output[:1][0].shape)
	print(len(output[1:]))