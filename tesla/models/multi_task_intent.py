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
from tesla.models.utils import create_initializer
from tesla.models.model_lego import embedding_lookup,\
																		createMultiRNNCells,\
																		crfEncode, crfDecode

class MultiTaskIntentModel(BaseModel):
	def __init__(self,
							 config,
							 batch_size,
							 is_training,
							 input_texts,
							 input_texts_length,
							 input_chars,
							 input_chars_length,
							 output_tags=None):
		"""Constructor for MultiTaskIntentModel.

		Args:
			config: configuration.
			batch_size: required for embedding the character.
			is_training: Boolean, control whether train or not.
			input_texts: tf.int32 Tensor with shape [batch_size, seq_length].
			input_texts_length: tf.int32 Tensor with shape [batch_size].
			input_chars: tf.int32 Tensor with shape [batch_size, seq_length, word_length].
			input_chars_length: tf.int32 Tensor with shape  [batch_size, seq_length].
			output_tags: tf.int32 Tensor with shape [batch_size, num_tags], 
				default None when not training.
		"""
		config = copy.deepcopy(config)
		self.batch_size = batch_size
		self.is_training = is_training
		
		self.vocab_size = config.vocab_size
		self.char_size = config.char_size

		self.w_embedding_size = config.w_embedding_size
		self.c_embedding_size = config.c_embedding_size

		self.cell_type = config.cell_type
		self.hidden_size = config.hidden_size
		self.num_layers = config.num_layers
		self.forget_bias = config.forget_bias

		self.num_intents = config.num_intents
		self.num_tags = config.num_tags

		self.initialize_range = config.initialize_range
		self.dropout = config.dropout

		self.results = {}
		self.buildGraph(input_texts,
										input_texts_length,
										input_chars,
										input_chars_length,
										output_tags)
	
	def buildGraph(self,
								 input_texts,
								 input_texts_length,
								 input_chars,
								 input_chars_length,
								 output_tags=None):
		# build graph for the model
		with tf.variable_scope('MTI'):
			# word embedding
			with tf.variable_scope('word_embedding'):
				# (b, s) -> (b, s, e)
				word_embedding, _ = embedding_lookup(
					input_ids=input_texts,
					vocab_size=self.vocab_size,
					embedding_size=self.w_embedding_size,
					initialize_range=self.initialize_range)
			# 1st RNN on word
			with tf.variable_scope('word_rnn'):
				# num_layers refers to the sum of the forward and the backward
				assert_op = tf.assert_equal(self.num_layers % 2, 0)
				with tf.control_dependencies([assert_op]):
					num_bi_layers = int(self.num_layers / 2)
					num_bi_residual_layers = num_bi_layers - 1

					fw_cell_word = createMultiRNNCells(cell_type=self.cell_type,
																						 hidden_size=self.hidden_size,
																						 num_layers=num_bi_layers,
																						 dropout_prob=self.dropout,
																						 num_residual_layers=num_bi_residual_layers,
																						 forget_bias=self.forget_bias)

					bw_cell_word = createMultiRNNCells(cell_type=self.cell_type,
																						 hidden_size=self.hidden_size,
																						 num_layers=num_bi_layers,
																						 dropout_prob=self.dropout,
																						 num_residual_layers=num_bi_residual_layers,
																						 forget_bias=self.forget_bias)
					
					# bi_outputs_word: (fw, bw) <=> ((b, s, h), (b, s, h))
					# bi_states_word_temp: (fw, bw) <=> ([(b, h_f1), (b, h_f2)], [(b, h_b1), (b, h_b2)])
					bi_outputs_word, bi_states_word_temp = tf.nn.bidirectional_dynamic_rnn(
						fw_cell_word, bw_cell_word, word_embedding, dtype=tf.float32,
						sequence_length=input_texts_length)
					
					# (b, s, 2h)
					bi_outputs_word = tf.concat(bi_outputs_word, axis=-1)
					
					if num_bi_layers == 1:
						bi_states_word = bi_states_word_temp
					else:
						bi_states_word = []
						for layer_id in range(num_bi_layers):
							bi_states_word.append(bi_states_word_temp[0][layer_id])
							bi_states_word.append(bi_states_word_temp[1][layer_id])
						bi_states_word = tuple(bi_states_word)

			# use 1st RNN result to predict intent
			with tf.variable_scope('intent_predict'):
				# [(b, h_f_last), (b, h_b_last)] -> (b, 2h)
				last_time_layer_hidden_output = tf.concat(bi_outputs_word[-2:], axis=-1)
				intent_logits = tf.layers.dense(
					last_time_layer_hidden_output,
					self.num_intents,
					activation=tf.nn.relu,
					name='linear_intent_predict',
					kernel_initializer=create_initializer(initialize_range=self.initialize_range))
				self.results['intent_logits'] = intent_logits
			
			# char embedding
			with tf.variable_scope('char_embedding'):
				# (b, s, w) -> (b, s, w, e)
				char_embdding, _ = embedding_lookup(
					input_ids=input_chars,
					vocab_size=self.char_size,
					embedding_size=self.c_embedding_size,
					initialize_range=self.initialize_range)
			
			# 2nd RNN on char
			with tf.variable_scope('char_rnn'):
				fw_cell_char = createMultiRNNCells(cell_type=self.cell_type,
																					 hidden_size=self.hidden_size,
																					 num_layers=num_bi_layers,
																					 dropout_prob=self.dropout,
																					 num_residual_layers=num_bi_residual_layers,
																					 forget_bias=self.forget_bias)

				bw_cell_char = createMultiRNNCells(cell_type=self.cell_type,
																					 hidden_size=self.hidden_size,
																					 num_layers=num_bi_layers,
																					 dropout_prob=self.dropout,
																					 num_residual_layers=num_bi_residual_layers,
																					 forget_bias=self.forget_bias)
				# Important Node Here:
				#	As the char_embedding holds four dimensions, i.e., (b, s, w, e),
				# the tf.nn.bidirectional_dynamic_rnn() rejects input of four dimensions rather three,
				# so we ignore the batch first, and regard each sentence(2nd axis) as batch,
				# w(3rd axis) as sentence_length, so the final input should be (s, w, e).
				bi_outputs_char = []
				for b_idx in range(self.batch_size):
					single_sentence_inputs = char_embdding[b_idx, :]
					single_sentence_lengths = input_chars_length[b_idx]
					# bi_states_word_temp: (fw, bw) <=> ([(b, h_f1), (b, h_f2)], [(b, h_b1), (b, h_b2)])
					_, bi_states_char_single = tf.nn.bidirectional_dynamic_rnn(
						fw_cell_char, bw_cell_char, single_sentence_inputs, dtype=tf.float32,
						sequence_length=single_sentence_lengths)

					# pick the last time step hidden for each word -> (b, h), then concatenate
					fw_bi_states_char_single_last = bi_states_char_single[0][-1]			
					bw_bi_states_char_single_last = bi_states_char_single[0][-1]			
					bi_states_char_single_last = tf.concat((fw_bi_states_char_single_last,
																									bw_bi_states_char_single_last),
																									axis=-1)
					# expand the dimension for future batch size use -> (1, s, 2h)
					bi_outputs_char.append(tf.expand_dims(bi_states_char_single_last, axis=0))
				# (b, s, 2h)
				bi_outputs_char = tf.concat(bi_outputs_char, axis=0)
		
			# 3rd RNN
			# concatnate the word and char -> (b, s, 4h)
			final_inputs = tf.concat((bi_outputs_word, bi_outputs_char), axis=-1)
			with tf.variable_scope('final_rnn'):
				fw_cell_final = createMultiRNNCells(cell_type=self.cell_type,
																					 hidden_size=self.hidden_size,
																					 num_layers=num_bi_layers,
																					 dropout_prob=self.dropout,
																					 num_residual_layers=num_bi_residual_layers,
																					 forget_bias=self.forget_bias)

				bw_cell_final = createMultiRNNCells(cell_type=self.cell_type,
																					 hidden_size=self.hidden_size,
																					 num_layers=num_bi_layers,
																					 dropout_prob=self.dropout,
																					 num_residual_layers=num_bi_residual_layers,
																					 forget_bias=self.forget_bias)

				bi_outputs_final, _ = tf.nn.bidirectional_dynamic_rnn(
					fw_cell_final, bw_cell_final, final_inputs, dtype=tf.float32,
					sequence_length=input_texts_length)
				bi_outputs_final = tf.concat(bi_outputs_final, axis=-1)

			# linear transform for tag prediction
			with tf.variable_scope('tag_predict'):
				tag_logits = tf.layers.dense(
					bi_outputs_final,
					self.num_tags,
					activation=tf.nn.relu,
					name='linear_tag',
					kernel_initializer=create_initializer(initialize_range=self.initialize_range))
			self.results['tag_logits'] = tag_logits

			# send the logtis to the CRF
			with tf.variable_scope('crf', reuse=tf.AUTO_REUSE):
				self.transition_params = tf.get_variable(
					name='transition_params',
					shape=[self.num_tags, self.num_tags],
					initializer=create_initializer(initialize_range=self.initialize_range))
				if self.is_training:
					self.log_likelihood, _ = crfEncode(
						logits=tag_logits,
						labels=output_tags,
						sequence_lengths=input_texts_length,
						transition_params=self.transition_params)
					self.results['log_likelihood'] = self.log_likelihood
	
	def decode(self, logit, sequence_lengths):
		return crfDecode(logit, self.transition_params, sequence_lengths)
	
	def getResults(self, name):
		if name in self.results:
			return self.results[name]
		print('Cannot find `{}` in results.'.format(name))
		raise ValueError

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
		print(single_sentence.shape)
		bi_outputs, bi_states = tf.nn.bidirectional_dynamic_rnn(
			fw_cell, bw_cell, single_sentence, dtype=tf.float32,
			sequence_length=input_length)
		print(bi_states[0][-1].shape)
		results.append(tf.expand_dims(tf.concat((bi_states[0][-1], bi_states[1][-1]), axis=-1), axis=0))
	# print(tf.concat(results, axis=0).shape)

	# input_word = tf.random.normal((10, 30, 128))
	# output = tf.keras.layers.Bidirectional(
	# 					tf.keras.layers.LSTM(64, return_sequences=True, return_state=True))(input_word)
	# print(output[:1][0].shape)
	# print(len(output[1:]))