from __future__ import print_function
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

import sys
import keras
import numpy as np
from keras import optimizers
from seq2seq.models import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq
from generate_datasets import DataGenerator

class trainModel(object):
	def __init__(self, train_names, train_codes, model_name, hyperparams=None):
		self.train_names = train_names
		self.train_codes = train_codes
		self.hyperparams = hyperparams
		self.model_name = model_name

		self.model = None
		self.parameters = None # if the model has been trained, parameters will not be None.
		self.naming_data = None # the dictionary to decode/encode tokens

		self.output_folder = self.model_name + 'Results'
		if not os.path.exists(self.output_folder):
			os.mkdir(self.output_folder)

		# self._check_all_hyperparmeters_exist()
		# for seq2seq models
		# self.n_tokens = self.naming_data.all_tokens_dictionary.get_n_tokens()


	def _check_all_hyperparmeters_exist(self):
		if self.model_name == 'SimpleSeq2Seq':
			'''
			output_dim: for the second last layer of decoder
			output_length: the same with maximum name size
			input_length: the same with maximum code size
			'''
			all_params = ['output_dim', 'output_length', 'input_length']

			for param in all_params:
				assert param in self.hyperparams, param

	@staticmethod
	def one_hot_name(names, name_dim):
		n_samples, n_timesteps = names.shape
		X = np.zeros((n_samples, n_timesteps, name_dim))
		for i, name in enumerate(names):
			for j, token in enumerate(name):
				X[i, j, token] = 1.0
		return X

	def grid_search(self):
		assert len(self.train_names) == len(self.train_codes), (len(self.train_names), len(self.train_codes))

		output_dim = [256]
		output_length = [8]
		hidden_dim = [256]
		batch_size = [500] #, 200]
		input_length = [300]# , 500]
		depth = [1, 3]
		dropout = [0.3, 0.5]
		lr = [0.001, 0.0005] #, 0.0005, 0.001, 0.005]
		num_epoch = [50, 50] #, 500]
		pct_train = 0.9
		peek = [False, True]
		broadcast_state = [True]
		bidirectional = [False]
		best_f1 = 0
		best_hyparams = None
		best_model = None
		# grid search
		with open(self.output_folder + '/' + 'grid_search.txt', 'w') as f:
			for t_output_dim in output_dim:
				for t_output_length in output_length:
					for t_hidden_dim in hidden_dim:
						for t_batch_size in batch_size:
							for t_input_length in input_length:
								for t_depth in depth:
									for t_dropout in dropout:
										for t_lr in lr:
											for t_num_epoch in num_epoch:
												if self.model_name == 'SimpleSeq2Seq':
													hyperparams = dict(output_dim=t_output_dim, output_length=t_output_length, hidden_dim=t_hidden_dim, batch_size=t_batch_size, input_length=t_input_length, depth=t_depth, dropout=t_dropout)
													f.write(str(hyperparams) + '\n')
													model, exact_match, precision, recall, f1 = self.train(self.train_names, self.train_codes, hyperparams, pct_train, t_lr, t_num_epoch)
													f.write('exact match=%f, precision=%f, recall=%f, f1=%f\n\n' % (exact_match, precision, recall, f1))
													if f1 > best_f1: # use f1 to optimize hyperparams
														best_f1 = f1
														best_model = model
														best_hyparams = hyperparams
												elif self.model_name == 'Seq2Seq':
													for t_peek in peek:
														for t_broadcast_state in broadcast_state:
															hyperparams = dict(output_dim=t_output_dim, output_length=t_output_length, hidden_dim=t_hidden_dim, batch_size=t_batch_size, input_length=t_input_length, depth=t_depth, dropout=t_dropout, peek=t_peek, broadcast_state=t_broadcast_state)
															f.write(str(hyperparams) + '\n')
															model, exact_match, precision, recall, f1 = self.train(self.train_names, self.train_codes, hyperparams, pct_train, t_lr, t_num_epoch)
															f.write('exact match=%f, precision=%f, recall=%f, f1=%f\n\n' % (exact_match, precision, recall, f1))
															if f1 > best_f1: # use f1 to optimize hyperparams
																best_f1 = f1
																best_model = model
																best_hyparams = hyperparams
												elif self.model_name == 'AttentionSeq2Seq':
													for t_bidirectional in bidirectional:
														hyperparams = dict(output_dim=t_output_dim, output_length=t_output_length, hidden_dim=t_hidden_dim, batch_size=t_batch_size, input_length=t_input_length, depth=t_depth, dropout=t_dropout, bidirectional=t_bidirectional)
														f.write(str(hyperparams) + '\n')
														model, exact_match, precision, recall, f1 = self.train(self.train_names, self.train_codes, hyperparams, pct_train, t_lr, t_num_epoch)
														f.write('exact match=%f, precision=%f, recall=%f, f1=%f\n\n' % (exact_match, precision, recall, f1))
														if f1 > best_f1: # use f1 to optimize hyperparams
															best_f1 = f1
															best_model = model
															best_hyparams = hyperparams		
			f.write('the best hyperparam is %s' % str(best_hyparams))
		print ('the best hyperparam is ', str(best_hyparams))
		best_model.save(self.output_folder + '/best_' + self.model_name + '.h5')
		return best_f1, best_hyparams, best_model

	def train(self, train_names, train_codes, hyperparams, pct_train=0.8, lr=0.01, num_epoch=100):
		if self.model_name == 'SimpleSeq2Seq' or self.model_name == 'AttentionSeq2Seq':
			# split data into  training and validation
			train_name, train_code, val_name, val_code, naming_data, hyperparams['n_tokens'] = trainModel.split_data(train_names, train_codes, hyperparams['output_length'], hyperparams['input_length'], pct_train)
			# set hyperparams
			hyperparams['is_embedding'] = False
			# convert target name into one-hot encoding
			train_name = trainModel.one_hot_name(train_name, hyperparams['n_tokens'])
			# check if required params exist
			required_params = ['output_dim', 'output_length', 'input_length', 'is_embedding', 'n_tokens']
			for param in required_params:
				assert param in hyperparams, (param)
			# create the model
			if self.model_name == 'SimpleSeq2Seq':
				model = SimpleSeq2Seq(**hyperparams)
			elif self.model_name == 'AttentionSeq2Seq':
				model = AttentionSeq2Seq(**hyperparams)
			my_adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
			model.compile(optimizer=my_adam, loss='categorical_crossentropy')
			print ('fit...')
			model.fit(train_code, train_name, epochs=num_epoch)

			
		if self.model_name == 'Seq2Seq':
			train_name, train_code, val_name, val_code, naming_data, hyperparams['n_tokens'] = trainModel.split_data(train_names, train_codes, hyperparams['output_length'], hyperparams['input_length'], pct_train)
			hyperparams['is_embedding'] = False
			train_name = trainModel.one_hot_name(train_name, hyperparams['n_tokens'])
			# check if required params exist
			required_params = ['output_dim', 'output_length', 'input_length', 'is_embedding', 'n_tokens']
			for param in required_params:
				assert param in hyperparams, (param)
			# create the model
			model = Seq2Seq(**hyperparams)
			my_adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
			model.compile(optimizer=my_adam, loss='categorical_crossentropy')
			if 'teacher_force' in hyperparams and hyperparams['teacher_force'] == True:
				hyperparams['unroll'] = True
				inputs = [train_code, train_name]
			else:
				inputs = train_code
			model.fit(inputs, train_name, epochs=num_epoch)

		print ('predict...')
		predict_probs = model.predict(val_code)
		predict_idx = np.argmax(predict_probs, axis=2)

		print('evaluate...')
		exact_match = trainModel.exact_match(naming_data, predict_idx, val_name)
		precision, recall, f1 = trainModel.evaluate_tokens(naming_data, predict_idx, val_name)
		return model, exact_match, precision, recall, f1

	@staticmethod
	def split_data(train_names, train_codes, output_length, input_length, pct_train):
		assert pct_train > 0, pct_train
		assert pct_train < 1, pct_train
		assert len(train_names) == len(train_codes), (len(train_names), len(train_codes))

		id_train_name = []
		id_train_code = []
		id_val_name = []
		id_val_code = []
		naming_data = []
		n_tokens = []

		n_samples = len(train_names)
		train_size = int(n_samples * pct_train)

		train_name = train_names[:train_size]
		train_code = train_codes[:train_size]
		val_name = train_names[train_size:]
		val_code = train_codes[train_size:]
		assert len(train_name) == len(train_code), (len(train_name), len(train_code))
		assert len(val_name) == len(val_code), (len(val_name), len(val_code))

		naming_data = DataGenerator(train_name, train_code)
		id_train_name, id_train_code = naming_data.get_data_for_simple_seq2seq(train_name, train_code, output_length, input_length)
		id_val_name, id_val_code = naming_data.get_data_for_simple_seq2seq(val_name, val_code, output_length, input_length)
		n_tokens = naming_data.all_tokens_dictionary.get_n_tokens()

		return id_train_name, id_train_code, id_val_name, id_val_code, naming_data, n_tokens

	@staticmethod
	def exact_match(naming_data, predict_idx, val_name):
		n_correct = 0.0
		correct_idx = []
		end_token = naming_data.all_tokens_dictionary.get_id_or_unk(naming_data.NAME_END)
		assert predict_idx.shape == val_name.shape, (predict_idx.shape, val_name.shape)
		n_samples, n_timesteps = predict_idx.shape
		for i in range(n_samples):
			flag = True
			for j in range(n_timesteps):
				if val_name[i][j] == end_token:
					if not predict_idx[i][j] == end_token:
						flag = False
					break
				if not predict_idx[i][j] == val_name[i][j]:
					flag = False
					break
			if flag == True:
				correct_idx.append(predict_idx[i])
				n_correct += 1
		print ('n_extact_correct = ', n_correct)
		print ('n_samples = ', n_samples)
		print ('correct suggestions:')
		correct_idx = np.array(correct_idx, dtype=np.object)
		correct_suggestions = trainModel.show_names(naming_data, correct_idx)
		for i in range(len(correct_suggestions)):
			print (str(correct_suggestions[i]))
		return n_correct/float(n_samples)

	@staticmethod
	def evaluate_tokens(naming_data, predict_idx, val_name):
		sum_precision = 0.0
		sum_recall = 0.0
		sum_f1 = 0.0

		end_token = naming_data.all_tokens_dictionary.get_id_or_unk(naming_data.NAME_END)
		start_token = naming_data.all_tokens_dictionary.get_id_or_unk(naming_data.NAME_START)

		assert predict_idx.shape == val_name.shape, (predict_idx.shape, val_name.shape)

		n_samples, n_timesteps = predict_idx.shape

		for i in range(n_samples):
			val_start = 0
			val_end = 0
			pre_start = 0
			pre_end = 0
			begin = False
			for j in range(n_timesteps):
				if begin == False:
					if val_name[i][j] == start_token:
						val_start = j+1
						begin = True
				else:
					if val_name[i][j] == end_token:
						val_end = j
						break

			begin = False
			for j in range(n_timesteps):
				if begin == False:
					if predict_idx[i][j] == start_token:
						pre_start = j+1
						begin = True
				else:
					if predict_idx[i][j] == end_token:
						pre_end = j
						break

			name = val_name[i][val_start:val_end]
			pre_name = predict_idx[i][pre_start:pre_end]

			correct_tokens = [v for v in name if v in pre_name]

			if len(name) > 0:
				recall = len(correct_tokens) / float(len(name))
			else:
				recall = 0.0

			if len(pre_name) > 0:
				precision = len(correct_tokens) / float(len(pre_name))
			else:
				precision = 0.0

			if (precision + recall) > 0:
				f1 = 2 * precision * recall / (precision + recall)
			else:
				f1 = 0.0

			sum_precision += precision
			sum_recall += recall
			sum_f1 += f1

		assert n_samples > 0, (n_samples)
		average_precision = sum_precision / float (n_samples)
		average_recall = sum_recall / float (n_samples)
		average_f1 = sum_f1 / float (n_samples)
		return average_precision, average_recall, average_f1

	@staticmethod
	def show_names(naming_data, predict_idx):
		predict_names = []
		if len(predict_idx) == 0:
			print ('nothing correct')
			return predict_names
		else:
			n_samples, n_timesteps = predict_idx.shape
			for i in range(n_samples):
				name = []
				for j in range(n_timesteps):
					name.append(naming_data.all_tokens_dictionary.get_name_for_id(predict_idx[i][j]))
				predict_names.append(name)
			predict_names = np.array(predict_names, dtype=np.object)
			assert predict_names.shape == predict_idx.shape, (predict_names.shape, predict_idx.shape)
			return predict_names

if __name__ == '__main__':

	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)
	KTF.set_session(sess)

	if len(sys.argv) > 2:
		filepath = sys.argv[1]
		model_name = sys.argv[2]


		names, codes, sentences = DataGenerator.get_input_file(filepath)
		assert len(names) == len(codes), (len(names), len(codes))

		#0.7 for train and val, 0.3 for test
		train_size = int(0.7 * len(names))
		idx = np.arange(len(names))
		np.random.shuffle(idx)

		train_names = names[idx[:train_size]]
		train_codes = codes[idx[:train_size]]
		test_names = names[idx[train_size:]]
		test_codes = codes[idx[train_size:]]

		print ('the number of training and validation samples: ', len(train_names))
		print ('the number of testing samples: ', len(test_names))
		model = trainModel(train_names, train_codes, model_name)
		model.grid_search()
