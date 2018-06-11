from __future__ import print_function
import os
import pickle
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

import sys
import keras
import numpy as np
from keras import optimizers
from seq2seq.models import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq
from generate_datasets import DataGenerator

class trainModel(object):
	def __init__(self, train_names, train_codes, model_name, output_folder, hyperparams=None):
		self.train_names = train_names
		self.train_codes = train_codes

		self.hyperparams = hyperparams
		self.model_name = model_name

		self.model = None
		self.parameters = None # if the model has been trained, parameters will not be None.
		self.naming_data = None # the dictionary to decode/encode tokens

		self.output_folder = output_folder
		if not os.path.exists(self.output_folder):
			os.mkdir(self.output_folder)

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
		batch_size = [500]
		input_length = [300]
		depth = [1]
		dropout = [0.3]
		lr = [0.001]
		num_epoch = [50]
		pct_train = 0.9
		peek = [True]
		broadcast_state = [True]
		bidirectional = [False]
		best_f1 = 0
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
													f.flush()
													model, exact_match, precision, recall, f1, naming_data = self.train(self.train_names, self.train_codes, hyperparams, pct_train, t_lr, t_num_epoch)
													f.write('exact match=%f, precision=%f, recall=%f, f1=%f\n\n' % (exact_match, precision, recall, f1))
													f.flush()
													if f1 > best_f1: # use f1 to optimize hyperparams
														best_f1 = f1
														self.model = model
														self.hyperparams = hyperparams
														self.naming_data = naming_data
												elif self.model_name == 'Seq2Seq':
													for t_peek in peek:
														for t_broadcast_state in broadcast_state:
															hyperparams = dict(output_dim=t_output_dim, output_length=t_output_length, hidden_dim=t_hidden_dim, batch_size=t_batch_size, input_length=t_input_length, depth=t_depth, dropout=t_dropout, peek=t_peek, broadcast_state=t_broadcast_state)
															f.write(str(hyperparams) + '\n')
															f.flush()
															model, exact_match, precision, recall, f1, naming_data = self.train(self.train_names, self.train_codes, hyperparams, pct_train, t_lr, t_num_epoch)
															f.flush()
															f.write('exact match=%f, precision=%f, recall=%f, f1=%f\n\n' % (exact_match, precision, recall, f1))
															f.flush()
															if f1 > best_f1: # use f1 to optimize hyperparams
																best_f1 = f1
																self.model = model
																self.hyperparams = hyperparams
																self.naming_data = naming_data
												elif self.model_name == 'AttentionSeq2Seq':
													for t_bidirectional in bidirectional:
														hyperparams = dict(output_dim=t_output_dim, output_length=t_output_length, hidden_dim=t_hidden_dim, batch_size=t_batch_size, input_length=t_input_length, depth=t_depth, dropout=t_dropout, bidirectional=t_bidirectional)
														f.write(str(hyperparams) + '\n')
														f.flush()
														model, exact_match, precision, recall, f1, naming_data = self.train(self.train_names, self.train_codes, hyperparams, pct_train, t_lr, t_num_epoch)
														f.write('exact match=%f, precision=%f, recall=%f, f1=%f\n\n' % (exact_match, precision, recall, f1))
														f.flush()
														if f1 > best_f1: # use f1 to optimize hyperparams
															best_f1 = f1
															self.model = model
															self.hyperparams = hyperparams
															self.naming_data = naming_data
			# best params
			f.write('the best hyperparam is %s' % str(self.hyperparams))
			f.flush()
			print ('the best hyperparam is ', str(self.hyperparams))
			# best model
			self.model.save(self.output_folder + '/best_' + self.model_name + '.h5')

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
                        else:
                            raise TypeError
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
			model.fit(train_code, train_name, epochs=num_epoch)

		print ('predict...')
		predict_probs = model.predict(val_code)
		predict_idx = np.argmax(predict_probs, axis=2)

		print('evaluate...')
		exact_match, _ = trainModel.exact_match(naming_data, predict_idx, val_name)
		precision, recall, f1, _, _ = trainModel.evaluate_tokens(naming_data, predict_idx, val_name)
		return model, exact_match, precision, recall, f1, naming_data

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
		correct_idx = np.array(correct_idx, dtype=np.object)
		correct_suggestions = trainModel.show_names(naming_data, correct_idx)
		return n_correct/float(n_samples), correct_suggestions

	@staticmethod
	def evaluate_tokens(naming_data, predict_idx, val_name, threshold=0.5):
		sum_precision = 0.0
		sum_recall = 0.0
		sum_f1 = 0.0

		end_token = naming_data.all_tokens_dictionary.get_id_or_unk(naming_data.NAME_END)
		start_token = naming_data.all_tokens_dictionary.get_id_or_unk(naming_data.NAME_START)

		assert predict_idx.shape == val_name.shape, (predict_idx.shape, val_name.shape)

		n_samples, n_timesteps = predict_idx.shape
                correct_idx = []
                original_idx = []
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

                        if precision >= threshold:
                            correct_idx.append(predict_idx[i])
                            original_idx.append(val_name[i])

			sum_precision += precision
			sum_recall += recall
			sum_f1 += f1

		assert n_samples > 0, (n_samples)
		average_precision = sum_precision / float (n_samples)
		average_recall = sum_recall / float (n_samples)
		average_f1 = sum_f1 / float (n_samples)
                correct_idx = np.array(correct_idx, dtype = np.object)
                original_idx = np.array(original_idx, dtype = np.object)
                correct_suggestions = trainModel.show_names(naming_data, correct_idx)
                original_names = trainModel.show_names(naming_data, original_idx)
		return average_precision, average_recall, average_f1, correct_suggestions, original_names

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

	@staticmethod
	def test(model, test_names, test_codes):
		id_test_name, id_test_code = model.naming_data.get_data_for_simple_seq2seq(test_names, test_codes, model.hyperparams['output_length'], model.hyperparams['input_length'])
		predict_probs = model.model.predict(id_test_code)
		predict_idx = np.argmax(predict_probs, axis=2)
		exact_match, correct_suggestions = trainModel.exact_match(model.naming_data, predict_idx, id_test_name)
		precision, recall, f1, suggestions, original_names = trainModel.evaluate_tokens(model.naming_data, predict_idx, id_test_name)
		return exact_match, correct_suggestions, precision, recall, f1, suggestions, original_names

if __name__ == '__main__':
	# set run environment
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)
	KTF.set_session(sess)

	# set input_folder_path and model_name
	if len(sys.argv) > 2:
		folder_path = sys.argv[1]
		model_name = sys.argv[2]
	else:
		raise TypeError

	# extract methods in input_folder_path
	with open('test_summarization.txt', 'a+') as f_all:
		print ('folder_path: ', folder_path)
		f_all.write('folder_path: %s\n' % folder_path)
		files = os.listdir(folder_path)
		names = []
		codes = []
		sentences = []
		for file in files:
			if not os.path.isdir(file):
				t_names, t_codes, t_sentences = DataGenerator.get_input_file(folder_path + '/' + file)
				print ('len(t_names)', len(t_names))
				names.extend(t_names)
				codes.extend(t_codes)
				sentences.extend(t_sentences)
				print ('len(names)', len(names))
		names = np.array(names, dtype = np.object)
		codes = np.array(codes, dtype = np.object)
		sentences = np.array(sentences, dtype = np.object)
		assert len(names) == len(codes), (len(names), len(codes))

		# split data into training and testing dataset
		train_size = int(0.7 * len(names))
		idx = np.arange(len(names))
		#np.random.shuffle(idx)
		train_names = names[idx[:train_size]]
		train_codes = codes[idx[:train_size]]
		test_names = names[idx[train_size:]]
		test_codes = codes[idx[train_size:]]
		print ('the number of training and validation samples: ', len(train_names))
		print ('the number of testing samples: ', len(test_names))
		f_all.write('training and validation samples: %d\n' % len(train_names))

		# set output_folder to be the input_folder_path/model_name_Results/
		output_folder = folder_path + '/' + model_name + '_Results'

		# train
		train_model = trainModel(train_names, train_codes, model_name, output_folder)
		train_model.grid_search()

		# test
		print ('test in ', len(test_names), 'samples:')
		f_all.write('test in %d samples:\n' % len(test_names))
		exact_match, correct_suggestions, precision, recall, f1, suggestions, original_names = trainModel.test(train_model, test_names, test_codes)
		print ('exact match = ', exact_match)
		print ('precision = ', precision)
		print ('recall = ', recall)
		print ('f1 = ', f1)
		f_all.write('exact match = %d, ' % exact_match)
		f_all.write('precision = %f, ' % precision)
		f_all.write('recall = %f, ' % recall)
		f_all.write('f1 = %f \n\n' % f1)

		with open(output_folder + '/exact_predictions.txt', 'w') as f:
			for name in correct_suggestions:
				f.write(str(name) + '\n')
		with open(output_folder + '/tokens_predictions.txt', 'w') as f:
			for i, name in enumerate(suggestions):
				f.write('method name: ' + str(original_names[i]) + '\n')
				f.write('prediction: ' + str(name) + '\n')
				f.write('\n')
