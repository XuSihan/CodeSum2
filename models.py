from __future__ import print_function
import sys
import numpy as np
from keras import optimizers
from seq2seq.models import SimpleSeq2Seq
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
		
		# self._check_all_hyperparmeters_exist()
		# for seq2seq models
		# self.n_tokens = self.naming_data.all_tokens_dictionary.get_n_tokens()


	def _check_all_hyperparmeters_exist(self):
		if self.model_name == 'SimpleSeq2Seq':
			'''
			output_dim: for the last layer of decoder
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
		output_dim = [200, 500]
		output_length = [5, 8]
		hidden_dim = [200, 500]
		batch_size = [100, 200]
		input_length = [300, 500]
		depth = [1, 3]
		dropout = [0.1, 0.3, 0.5]

		lr = [0.0001, 0.0005, 0.001, 0.005] 
		num_epoch = [20, 100, 500]

		k_fold = 10
		best_score = 0
		best_hyparams = []
		assert len(self.train_names) == len(self.train_codes), (len(self.train_names), len(self.train_codes))
		# grid search
		with open('grid_search_results.txt', 'w') as f:
			for __, t_output_dim in enumerate(output_dim):
				for __, t_output_length in enumerate(output_length):
					for __, t_hidden_dim in enumerate(hidden_dim):
						for __, t_batch_size in enumerate(batch_size):
							for __, t_input_length in enumerate(input_length):
								for __, t_depth in enumerate(depth):
									for __, t_dropout in enumerate(dropout):
										for __, t_lr in enumerate(lr):
											for __, t_num_epoch in enumerate(num_epoch):
												t_params = dict(output_dim=t_output_dim, output_length=t_output_length, hidden_dim=t_hidden_dim, batch_size=t_batch_size, input_length=t_input_length, depth=t_depth, dropout=t_dropout, lr=t_lr, num_epoch=t_num_epoch)
												f.write(str(t_params) + '\n')
												accuracy_sum = 0.0
												# cross validation												
												id_train_name, id_train_code, id_val_name, id_val_code, naming_data, n_tokens = trainModel.cross_validation_data(self.train_names, self.train_codes, t_output_length, t_input_length, k_fold)
												assert len(id_train_name) == k_fold, (len(id_train_name), k_fold)
												for i in range(k_fold):
													t_train_name = id_train_name[i]
													t_train_code = id_train_code[i]
													t_val_name = id_val_name[i]
													t_val_code = id_val_code[i]
													t_naming_data = naming_data[i]
													t_n_tokens = n_tokens[i]
													print ('k=', i, ', n_tokens=', t_n_tokens)

													# set hyperparameters
													if self.model_name == 'SimpleSeq2Seq':
														self.hyperparams = dict(output_dim=t_output_dim, output_length=t_output_length, hidden_dim=t_hidden_dim, \
																			batch_size=t_batch_size, input_length=t_input_length, is_embedding=False, \
																			n_tokens=t_n_tokens, depth=t_depth,	dropout=t_dropout)

													t_train_name = trainModel.one_hot_name(t_train_name, t_n_tokens)

													t_model = SimpleSeq2Seq(**self.hyperparams)

													t_my_adam = optimizers.Adam(lr=t_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
													t_model.compile(optimizer=t_my_adam, loss='categorical_crossentropy')

													print ('fit...')		
													t_model.fit(t_train_code, t_train_name, epochs = t_num_epoch)

													print ('predict...')
													t_predict_probs = t_model.predict(t_val_code)
													print ('predict_probs.shape: ', t_predict_probs.shape)
													
													t_predict_idx = np.argmax(t_predict_probs, axis=2)
													print('predict_idx.shape: ', t_predict_idx.shape)

													print('Exact match evaluate...')
													t_exact_match_accuracy = trainModel.exact_match(t_naming_data, t_predict_idx, t_val_name)
													print('exact_match_accuracy: ', t_exact_match_accuracy)
													f.write('%d, ' % t_exact_match_accuracy)
													accuracy_sum += t_exact_match_accuracy

													'''
													suggestions = trainModel.show_names(t_naming_data, t_predict_idx)
													original_names = trainModel.show_names(t_naming_data, t_val_name)

													with open('suggestions.txt', 'w') as f:
														for i in range(len(suggestions)):
															f.write('original name: ' + str(original_names[i]) + '\n')
															f.write('suggestions: ' + str(suggestions[i]) + '\n')
															f.write('\n')
													'''
												average_accuracy = accuracy_sum/k_fold
												f.write('\naverage accuracy: %d \n\n' % average_accuracy)
												if average_accuracy > best_score:
													best_score = average_accuracy
													best_hyparams = t_params
		return best_score, best_hyparams



	@staticmethod
	def cross_validation_data(train_names, train_codes, output_length, input_length, k_fold=5):

		id_train_name = []
		id_train_code = []
		id_val_name = []
		id_val_code = []
		naming_data = []
		n_tokens = []
	
		n_samples = len(train_names)
		per_size = int(n_samples/k_fold)
		print ('per_size: ', per_size)
	
		#cross validation
		for i in range(k_fold - 1):
			val_name = train_names[i*per_size:(i+1)*per_size]
			val_code = train_codes[i*per_size:(i+1)*per_size]
			train_name = np.delete(train_names, range(i*per_size,(i+1)*per_size), 0)
			train_code = np.delete(train_codes, range(i*per_size,(i+1)*per_size), 0)
			assert len(train_name) == len(train_code), (len(train_name), len(train_code))
			assert len(val_name) == len(val_code), (len(val_name), len(val_code))
			print ('the number of training samples: ', len(train_name))
			print ('the number of validation samples: ', len(val_name))
			t_naming_data = DataGenerator(train_name, train_code)												
			t_id_train_name, t_id_train_code = t_naming_data.get_data_for_simple_seq2seq(train_name, train_code, output_length, input_length)
			t_id_val_name, t_id_val_code = t_naming_data.get_data_for_simple_seq2seq(val_name, val_code, output_length, input_length)
			t_n_tokens = t_naming_data.all_tokens_dictionary.get_n_tokens()
			
			id_train_name.append(t_id_train_name)
			id_train_code.append(t_id_train_code)
			id_val_name.append(t_id_val_name)
			id_val_code.append(t_id_val_code)
			naming_data.append(t_naming_data)
			n_tokens.append(t_n_tokens)

		val_name = train_names[(k_fold-1)*per_size:]
		val_code = train_codes[(k_fold-1)*per_size:]
		train_name = np.delete(train_names, range(i*per_size,(i+1)*per_size), 0)
		train_code = np.delete(train_codes, range(i*per_size,(i+1)*per_size), 0)

		assert len(train_name) == len(train_code), (len(train_name), len(train_code))
		assert len(val_name) == len(val_code), (len(val_name), len(val_code))
		print ('the number of training samples: ', len(train_name))
		print ('the number of validation samples: ', len(val_name))

		t_naming_data = DataGenerator(train_name, train_code)												
		t_id_train_name, t_id_train_code = t_naming_data.get_data_for_simple_seq2seq(train_name, train_code, output_length, input_length)
		t_id_val_name, t_id_val_code = t_naming_data.get_data_for_simple_seq2seq(val_name, val_code, output_length, input_length)
		t_n_tokens = t_naming_data.all_tokens_dictionary.get_n_tokens()

		id_train_name.append(t_id_train_name)
		id_train_code.append(t_id_train_code)
		id_val_name.append(t_id_val_name)
		id_val_code.append(t_id_val_code)
		naming_data.append(t_naming_data)
		n_tokens.append(t_n_tokens)

		return id_train_name, id_train_code, id_val_name, id_val_code, naming_data, n_tokens

	@staticmethod
	def exact_match(naming_data, predict_idx, val_name):
		n_correct = 0
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
		print ('n_correct = ', n_correct)
		print ('n_samples = ', n_samples)
		print ('correct suggestions:')
		correct_idx = np.array(correct_idx, dtype=np.object)
		correct_suggestions = trainModel.show_names(naming_data, correct_idx)
		for i in range(len(correct_suggestions)):
			print (str(correct_suggestions[i]))
		return n_correct/float(n_samples)

	@staticmethod
	def show_names(naming_data, predict_idx):
		n_samples, n_timesteps = predict_idx.shape
		predict_names = []
		for i in range(n_samples):
			name = []
			for j in range(n_timesteps):
				name.append(naming_data.all_tokens_dictionary.get_name_for_id(predict_idx[i][j]))
			predict_names.append(name)
		predict_names = np.array(predict_names, dtype=np.object)
		assert predict_names.shape == predict_idx.shape, (predict_names.shape, predict_idx.shape)
		return predict_names

if __name__ == '__main__':
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
