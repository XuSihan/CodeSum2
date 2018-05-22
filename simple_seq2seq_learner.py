from __future__ import print_function
from keras import optimizers
from seq2seq.models import SimpleSeq2Seq
from generate_datasets import DataGenerator
import sys
import numpy as np

class SimpleSeq2SeqLearner(object):
	def __init__(self, hyperparameters):
		self.hyperparameters = hyperparameters
		self.parameters = None	
		self.naming_data = None
		self.test_name = None
		self.test_code = None
		self.model = None
		self._check_all_hyperparmeters_exist()

	def train(self, input_file, pct_train=0.65, pct_val=0.05, pct_test=0.3):
		assert self.parameters == None, ("The model has already been trained!")
		assert "input_length" in self.hyperparameters, ('input_length')

		train_data, val_data, test_data, self.naming_data = DataGenerator.get_data_for_simple_seq2seq_with_validation_and_test(input_file, self.hyperparameters['output_length'], self.hyperparameters['input_length'], pct_train, pct_val, pct_test)
		train_name, train_code = train_data
		val_name, val_code = val_data
		self.test_name, self.test_code = test_data
		self.hyperparameters['n_tokens'] = self.naming_data.all_tokens_dictionary.get_n_tokens()
		
		def one_hot_name(names, max_name_size, name_dim):
			X = np.zeros((len(names), max_name_size, name_dim))
			for i, name in enumerate(names):
				for j, token in enumerate(name):
					X[i, j, token] = 1.0
			return X

		train_name = one_hot_name(train_name, self.hyperparameters['output_length'], self.hyperparameters['n_tokens'])
		print ('n_tokens: ', self.hyperparameters['n_tokens'])
		with open('train_name.txt', 'w') as f:
			f.write(str(train_name))

		model = SimpleSeq2Seq(**self.hyperparameters)
		my_rms = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-06)
		my_adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
		model.compile(optimizer=my_adam, loss='categorical_crossentropy')
		self.model = model

		print ('fit...')		
		model.fit(train_code, train_name, epochs = 20)

		print ('predict...')
		predict_probs = model.predict(val_code)
		print ('predict_probs.shape: ', predict_probs.shape)
		
		predict_idx = np.argmax(predict_probs, axis=2)
		print('predict_idx.shape: ', predict_idx.shape)

		print('Exact match evaluate...')
		exact_match_accuracy = self.exact_match(predict_idx, val_name)
		print('exact_match_accuracy: ', exact_match_accuracy)

		suggestions = self.show_names(predict_idx)
		original_names = self.show_names(val_name)
		with open('suggestions.txt', 'w') as f:
			for i in range(len(suggestions)):
				f.write('original name: ' + str(original_names[i]) + '\n')
				f.write('suggestions: ' + str(suggestions[i]) + '\n')
				f.write('\n')

	def exact_match(self, predict_idx, val_name):
		n_correct = 0
		correct_idx = []
		end_token = self.naming_data.all_tokens_dictionary.get_id_or_unk(self.naming_data.NAME_END)
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
		correct_suggestions = self.show_names(correct_idx)
		for i in range(len(correct_suggestions)):
			print (str(correct_suggestions[i]))
		return n_correct/float(n_samples)


	def show_names(self, predict_idx):
		n_samples, n_timesteps = predict_idx.shape
		predict_names = []
		for i in range(n_samples):
			name = []
			for j in range(n_timesteps):
				name.append(self.naming_data.all_tokens_dictionary.get_name_for_id(predict_idx[i][j]))
			predict_names.append(name)
		predict_names = np.array(predict_names, dtype=np.object)
		assert predict_names.shape == predict_idx.shape, (predict_names.shape, predict_idx.shape)
		return predict_names

	def _check_all_hyperparmeters_exist(self):
		all_params = ["output_dim", "output_length"]

		for param in all_params:
			assert param in self.hyperparameters, param
			
