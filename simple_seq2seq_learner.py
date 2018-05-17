from __future__ import print_function
from seq2seq.models import SimpleSeq2Seq
from generate_datasets import DataGenerator
import sys

class SimpleSeq2SeqLearner(object):
	def __init__(self, hyperparameters):
		self.hyperparameters = hyperparameters
		self.parameters = None	
		self.naming_data = None
		self.test_name = None
		self.test_code = None
		self.model = None
		self.__check_all_hyperparmeters_exist()

	def train(self, input_file, pct_train=0.65, pct_val=0.05, pct_test=.3, patience=5, max_epochs=1000):
		assert self.parameters == None, ("The model has already been trained!")
		assert "input_length" in self.hyperparameters, ('input_length')
		train_data, val_data, test_data, self.naming_data = DataGenerator.get_data_for_simple_seq2seq_with_validation_and_test(input_file, self.hyperparameters['output_length'], self.hyperparameters['input_length'], pct_train, pct_val, pct_test)
		train_name, train_code = train_data
		val_name, val_code = val_data
		self.test_name, self.test_code = test_data
		self.hyperparameters['n_tokens'] = self.naming_data.all_tokens_dictionary.get_n_tokens()
		model = SimpleSeq2Seq(**self.hyperparameters)
		model.compile(loss='mse', optimizer='rmsprop')
		self.model = model
		
		
		model.fit(train_code, train_name)
		print ('predict...')
		print (model.predict(val_name))
		print ('evaluate...')
		print (model.evaluate(val_code,val_name))
		
	def __check_all_hyperparmeters_exist(self):
		all_params = ["output_dim", "output_length"]

		for param in all_params:
			assert param in self.hyperparameters, param
			
if __name__ == '__main__':
	if len(sys.argv) > 1:
		filepath = sys.argv[1]
		'''	
		with open ('names.txt','w') as f:
			for name in test.names:
				f.write(' '.join(name) + '\n')
		with open ('codes.txt','w') as f:
			for code in test.codes:
				f.write(' '.join(code) + '\n')
		'''
		params = dict(output_dim=128,output_length=8, input_length=300, is_embedding=False)
		model = SimpleSeq2SeqLearner(params)
		model.train(filepath)