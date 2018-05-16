from __future__ import print_function
from seq2seq import *
from generate_datasets import DataGenerator
import sys

class SimpleSeq2SeqLearner(object):
	def __init__(self, hyperparameters):
		self.hyperparameters = hyperparameters
		self.parameters = None	
		self.naming_data = None

	def train(self, input_file, max_name_size=10, max_code_size=300, pct_train=0.65, pct_val=0.05, pct_test=.3, patience=5, max_epochs=1000):
		assert self.parameters == None, ("The model has already been trained!")
		train_data, val_data, test_data, self.naming_data = DataGenerator.get_data_for_simple_seq2seq_with_validation_and_test(input_file, max_name_size, max_code_size, pct_train, pct_val, pct_test)
		

if __name__ == '__main__':
	if len(sys.argv) > 1:
		filepath = sys.argv[1]
		test = DataGenerator(filepath)
		'''	
		with open ('names.txt','w') as f:
			for name in test.names:
				f.write(' '.join(name) + '\n')
		with open ('codes.txt','w') as f:
			for code in test.codes:
				f.write(' '.join(code) + '\n')
		'''
		id_names, id_codes, vocabulary_size = test.get_data_for_basic_seq2seq(300,10)
		print ('id_names.shape: ', id_names.shape)
		print ('id_codes.shape: ', id_codes.shape)
		print ('vocabulary_size: ', vocabulary_size)