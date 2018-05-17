from __future__ import print_function 
import json
import yaml
import numpy as np
import re
import sys
import pickle
from itertools import chain
from feature_dict import FeatureDictionary

class DataGenerator(object):
	METHOD_START = "%M_START%" 
	METHOD_END = "%M_END%"
	SENTENCE_START = "%S_START%" 
	SENTENCE_END = "%S_END%"
	NAME_START = "%N_START%"
	NAME_END = "%N_END%"

	def __init__(self, names, codes): 
		'''
		mode:
			String: convert a method into a string
			Sentences: convert a method into sentences
		'''
		self.names = names
		self.codes = codes

		#not used for now
		self.name_dictionary = FeatureDictionary()
		self.name_dictionary.add_or_get_id(self.NAME_START)
		self.name_dictionary.add_or_get_id(self.NAME_END)
		self.name_dictionary.get_feature_dictionary_for(chain.from_iterable(self.names), 2)
		
		#used to map both names and codes
		self.all_tokens_dictionary = FeatureDictionary()
		self.all_tokens_dictionary.add_or_get_id(self.METHOD_START)
		self.all_tokens_dictionary.add_or_get_id(self.METHOD_END)
		self.all_tokens_dictionary.add_or_get_id(self.NAME_START)
		self.all_tokens_dictionary.add_or_get_id(self.NAME_END)
		self.all_tokens_dictionary.get_feature_dictionary_for(chain.from_iterable([chain.from_iterable(self.codes), chain.from_iterable(self.names)]), 5)

	@staticmethod
	def split_str(string):
		"""
		Tokenization/string cleaning for dataset
		Every dataset is lower cased except
		"""
		string = re.sub(r"\\", "", string)    
		string = re.sub(r"\'", "", string)    
		string = re.sub(r"\"", "", string) 
		a = re.split(r"([^a-zA-Z]+)",string)
		result = []
		for items in a:
			items = re.sub(r"\s", "", items)
			if len(items) == 0:
				pass
			elif len(items) == 1:
				result.append(items.lower())
			elif items.upper() == items:
					result.append(items.lower())
			else:
				for idx,c in enumerate(items): 
					if idx == 0:
						s = c 
					else:
						if c.isupper() == True:
								result.append(s.lower())
								s = c
						else:
							s += c
						if idx == (len(items) -1):
							result.append(s.lower())
		return (result)

	@staticmethod
	def get_input_file(input_file):
		names = []
		codes = []
		sentences = []
		with open (input_file,'r') as f:
			print ('load data...')
			unicode_data = json.load(f) # read files
			str_data = json.dumps(unicode_data) # convert into str
			all_methods = yaml.safe_load(str_data) # safely load (remove 'u')
			for method in all_methods:
				m_name = method['methodName'][0]
				methodBody = method['methodBody']
				if len(m_name) == 0 or len(methodBody) == 0:
					continue
				strBody = []
				sentBody = []
				for sentence in methodBody:
					if sentence == 'METHOD_START' or sentence == 'METHOD_END':
						continue
					if len(sentence) == 0:
						continue 
					tokens = DataGenerator.split_str(sentence)
					strBody += tokens
					sentBody.append([DataGenerator.SENTENCE_START] + tokens + [DataGenerator.SENTENCE_END])
				# filter methods
				sentences.append([DataGenerator.METHOD_START] + sentBody + [DataGenerator.METHOD_END])
				strBody = [DataGenerator.METHOD_START] + strBody + [DataGenerator.METHOD_END]
				codes.append(strBody)
				names.append([DataGenerator.NAME_START] + DataGenerator.split_str(method['methodName'][0]) + [DataGenerator.NAME_END])
		names = np.array(names, dtype = np.object)		
		codes = np.array(codes, dtype = np.object)		
		sentences = np.array(sentences, dtype = np.object)		
		return names,codes,sentences

	def get_data_for_simple_seq2seq(self, names, codes, max_name_size, max_code_size):
		assert len(names) == len(codes), (len(names), len(codes))
		if len(names) == 0:
			return None, None
		id_names = []
		id_codes = []
		padding = [self.all_tokens_dictionary.get_id_or_unk(self.all_tokens_dictionary.get_none())] # padding = 0
		print ('padding integer = ', padding)
		with open('err_names.txt','w') as f, open('err_codes.txt','w') as f2:
			for i, name in enumerate(names):
				t_name = []
				t_codes = []
				
				for j in range(len(name)):
					t_name.append(self.all_tokens_dictionary.get_id_or_unk(name[j]))
				
				for j in range(len(codes[i])):
					t_codes.append(self.all_tokens_dictionary.get_id_or_unk(codes[i][j]))
				
				if len(t_codes) <= max_code_size:
					t_codes += padding * (max_code_size- len(t_codes))
				else:
					f2.write('len(t_codes) == %d \n' % len(t_codes))
					continue
				assert len(t_codes) == max_code_size, (len(t_codes), max_code_size)
				
				if len(t_name) <= max_name_size:
					t_name += padding * (max_name_size- len(t_name))
				else:
					f.write('len(t_name) == %d \n' % len(t_name))
					continue
				assert len(t_name) == max_name_size, (len(t_name),max_name_size)

				id_names.append(t_name)
				id_codes.append(t_codes)
		assert len(id_names) == len(id_codes), (len(id_names), len(id_codes))	
		id_names = np.array(id_names,dtype = np.int32)
		id_codes = np.array(id_codes,dtype = np.int32)
		return id_names, id_codes

	@staticmethod
	def get_data_for_simple_seq2seq_with_validation_and_test(input_file, max_name_size, max_code_size, pct_train, pct_val, pct_test):
		assert pct_train > 0
		assert pct_val > 0
		assert pct_test >= 0
		assert (pct_train + pct_val + pct_test) == 1.0
		all_names, all_codes, all_sentences = DataGenerator.get_input_file(input_file)	
		assert len(all_names) == len(all_codes), (len(all_names), len(all_codes))
		train_and_val = int((pct_train + pct_val) * len(all_names))
		train_size = int(pct_train * len(all_names))
		idxs = np.arange(len(all_names))
		np.random.shuffle(idxs)
		naming = DataGenerator(all_names[idxs[:train_size]], all_codes[idxs[:train_size]])
		return naming.get_data_for_simple_seq2seq(all_names[idxs[:train_size]], all_codes[idxs[:train_size]], max_name_size, max_code_size),\
			naming.get_data_for_simple_seq2seq(all_names[idxs[train_size:train_and_val]], all_codes[idxs[train_size:train_and_val]], max_name_size, max_code_size),\
			naming.get_data_for_simple_seq2seq(all_names[idxs[train_and_val:]], all_codes[idxs[train_and_val:]], max_name_size, max_code_size),\
			naming

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
		DataGenerator.get_data_for_simple_seq2seq_with_validation_and_test(filepath, 10,300, 0.65, 0.05, 0.3)