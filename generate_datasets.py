from __future__ import print_function 
import json
import yaml
import numpy as np
import re
import sys
import pickle
from itertools import chain
from feature_dict import FeatureDictionary

class dataGenerator(object):
	METHOD_START = "%M_START%" 
	METHOD_END = "%M_END%"
	SENTENCE_START = "%S_START%" 
	SENTENCE_END = "%S_END%"
	NAME_START = "%N_START%"
	NAME_END = "%N_END%"

	def __init__(self,filepath): 
		'''
		mode:
			String: convert a method into a string
			Sentences: convert a method into sentences
		'''
		self.filepath = filepath

		self.names, self.codes, self.sentences = self.get_input_file()
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

	def split_str(self,string):
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

	def get_input_file(self):
		names = []
		codes = []
		sentences = []
		with open (self.filepath,'r') as f:
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
					tokens = self.split_str(sentence)
					strBody += tokens
					sentBody.append([self.SENTENCE_START] + tokens + [self.SENTENCE_END])
				# filter methods
				sentences.append([self.METHOD_START] + sentBody + [self.METHOD_END])
				strBody = [self.METHOD_START] + strBody + [self.METHOD_END]
				codes.append(strBody)
				names.append([self.NAME_START] + self.split_str(method['methodName'][0]) + [self.NAME_END])
		return names,codes,sentences

	def get_data_for_basic_seq2seq(self,max_code_tokens, max_name_tokens):
		assert len(self.names) == len(self.codes), (len(self.names), len(self.codes))
		id_names = []
		id_codes = []
		padding = [self.all_tokens_dictionary.get_id_or_unk(self.all_tokens_dictionary.get_none())] # padding = 0
		with open('err_names.txt','w') as f, open('err_codes.txt','w') as f2:
			for i, name in enumerate(self.names):
				t_name = []
				t_codes = []
				
				for j in range(len(name)):
					t_name.append(self.all_tokens_dictionary.get_id_or_unk(name[j]))
				
				for j in range(len(self.codes[i])):
					t_codes.append(self.all_tokens_dictionary.get_id_or_unk(self.codes[i][j]))
				
				if len(t_codes) <= max_code_tokens:
					t_codes += padding * (max_code_tokens- len(t_codes))
				else:
					f2.write('len(t_codes) == %d \n' % len(t_codes))
					continue
				assert len(t_codes) == max_code_tokens, (len(t_codes),max_code_tokens)
				
				if len(t_name) <= max_name_tokens:
					t_name += padding * (max_name_tokens- len(t_name))
				else:
					f.write('len(t_name) == %d \n' % len(t_name))
					continue
				assert len(t_name) == max_name_tokens, (len(t_name),max_name_tokens)

				id_names.append(t_name)
				id_codes.append(t_codes)
		assert len(id_names) == len(id_codes), (len(id_names), len(id_codes))	
		id_names = np.array(id_names,dtype = np.int32)
		id_codes = np.array(id_codes,dtype = np.int32)
		print ('id_names.shape: ', id_names.shape)
		print ('id_codes.shape: ', id_codes.shape)
		vocabulary_size = self.all_tokens_dictionary.get_n_tokens() 
		print ('vocabulary_size (including none, unknown..): ', vocabulary_size)
		return id_names, id_codes, vocabulary_size

if __name__ == '__main__':
	if len(sys.argv) > 1:
		filepath = sys.argv[1]
		test = dataGenerator(filepath)
		'''	
		with open ('names.txt','w') as f:
			for name in test.names:
				f.write(' '.join(name) + '\n')
		with open ('codes.txt','w') as f:
			for code in test.codes:
				f.write(' '.join(code) + '\n')
		'''
		test.get_data_for_basic_seq2seq(300,10)