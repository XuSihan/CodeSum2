from __future__ import print_function 
import json
import yaml
import numpy as np
import re
import sys
import pickle
from itertools import chain
import FeatureDictionary

MAX_NAME = 5
MAX_TOKENS = 200
MAX_SENTENCES = 20
MAX_LENGTH = 20

class dataGenerator(object):
	METHOD_START = "%M_START%" 
	METHOD_END = "%M_END%"
	SENTENCE_START = "%S_START%" 
	SENTENCE_END = "%S_END%"
	NONE = "%NONE%" 
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

		self.name_dictionary = FeatureDictionary.get_feature_dictionary_for(chain.from_iterable(self.names), 2)
        self.name_dictionary.add_or_get_id(NONE)
        self.name_dictionary.add_or_get_id(self.NAME_START)
        self.name_dictionary.add_or_get_id(self.NAME_END)

        self.all_tokens_dictionary = FeatureDictionary.get_feature_dictionary_for(chain.from_iterable(
            [chain.from_iterable(code), chain.from_iterable(self.names)]), 5)
        self.all_tokens_dictionary.add_or_get_id(self.NONE)
        self.all_tokens_dictionary.add_or_get_id(self.METHOD_START)
        self.all_tokens_dictionary.add_or_get_id(self.METHOD_END)

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
			n_remove_methods = 0
			n_remove_names = 0
			for method in all_methods:
				m_name = method['methodName'][0]
				methodBody = method['methodBody']
				if len(m_name) == 0 or len(methodBody) == 0:
					continue
				# filter methods
				if len(self.split_str(m_name)) > MAX_NAME:
					n_remove_names += 1
					print('The length of name is %d' % len(self.split_str(m_name)))
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
				if len(strBody) > MAX_TOKENS:
					n_remove_methods += 1
					print('The number of tokens is %d' % len(strBody))
					continue
				sentences.append([self.METHOD_START] + sentBody + [self.METHOD_END])
				strBody.append([self.METHOD_START] + strBody + [self.METHOD_END])
				names.append([self.NAME_START] + self.split_str(method['methodName'][0]) + self.NAME_END)
		return names,codes,sentences

	def get_data_for_basic_seq2seq(self):
		assert len(self.names) == len(self.codes), (len(self.names), len(self.codes))
if __name__ == '__main__':
	if len(sys.argv) > 1:
		filepath = sys.argv[1]
		test = dataGenerator(filepath)
		with open ('names.txt','w') as f:
			for name in test.names:
				f.write(' '.join(name) + '\n')
		with open ('codes.txt','w') as f:
			for code in test.codes:
				f.write(' '.join(code) + '\n')


