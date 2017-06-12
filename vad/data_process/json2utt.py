import json
"""
Author: Jeff Lai
JHU HLTCOE 2107 

We intend to classify the google audioset into 3 categories: speech, speech + noise, and noise (human sounds other than speech). Classifications are executed according to the ontology.json file provided from Google. 

json file is a way to organize and trasmit data in a tree-like structure. In ontology.json, every data is originated from the parent catergoy "Human sounds". This script should output a textfile in the same format as kaldi's utt2spk, with tuples of .wav file name and the three categories. 

/export/b14/jlai/scale/vad/google_audioset/balanced_train/balanced_train_segments.csv --> contains .wav file name v.s. id 

/export/b14/jlai/scale/vad/google_audioset/ontology.json --> contains id to google labeled categories   

Procedure:
1. Create a dictionary with values as speech or noise and keys as ids from ontology.json
2. Output tuples of .wav file name and (noise, speech, noisy_speech) label to utt2spk 
"""

jsonD = {} #id v.s. (speech, noise, noisy_speech) label 
raw_list = [] #list of dictionaries of raw json data
raw_dic = {} #dictionary of dictionaries of raw json data
wav_label = {} #.wav file name v.s. speech, noise, noisy_speech) label 

def create_D(file_ID):
    """
    Create a dictionary, jsonD, with values as the three desired categoreis and keys as ids from ontology.json

    -argument: json_file_ID
    -return dictionary jsonD 
    """
    with open(file_ID) as json_data: 
    	#import jason data 
	    raw_list = json.load(json_data)
	
	list2dic(raw_list)

	speech_id_search(raw_dic['/m/09x0r']) #speech parent 
	for i in raw_dic.keys():
		if i not in jsonD.keys():
			noise_id_search(raw_dic[i]) #noise parent 

	# Test case 
	assert(len(jsonD.keys()) == len(raw_list))
	print(jsonD)

def list2dic(list):
	"""
	Helper method for create_D(). 

	-argument: entire dataset in list
	-return: entire datatset in dictionary with key equals to their respective ids
	"""
	for i in list: 
		raw_dic[i['id']] = i		

def speech_id_search(parent_data):
	"""
	Helper method for create_D(). Recursively search the entire ontology.json for speech data
	
	-argument: parent_data, a dictionary 
	"""
	jsonD[parent_data['id']] = 'speech' 

	if parent_data['child_ids'] == []:
		# no child_ids, add to jsonD
		jsonD[parent_data['id']] = 'speech'
		return 
	
	for i in xrange(len(parent_data['child_ids'])):
		speech_id_search(raw_dic[parent_data['child_ids'][i]])

def noise_id_search(parent_data):
	"""
	Helper method for create_D(). Recursively search the entire ontology.json for noise data
	
	-argument: parent_data, a dictionary 
	"""
	if parent_data['name'] == 'Speech': 
		#avoid storing speech data 
		return 
	jsonD[parent_data['id']] = 'noise' 

	if parent_data['child_ids'] == []:
		# no child_ids, add to jsonD
		jsonD[parent_data['id']] = 'noise'
		return 
	
	for i in xrange(len(parent_data['child_ids'])):
		noise_id_search(raw_dic[parent_data['child_ids'][i]])

def log_utt2spk(file_ID):
	"""
	Output tuples of .wav file name and (noise, speech, noisy_speech) label to text file utt2spk 

	-argument: contains .wav file name v.s. id
	"""
	load_wav(file_ID)
	with open("utt2spk", "w") as f:
		for key in wav_label.keys():
			f.write("%s %s\n" % (key, wav_label[key]))

def load_wav(file_ID):
	"""
	Helper method for utt2spk(). Create a dictionary with key as .wav file name and value as (noise, speech, noisy_speech) label 
	
	-argument: contains .wav file name v.s. id 
	-return: dictionary 
	"""
    with open(file_ID, 'r') as f:
    	#store balanced_train_segments.csv into a list 
		wav_list = [line.rstrip('\n') for line in f] 
        
	for i in xrange(len(wav_list)):
		key = wav_list[i].split('"')[0].split(',')[0]
		value = wav_list[i].split('"')[1]
		wav_label[key] = find_label(value)

	print(wav_label)

def find_label(str_id):
	"""
	Helper method for load_wav(). Find the label for a given id(s).

	-argument: id in string 
	-return: one of the three (speech, noise, noisy_speech) label. 

	"""
	list_id = str_id.split(',')
	if len(list_id) == 1: #only one id
		return jsonD[list_id[0]]
	else: #more than one id
		if all(jsonD[list_id[0]] == jsonD[item] for item in list_id):
			#all values of the ids are the same
			return jsonD[list_id[0]]
		else: #not all values of the ids are the same 
			return 'noisy_speech'

if __name__ == '__main__':
	create_D('/export/b14/jlai/scale/vad/google_audioset/ontology.json')
	log_utt2spk('/export/b14/jlai/scale/vad/google_audioset/balanced_train/balanced_train_segments.csv')