import h5py
from hyp_data_reader import HypDataReader
import numpy as np 
"""
Author: Jeff Lai, Jesus 
JHU hltcoe 2017 

Read mfcc data from 'mfcc_cmvn.h5' to train, validate, test a lstm. Using h5py library and Jesus' hyp_data_reader.
"""

mfcc_h5 = HypDataReader('/export/b14/jlai/scale/vad/kaldi/mfcc_cmvn.h5') #an instance of hyp_data_reader

def get_utt():
	"""	
	-returns a list of .wav file names 
	"""
	return mfcc_h5.get_datasets()
	
def get_frame():
	"""
	-returns a dictionary with the .wav file name as key and its mfcc (frame*dimension) matrix as value 
	"""
	all_frames = {}
	keys = get_utt()
	for i, matrix_id in enumerate(mfcc_h5.read(keys)):	
		matrix = np.asarray(matrix_id)	
		all_frames[keys[i]] = matrix 
	
	return all_frames 

def combine_mfcc_id():
	"""
	Given all_frames (a dict with key as wav file name and value as mfcc matrix) and utt2spk (tuples of wav file name and 3 ids). Convert all_frames to a list and create a list of utt2spk's id in the order of all_frames's keys.  
	
	Secondly, either ensure the mfcc is 10-second with dim (1003, 20) or divide the 10-second mfcc frame-by-frame.  

	-returns numpy array of mfcc, vad (same order and number) and their corresponding mask 
	"""
	mfcc, vad = [], []
	all_frames = get_frame()
	#read file utt2spk, sorted and store in an array vad
	with open('utt2spk') as f:
		content = f.readlines()
	content = [x.strip() for x in content]

	for key in all_frames.keys():
		mfcc.append(all_frames[key]) #append mfcc to a list 
		for tuples in content: 
			wav = tuples.split(' ')[0]
			ID = tuples.split(' ')[1]
			if wav == key: #same wav file name
				if ID == 'speech': 
			 		vad.append([1, 0, 0])
				elif ID == 'noise':
					vad.append([0, 1, 0])
				else:
					vad.append([0, 0, 1])

	train_mfcc = mfcc[:len(mfcc)/2]
	val_mfcc = mfcc[len(mfcc)/2:len(mfcc)*3/4]
	test_mfcc = mfcc[len(mfcc)*3/4:]
	train_vad = vad[:len(vad)/2]
	val_vad = vad[len(vad)/2:len(vad)*3/4]
	test_vad = vad[len(vad)*3/4:]
	
	################### Train and Validation data are 2D array ######
	new_train_mfcc, new_train_vad = [], []
	for count, matrix in enumerate(train_mfcc): 
		for row in matrix:
			new_train_mfcc.append(row) 
			new_train_vad.append(train_vad[count])

	new_val_mfcc, new_val_vad = [], []
	for count, matrix in enumerate(val_mfcc):
		for row in matrix:
			new_val_mfcc.append(row)
			new_val_vad.append(val_vad[count])
	#################### Test data is a 3D array #####################
	new_test_mfcc, new_test_vad = [], []
	for count, matrix in enumerate(test_mfcc): #ensure dimension of the data (10-sec)
		if np.array(matrix).shape == (1003,20): 
			new_test_mfcc.append(matrix)
			new_test_vad.append([vad[count]])
	
	np_train_mfcc = np.array(new_train_mfcc) 
	np_val_mfcc = np.array(new_val_mfcc)
	np_test_mfcc = np.array(new_test_mfcc)
	np_train_vad = np.array(new_train_vad)
	np_val_vad = np.array(new_val_vad)
	np_test_vad = np.array(new_test_vad)

	print("The shape of the train mfcc numpy array is %s" % (np_train_mfcc.shape,))
	print("The shape of the validation mfcc numpy array is %s" % (np_val_mfcc.shape,))
	print("The shape of the test mfcc numpy array is %s" % (np_test_mfcc.shape,))
	print("The shape of the train vad numpy array is %s" % (np_train_vad.shape,))
	print("The shape of the validation  vad numpy array is %s" % (np_val_vad.shape,))
	print("The shape of the test vad numpy array is %s" % (np_test_vad.shape,))

	return (np_train_mfcc,np_val_mfcc,np_test_mfcc,np_train_vad,np_val_vad,np_test_vad)
        
if __name__ == '__main__':
	combine_mfcc_id()
