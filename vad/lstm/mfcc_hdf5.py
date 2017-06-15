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
	
	#print(all_frames[keys[0]])
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
	
	mod = 2
	mfcc_new, vad_new, count = [], [], 0
	#################### Modification ################################
	if mod==1:
		#Modification 1: Ensure the dimension of the data (10-sec)
		for matrix in mfcc:
			if np.array(matrix).shape == (1003, 20):
				mfcc_new.append(matrix)
				vad_new.append(vad[count])
				count += 1
		#test 
		assert len(mfcc_new) == len(vad_new), "Check length of mfcc_new and vad_new"
		print("Original information is %d" % len(vad))
		print("Current information is %d" % count)
		print("Loss of information is %d" % (len(vad)-count))
	elif mod == 2: 
		#Modification 2: Divide mfcc, and their respective vad, into frames  
		for count, matrix in enumerate(mfcc): 
			for row in matrix:
				mfcc_new.append([row])
				vad_new.append(vad[count])
		#test 
		assert len(mfcc_new) == len(vad_new), "Check length of mfcc_new and vad_new"
	###################################################################
	np_mfcc = np.array(mfcc_new)
	np_vad = np.array(vad_new)
	print("The shape of the mfcc numpy array is %s" % (np_mfcc.shape,))
	print("The shape of the vad numpy array is %s" % (np_vad.shape,))
	return np_mfcc, np_vad

def split_frame_re():
	"""
	Splits the data into training, validation and testing (50%, 25%, 25%). 
	
	-returns a training data mfcc vad, validation data mfcc vad, testing data mfcc vad
	"""
	mfcc, vad = combine_mfcc_id()
	train_mfcc = mfcc[:len(mfcc)/2]
	val_mfcc = mfcc[len(mfcc)/2:len(mfcc)*3/4]
	test_mfcc = mfcc[len(mfcc)*3/4:]
	train_vad = vad[:len(vad)/2]
	val_vad = vad[len(vad)/2:len(vad)*3/4]
	test_vad = vad[len(vad)*3/4:]

	return (train_mfcc,val_mfcc,test_mfcc,train_vad,val_vad,test_vad)
        
if __name__ == '__main__':
	combine_mfcc_id()
	#split_frame_re()
