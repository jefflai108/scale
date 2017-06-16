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

def combine_mfcc_id(mod_sel):
	"""
	Given all_frames (a dict with key as wav file name and value as mfcc matrix) and utt2spk (tuples of wav file name and 3 ids). Convert all_frames to a list and create a list of utt2spk's id in the order of all_frames's keys.  
	
	Secondly, either ensure the mfcc is 10-second with dim (1003, 20) or divide the 10-second mfcc frame-by-frame.  

	-returns 3 numpy arrays of mfcc according to the vad (labels)
	"""
	mfcc_speech, mfcc_noise, mfcc_noisy = [], [], []
	all_frames = get_frame()
	#read file utt2spk, sorted and store in an array vad
	with open('utt2spk') as f:
		content = f.readlines()
	content = [x.strip() for x in content]

	for key in all_frames.keys():
		for tuples in content: 
			wav = tuples.split(' ')[0]
			ID = tuples.split(' ')[1]
			if wav == key: #same wav file name
				if ID == 'speech': 
					mfcc_speech.append(all_frames[key])
				elif ID == 'noise':
					mfcc_noise.append(all_frames[key])
				else:
					mfcc_noisy.append(all_frames[key])
	print("mfcc_speech shape is %s" % (np.array(mfcc_speech).shape,))
	print("mfcc_noise shape is %s" % (np.array(mfcc_noise).shape,))
	print("mfcc_noisy shape is %s" % (np.array(mfcc_noisy).shape,))
	#Split frame into 80% training and 20% testing. Training should be a 2D numpy array. Testing should be a 3D numpy array. 
	#train_speech = mfcc_speech[:len(mfcc_speech)*4/5] 
	test_speech = mfcc_speech[len(mfcc_speech)*4/5:]
	#train_noise = mfcc_noise[:len(mfcc_noise)*4/5]
	test_noise = mfcc_noise[len(mfcc_noise)*4/5:]
	#train_noisy = mfcc_noisy[:len(mfcc_noisy)*4/5]
	test_noisy = mfcc_noisy[len(mfcc_noisy)*4/5:]
	
	new_test_s, new_test_n, new_test_c = [], [], []
	for matrix in test_speech:
		if np.array(matrix).shape == (1003, 20):	
			new_test_s.append(matrix)
	for matrix in test_noise: 
		if np.array(matrix).shape == (1003, 20):
			new_test_n.append(matrix)
	for matrix in test_noisy: 
		if np.array(matrix).shape == (1003, 20):
			new_test_c.append(matrix)

	"""
	mod = mod_sel
	new_speech, new_noise, new_noisy = [], [], []
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
		#Modification 2: Divide the mfcc lists into frames  
		for matrix in train_speech: 
			for row in matrix: 
				new_speech.append(row)
		for matrix in train_noise: 
			for row in matrix: 
				new_noise.append(row)
		for matrix in train_noisy:
			for row in matrix:
				new_noisy.append(row)
	###################################################################
	train_speech = np.array(new_speech)
	train_noise = np.array(new_noise)
	train_noisy = np.array(new_noisy)
	"""
	test_speech = np.array(new_test_s) 
	test_noise = np.array(new_test_n)
	test_noisy = np.array(new_test_c)
	#print("The shape of the train speech numpy array is %s" % (train_speech.shape,))
	#print("The shape of the train noise numpy array is %s" % (train_noise.shape,))
	#print("The shape of the train noisy numpy array is %s" % (train_noisy.shape,))
	print("The shape of the test speech numpy array is %s" % (test_speech.shape,))
	print("The shape of the test noise numpy array is %s" % (test_noise.shape,))
	print("The shape of the test noisy numpy array is %s" % (test_noisy.shape,))
	return test_speech, test_noise, test_noisy

if __name__ == '__main__':
	combine_mfcc_id(2)
