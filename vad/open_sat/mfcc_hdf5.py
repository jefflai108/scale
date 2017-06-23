import h5py
from hyp_data_reader import HypDataReader
from dev2vad import vad_import 
import numpy as np 

"""
Author: Jeff Lai, Jesus 
JHU hltcoe 2017 

Read mfcc data from 'mfcc_cmvn.h5' to train, validate, test a lstm. Using h5py library and Jesus' hyp_data_reader.
"""

mfcc_h5 = HypDataReader('/export/b13/jlai/scale/vad/open_sat/mfcc_cmvn.h5') #an instance of hyp_data_reader

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

def combine_mfcc_vad():
	_, vad_dic = vad_import()
	mfcc_dic = get_frame()
	#Data Processing 
	for key in vad_dic.keys(): 
		diff = len(vad_dic[key])-len(mfcc_dic[key])
		if diff > 0: #vad has more --> omit vad 
			temp = vad_dic[key]
			temp = temp[:len(mfcc_dic[key])]
			assert len(temp) == len(mfcc_dic[key])
			vad_dic[key] = temp
		else: #mfcc has more --> add 0 to vad:
			temp = vad_dic[key]
			for _ in np.arange(np.abs(diff)):
				temp.append([0])
			assert len(temp) == len(mfcc_dic[key])
			vad_dic[key] = temp
		assert len(vad_dic[key]) == len(mfcc_dic[key]) 

	#masking 
	max_mfcc = max(len(x) for x in mfcc_dic.values())
	max_vad = max(len(x) for x in vad_dic.values())
	assert max_mfcc == max_vad, "different mfcc and vad mask"
	no_mfcc = len(mfcc_dic.values())
	no_vad = len(vad_dic.values())
	assert no_mfcc == no_vad, "different number of mfcc and vad"
	
	mfcc, vad = np.zeros((no_mfcc, max_mfcc, 20))-1, np.zeros((no_vad, max_vad, 1))-1 #mask value is -1
	i = 0
	for key, value in vad_dic.items():
		assert len(mfcc_dic[key]) == len(value)			
		j = len(value)
		mfcc[i,:j,:20] = mfcc_dic[key]
		vad[i,:j,:1] = value 	
		i += 1	
	mfcc = np.array(mfcc)
	vad = np.array(vad)
	return mfcc, vad	

def split_frame(): 
	mfcc, vad = combine_mfcc_vad()
	new_mfcc, new_vad = [], []
	for i in mfcc: #each recording 
		i = i[:int(len(i)/500)*500] #division error
		j = np.split(i, 500, axis=0) #split frames into 10 
	 	for k in j:
			new_mfcc.append(k)
	for i in vad: 
		i = i[:int(len(i)/500)*500]
		j = np.split(i, 500, axis=0)
		for k in j:
			new_vad.append(k)
	mfcc = np.array(new_mfcc)
	vad = np.array(new_vad)
	print(mfcc.shape)
	print(vad.shape)
	return mfcc, vad

if __name__ == '__main__':
	split_frame()
