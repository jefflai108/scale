import h5py
from hyp_data_reader import HypDataReader
from dev2vad import vad_import 
import numpy as np 
import math 

"""
Author: Jeff Lai, Jesus 
JHU hltcoe 2017 

Read mfcc data from 'mfcc_cmvn.h5' to train, validate, test a lstm. Using h5py library and Jesus' hyp_data_reader.

*modification: 
return a generator for mfcc and vad respectively. Speeds up the process. 
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

def nist():
	"""
	return mfcc, vad dictionary 
	"""
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

	return vad_dic	

def test():
	#a=list(combine_gen())
	#print(len(a))
        a=mfcc_gen()
	count = 0
	for _ in a: #i is a tuple of 3D tensors 
        	count += 1
	print(count)

def combine_gen():
	#generate (mfcc,vad) tuples. mfcc and vad should be a 3D tensor 
	n, batch_size, batch_count = 200, 32, 0 
	vad_dic = nist()
	mfcc_dic = get_frame()
	vad, mfcc = [], []
	while True: #keras expect infinite generator 
		for key, value in vad_dic.items():
			vad_value = value 
			mfcc_value = mfcc_dic[key]
			assert len(vad_value) == len(mfcc_value)  
			for i in np.arange(0, len(vad_value), n):
				#vad
				vad_temp = vad_value[i:i+n]
				vad_temp += [[-1]]*(n-len(vad_temp))
				#mfcc
				mfcc_temp = mfcc_value[i:i+n].tolist()
				shit = [-1]*20 #20 dimenson 
				mfcc_temp += [shit]*(n-len(mfcc_temp))
				#batch_size
				vad.append(vad_temp)
				mfcc.append(mfcc_temp)
				batch_count += 1
				if batch_count == batch_size:
					yield (np.array(mfcc), np.array(vad)) 
					batch_count = 0 
					vad, mfcc = [], [] 	

def vad_gen():
	#vad generator 
	vad_dic = nist()
	for value in vad_dic.values():
		for i in np.arange(0, len(value), 1000):
			temp = value[i:i+1000]
			temp += [[-1]]*(1000-len(temp))
			yield temp		

def vad_array():
	#return a vad numpy array (3D tensor):
	a = vad_gen() #generator 
	vad = []
	for i in a:	
		vad.append(i)
	return np.array(vad)

def mfcc_gen():
	#mfcc generator 
	n, batch_size, batch_count = 1000, 1, 0 
	vad_dic = nist()
	mfcc_dic = get_frame()
	mfcc = []
	while True:
		for key in vad_dic.keys():
			value = mfcc_dic[key]
			for i in np.arange(0, len(value), 1000):
				temp = value[i:i+1000].tolist()
				shit = [-1]*20
				temp += [shit]*(1000-len(temp))
				mfcc.append(temp)
				batch_count += 1
				if batch_count == batch_size: 
					yield np.array(mfcc) 
					batch_count = 0 	
					mfcc = []
		break 

def chunks(dic, n): 
	### Stackoverflow example Generator
	for value in dic.values():
		for i in np.arange(0, len(value), n):
			yield value[i:i+n]

if __name__ == '__main__':
	#combine_gen()
	test()
	#vad_gen()
	#mfcc_gen()
