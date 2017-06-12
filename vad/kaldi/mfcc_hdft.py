import h5py
from hyp_data_reader import HypDataReader

"""
Author: Jeff Lai, Jesus 
JHU hltcoe 2017 

Read mfcc data from 'mfcc_cmvn.h5' for later use (lstm). Using h5py library and Jesus' hyp_data_reader. 

Procedure: 
1. 
"""

mfcc_h5 = HypDataReader('/export/b14/jlai/scale/vad/kaldi/mfcc_cmvn.h5') #an instance of hyp_data_reader


def get_keys():
	keys = mfcc_h5.get_datasets()
	print(keys)
	
                             
if __name__ == '__main__':
	get_keys()