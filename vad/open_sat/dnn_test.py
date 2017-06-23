from mfcc_hdf5 import split_frame
from keras.models import load_model 
import numpy as np
import pickle  

"""
Test script for dnn
"""

def main():
	#load data
	print('Loading data...')
	f = open('mfcc_vad_3','rb')
	mfcc = pickle.load(f)
	vad = pickle.load(f)
	f.close()
	
	#load model
	print('Loading model...')
	model = load_model('DNN_model_1.h5')
	print(model.summary())

	#Evaluate 
	print('Evaluating...')
	score, acc = model.evaluate(mfcc, vad, batch_size=128)
	print('Test score:', score)
	print('Test accuracy:', acc)
 
if __name__ == '__main__':
	main()
