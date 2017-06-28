
from mfcc_hdf5 import combine_mfcc_vad
from keras.models import load_model 
import numpy as np
import pickle  

"""
LSTM model test script 
"""
def main():
	#load data
	print('Loading data...')
	f = open('mfcc_vad_3', 'rb')
	mfcc = pickle.load(f)
	vad = pickle.load(f)
	f.close()

	#load model 
	print('Loading model...')
	model = load_model('LSTM_model_2.h5')

	#Evaluate 
	print('Evaluating...')
	vad_predict = model.predict(mfcc, batch_size=128) #(300000, 66, 1)
	
	max, best_threshold = 0., 0.
	for threshold in np.arange(0.49,1.0,0.01):
		total, count = 0, 0
		for i in np.arange(vad.shape[0]):
			for j in np.arange(vad.shape[1]):
				for k in np.arange(vad.shape[2]):
					if vad[i][j][k] == 0:
						total += 1
						if vad_predict[i][j][k] <= threshold: 
							count += 1 
					elif vad[i][j][k] == 1:
						total += 1
						if vad_predict[i][j][k] >= threshold:
							count += 1
					else: continue 	
		accuracy = count*1./total
		if (accuracy > max): 
			max = accuracy
			best_threshold = threshold 
			print(max)
	print("Max is %.3f with threshold %.2f" % (max, best_threshold))
 
if __name__ == '__main__':
	main()  
