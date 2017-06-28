from mfcc_hdf5 import combine_mfcc_vad
from dev2vad import vad_import 
from keras.models import load_model
import numpy as np
import pickle
import os 
"""
LSTM model test script
"""
def main():
        _, dic = vad_import()
	items = []
	for key, _ in dic.items():
		items.append(key) #vad order 
	directory = '/export/b13/jlai/scale/vad/open_sat/vad_nist/jeff_vast_dev'
	if not os.path.exists(directory):
    		os.makedirs(directory)

	############################ my NN vad #######################################
	#load data
        print('Loading data...')
        mfcc, vad = combine_mfcc_vad

        #load model
        print('Loading model...')
        model = load_model('LSTM_model_2.h5')

        #Evaluate
        print('Evaluating...')
        vad_predict = model.predict(mfcc, batch_size=128) #(300000, 66, 1)
	print("vad_predict shape is %s" % (vad_predict.shape,))
	print("vad shape is %s" % (vad.shape,))
	
	threshold = 0.50
 	r = np.logical_or(vad==0,vad==1)
	result = np.copy(vad_predict)
	result[r] = (vad_predict[r] > threshold) 
	result[vad==-1] = -1
	vad_predict = result 
	
	count, temp, i = 0, [], 0
	for utt in vad_predict: 
		temp.append(utt)
		count += 1
		if count == 999: #stack every 1000 vads to their corresponding utterance 
			temp = np.concatenate(temp, axis=0)	
			with open(directory+'/'+items[i]+'.txt', 'w') as f:
				accu, diff = 0., 0.01
				if temp[0] == 1: prev_tag = 'speech'
				if temp[0] == 0: prev_tag = 'non-speech'
				for vad_temp in temp[1:]:
					print(vad_temp)
					if vad_temp == 1: 
						current_tag = 'speech'
					if vad_temp == 0: 
						current_tag = 'non-speech'
					if vad_temp == -1: 
						continue 
					if current_tag == prev_tag:
						diff += 0.01 #10 ms
					else:
						f.write("X\tX\tX\tSAD\t%s\t%.2f\t%.2f\t%s\t1.00\n" % (items[i],accu,accu+diff,prev_tag))
						prev_tag = current_tag
						accu = accu + diff
						diff = 0.01
			f.close()
			i += 1
			count, temp = 0, []

if __name__ == '__main__':
	main()
