from re_mfcc_hdf5 import mfcc_gen, vad_array
from dev2vad import vad_import 
from keras.models import load_model
import numpy as np
import pickle
import os 
"""
Prepare the vad in open_sat format
"""
def main():
        _, dic = vad_import()
	items = []
	for key, _ in dic.items():
		items.append(key) #vad order 
	directory = '/export/b13/jlai/scale/vad/open_sat/vad_nist/jeff_vast_dev'
	if not os.path.exists(directory):
    		os.makedirs(directory)

	#load model
        print('Loading model...')
        model = load_model('LSTM_model_3.h5')

        #Evaluate
        print('Evaluating...')
	batch_size = 1
        vad_predict = model.predict_generator(mfcc_gen(), steps=math.ceil(5136/batch_size)), #return numpy array 
	vad_original = vad_array() #numpy array 
	assert vad_predict.shape == vad_original.shape, "vad shape not same"

	#Convert vad_predict from probability to 0 and 1 and -1 (for masking) 
	threshold = 0.50
 	r = np.logical_or(vad==0,vad==1)
	result = np.copy(vad_predict)
	result[r] = (vad_predict[r] > threshold) 
	result[vad==-1] = -1
	vad_predict = result 
	
	count, utt = 0, []
	for sample in vad_predict: 
		utt.append(sample) 
		if sample[-1] == -1: #end of an utterance 
			utt = np.concatenate(utt, aixs=0)
			with open(directory+'/'+items[count]+'.txt', 'w') as f:
				accu, diff = 0., 0.01
				if utt[0] == 1: 
					prev_tag = 'speech'
				if utt[0] == 0: 
\					prev_tag = 'non-speech'
				for vad_temp in utt[1:]:
					if vad_temp == 1: 
						current_tag = 'speech'
					if vad_temp == 0: 
						current_tag = 'non-speech'
					if vad_temp == -1: 
						f.write("X\tX\tX\tSAD\t%s\t%.2f\t%.2f\t%s\t1.00\n" % (items[i],accu,accu+diff,prev_tag))	
						break  
					if current_tag == prev_tag:
						diff += 0.01 #10 ms
					else:
						f.write("X\tX\tX\tSAD\t%s\t%.2f\t%.2f\t%s\t1.00\n" % (items[i],accu,accu+diff,prev_tag))
						prev_tag = current_tag
						accu = accu + diff
						diff = 0.01
			f.close()
			count += 1
			utt = []			
	"""
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
	"""
if __name__ == '__main__':
	main()
