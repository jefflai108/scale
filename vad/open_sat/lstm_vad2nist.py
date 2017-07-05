from re_mfcc_hdf5 import mfcc_gen, vad_array, index, mfcc_array
from hmm_vad2nist import compute_hmm
from dev2vad import vad_import 
from keras.models import load_model
import numpy as np
import pickle
import math 
import os 
"""
Prepare the vad in open_sat format
"""
def main():
    _, dic = vad_import()
    items = []
    for key, _ in dic.items():
        items.append(key) #vad order 
    directory = '/export/b13/jlai/scale/vad/open_sat/vad_nist/jeff_vast_dev_3'
    if not os.path.exists(directory):
        os.makedirs(directory)

    #load model
    print('Loading model...')
    model = load_model('LSTM_model_1.h5')

    #Evaluate
    print('Evaluating...')
    batch_size = 1
    input_data = mfcc_array()
    vad_predict = model.predict(input_data, batch_size=batch_size)
    #vad_predict = model.predict_generator(mfcc_gen(), steps=math.ceil(25083/batch_size)) #return numpy array 
    vad_original = vad_array() #numpy array 
    print("vad_predict.shape is %s" % (vad_predict.shape,)) 
    print("vad_original.shape is %s" % (vad_original.shape,))
    assert vad_predict.shape == vad_original.shape, "vad shape not same"

    f = open('vad2nist_3', 'wb')
    pickle.dump(vad_predict, f)
    pickle.dump(vad_original, f)
    f.close()

def hmm():
    f = open('vad2nist_3','rb')
    vad_predict  = pickle.load(f)
    f.close()

    print("vad_predict shape is %s" % (vad_predict.shape,))
    return_vad = compute_hmm(vad_predict,0.99,0.3,0.75)
    print("return_vad shape is %s" % (return_vad.shape,))
    return return_vad

def test():
    f = open('vad2nist_3', 'rb')
    vad_predict = pickle.load(f)
    vad_original = pickle.load(f)
    f.close()

    hmm_vad = hmm()
    nega, zero, posi = 0, 0, 0
    for utt in hmm_vad:
        for frame in utt:
            if frame == [0]:
                zero += 1
            elif frame == [1]:
                posi += 1
            elif frame == [-1]:
                nega += 1
    print("%d %d %d" % (nega, zero, posi)) 

def hmm_to_file():
    f = open('vad2nist_3', 'rb')
    _ = pickle.load(f)
    vad_original = pickle.load(f)
    f.close()

    hmm_vad = hmm() 
    _, dic = vad_import()
    items = []
    for key, _ in dic.items():
        items.append(key) #vad order 
    directory = '/export/b13/jlai/scale/vad/open_sat/vad_nist/jeff_vast_dev_5'
    
    #masking 
    neg_count = 0
    for i in np.arange(0,vad_original.shape[0]):
        for j in np.arange(0,vad_original.shape[1]):
            for k in np.arange(0,vad_original.shape[2]):
                if vad_original[i][j][k] == -1:
                    neg_count += 1
                    hmm_vad[i][j][k] = -1
    print("number of -1 in vad_original is %d" % neg_count)

    #subsample --> sample --> utterance --> whole dataset 
    vad_index = index()
    per5, count, utt = 0, 0, []
    for sub_sample in hmm_vad: 
        utt.append(sub_sample) 
        per5 += 1
        if per5 == vad_index[count]: #end of an utterance 
            utt = np.concatenate(utt, axis=0)
            with open(directory+'/'+items[count]+'.txt', 'w') as f:
                accu, diff = 0., 0.01
                if utt[0] == 1: 
                    prev_tag = 'speech'
                if utt[0] == 0: 
                    prev_tag = 'non-speech'
                for vad_temp in utt[1:]:
                    if vad_temp == 1: 
                        current_tag = 'speech'
                    if vad_temp == 0: 
                        current_tag = 'non-speech'
                    if vad_temp == -1: 
                        f.write("X\tX\tX\tSAD\t%s\t%.2f\t%.2f\t%s\t1.00\n" % (items[count],accu,accu+diff,prev_tag))    
                        break  
                    if current_tag == prev_tag:
                        diff += 0.01 #10 ms
                    else:
                        f.write("X\tX\tX\tSAD\t%s\t%.2f\t%.2f\t%s\t1.00\n" % (items[count],accu,accu+diff,prev_tag))
                        prev_tag = current_tag
                        accu = accu + diff
                        diff = 0.01
            f.close()
            per5 = 0 
            count += 1
            utt = []

def eval():
    f = open('vad2nist_3','rb')
    vad_predict  = pickle.load(f)
    vad_original = pickle.load(f)
    f.close()
       
    #Convert vad_predict from probability to 0 and 1 and -1 (for masking) 
    threshold = 0.50
    r = np.logical_or(vad_original==0,vad_original==1)
    result = np.copy(vad_predict)
    result[r] = (vad_predict[r] > threshold) 
    result[vad_original==-1] = -1
    vad_predict = result 
   
    _, dic = vad_import()
    items = []
    for key, _ in dic.items():
        items.append(key) #vad order 
    directory = '/export/b13/jlai/scale/vad/open_sat/vad_nist/jeff_vast_dev_4'
 
    #subsample --> sample --> utterance --> whole dataset 
    vad_index = index()
    per5, count, utt = 0, 0, []
    for sub_sample in vad_predict: 
        utt.append(sub_sample) 
        per5 += 1
        if per5 == vad_index[count]: #end of an utterance 
            utt = np.concatenate(utt, axis=0)
            with open(directory+'/'+items[count]+'.txt', 'w') as f:
                accu, diff = 0., 0.01
                if utt[0] == 1: 
                    prev_tag = 'speech'
                if utt[0] == 0: 
                    prev_tag = 'non-speech'
                for vad_temp in utt[1:]:
                    if vad_temp == 1: 
                        current_tag = 'speech'
                    if vad_temp == 0: 
                        current_tag = 'non-speech'
                    if vad_temp == -1: 
                        f.write("X\tX\tX\tSAD\t%s\t%.2f\t%.2f\t%s\t1.00\n" % (items[count],accu,accu+diff,prev_tag))    
                        break  
                    if current_tag == prev_tag:
                        diff += 0.01 #10 ms
                    else:
                        f.write("X\tX\tX\tSAD\t%s\t%.2f\t%.2f\t%s\t1.00\n" % (items[count],accu,accu+diff,prev_tag))
                        prev_tag = current_tag
                        accu = accu + diff
                        diff = 0.01
            f.close()
            per5 = 0 
            count += 1
            utt = []

if __name__ == '__main__':
    #main()
    #eval()
    #hmm()
    hmm_to_file()
    #test()
