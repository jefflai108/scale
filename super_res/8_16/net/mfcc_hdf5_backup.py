import h5py
from hyp_data_reader import HypDataReader
import numpy as np 
"""
Author: Jeff Lai, Jesus 
JHU hltcoe 2017 

Read mfcc data from 'mfcc_cmvn.h5' to train, validate, test a lstm. Using h5py library and Jesus' hyp_data_reader.
"""

ark2h5_08_16k = HypDataReader('/export/b13/jlai/scale/super_res/8_16/mfcc/ark2h5_08_16k.h5') #an instance of hyp_data_reader
ark2h5_08_8k = HypDataReader('/export/b13/jlai/scale/super_res/8_16/mfcc/ark2h5_08_8k.h5')
ark2h5_10_16k = HypDataReader('/export/b13/jlai/scale/super_res/8_16/mfcc/ark2h5_10_16k.h5')
ark2h5_10_8k = HypDataReader('/export/b13/jlai/scale/super_res/8_16/mfcc/ark2h5_10_8k.h5')

def test():
    key_08_8k = sorted(ark2h5_08_8k.get_datasets())
    key_08_16k = sorted(ark2h5_08_16k.get_datasets())
    key_10_8k = sorted(ark2h5_10_8k.get_datasets())
    key_10_16k = sorted(ark2h5_10_16k.get_datasets())
    count = 0 
    print(len(key_08_8k))
    print(len(key_08_16k))
    print(len(key_10_8k))
    print(len(key_10_16k))
    for i, value in enumerate(key_08_16k): 
        if value != key_08_8k[i]:
            print('shit')
            print(value)
            #key_08_16k.remove('vnuuh')
            break 
    assert key_10_8k == key_10_16k
    assert key_08_8k == key_08_16k
    dic_08_8k, dic_08_16k, dic_10_8k, dic_10_16k = {}, {}, {}, {}
    for i, matrix_id in enumerate(ark2h5_08_16k.read(key_08_16k)):    
        matrix = np.asarray(matrix_id)    
        dic_08_16k[key_08_16k[i]] = matrix 
    for i, matrix_id in enumerate(ark2h5_08_8k.read(key_08_8k)):    
        matrix = np.asarray(matrix_id)    
        dic_08_8k[key_08_8k[i]] = matrix 
    
    print(dic_08_16k['vnuuh']) 
    print(dic_08_8k['vnuuh'])

def return_frame():
    """
    -returns the frame*dimension 2D array 
    """
    #returns a list of .wav file names
    key_08_8k = ark2h5_08_8k.get_datasets()
    key_08_16k = ark2h5_08_16k.get_datasets()
    key_10_8k = ark2h5_10_8k.get_datasets()
    key_10_16k = ark2h5_10_16k.get_datasets()

    #returns a dictionary with the .wav file name as key and its mfcc (frame*dimension) matrix as value
    dic_08_8k, dic_08_16k, dic_10_8k, dic_10_16k = {}, {}, {}, {}
    for i, matrix_id in enumerate(ark2h5_08_8k.read(key_08_8k)):    
        matrix = np.asarray(matrix_id)    
        dic_08_8k[key_08_8k[i]] = matrix 
    for i, matrix_id in enumerate(ark2h5_08_16k.read(key_08_16k)):    
        matrix = np.asarray(matrix_id)    
        dic_08_16k[key_08_16k[i]] = matrix 
    for i, matrix_id in enumerate(ark2h5_10_8k.read(key_10_8k)):    
        matrix = np.asarray(matrix_id)    
        dic_10_8k[key_10_8k[i]] = matrix 
    for i, matrix_id in enumerate(ark2h5_10_16k.read(key_10_16k)):    
        matrix = np.asarray(matrix_id)    
        dic_10_16k[key_10_16k[i]] = matrix 

    X, Y = [], [] 
    for key, value in dic_08_8k.items():
        for i in value: 
            X.append(i)
        correspond = dic_08_16k[key]
        for u in correspond: 
            Y.append(u)
    for key, value in dic_10_8k.items():
        for i in value: 
            X.append(i)
        correspond = dic_10_16k[key]
        for u in correspond: 
            Y.append(u)
    X = np.array(X)
    Y = np.array(Y)
    print(X.shape) #(242139118, 20)
    print(Y.shape) #(242139118, 20)
    return X, Y 

def frame_content(M): 
    """
    2M + 1 input frame maps to 1 output frame 
    """
    X, Y = return_frame()
    final = []
    for i in np.arange(len(X[M:-M]))+M:
        final.append(X[i-M:i+M+1])
    Y = Y[M:-M]
    X = np.array(final)
    print(X.shape) #(242139110, 9, 20)
    print(Y.shape) #(242139110, 30)
    return X, Y

def split_frame():
    """
    Splits the data into training, validation and testing (50%, 25%, 25%). 
    
    -returns a training data mfcc vad, validation data mfcc vad, testing data mfcc vad
    """
    #X, Y = return_frame()
    X, Y = frame_content(4)
    
    #train_X = X[:len(X)/2]
    train_X = X[:len(X)*3/5]
    #val_X = X[len(X)/2:len(X)*3/4]
    val_X = X[len(X)*3/5:len(X)*4/5]
    #test_X = X[len(X)*3/4:]
    test_X = X[len(X)*4/5:]
    #train_Y = Y[:len(Y)/2]
    train_Y = Y[:len(Y)*3/5]
    #val_Y = Y[len(Y)/2:len(Y)*3/4]
    val_Y = Y[len(Y)*3/5:len(Y)*4/5]
    #test_Y = Y[len(Y)*3/4:]
    test_Y = Y[len(Y)*4/5:]
    
    print(len(train_X))
    print(len(val_X))
    print(len(test_X))
    return (train_X,val_X,test_X,train_Y,val_Y,test_Y)

def train_gen(batch):
    """
    yield tuples of 8k, 16k with argument batch size

    note: steps per epoch should be 242139110/batch 
    """
    train_X, _, _, train_Y, _, _ = split_frame()
    while True:
        for i in np.arange(0, batch, len(train_X)):
            temp_X, temp_Y = [], []
            temp_X = train_X[i*batch:(i+1)*batch]
            temp_Y = train_Y[i*batch:(i+1)*batch]
            yield (np.array(temp_X), np.array(temp_Y))

def val_gen(batch):
    _, val_X, _, _, val_Y, _ = split_frame()
    while True:
        for i in np.arange(0, batch, len(val_X)):
            temp_X, temp_Y = [], []
            temp_X = val_X[i*batch:(i+1)*batch]
            temp_Y = val_Y[i*batch:(i+1)*batch]
            yield (np.array(temp_X), np.array(temp_Y))

def test_gen(batch):
    _, _, test_X, _, _, test_Y = split_frame()
    while True:
        for i in np.arange(0, batch, len(test_X)):
            temp_X, temp_Y = [], []
            temp_X = test_X[i*batch:(i+1)*batch]
            temp_Y = test_Y[i*batch:(i+1)*batch]
            yield (np.array(temp_X), np.array(temp_Y))

def gen_test():
    a = train_gen(64)
    for i, u in a:
        print i.shape, u.shape

if __name__ == '__main__':
    #test()
    #return_frame()
    #train_gen(64)
    #gen_test()
    #frame_content(4)
    #combine_mfcc_id()
    split_frame()
