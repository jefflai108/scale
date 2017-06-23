from mfcc_hdf5_re import split_frame
from keras.models import Sequential 
from keras.layers import (
	Activation, 
	Dense, 
	LSTM, 
	TimeDistributed, 
	Bidirectional, 
	Masking, 
	Dropout, 
	Flatten,
	Conv2D, 
	MaxPooling2D
)   
from keras.constraints import maxnorm
import numpy as np
import pickle  

def process():
	mfcc, vad = split_frame()
	f = open('mfcc_vad_3', 'wb')
	pickle.dump(mfcc, f)
	pickle.dump(vad, f)
	f.close()

def main():
	"""
	main vgg code 
	"""
	#load data
	print('Loading data...')
	f = open('mfcc_vad_3','rb')
	mfcc = pickle.load(f)
	vad = pickle.load(f)
	f.close()
	
	#data process
	mfcc = mfcc.reshape(mfcc.shape[0],1,mfcc.shape[1],mfcc.shape[2])
	vad = vad.reshape(vad.shape[0],1,vad.shape[1],vad.shape[2])
	print("new mfcc shape is %s" % (mfcc.shape,))
	print("new vad shape is %s" % (vad.shape,))

	#Build model
	print('Building model...')
	model = Sequential()
	#model.add(Masking(mask_value=-1., input_shape=(1,None,20)))	
    	model.add(Conv2D(64,(3,3), activation='relu', data_format='channels_first', input_shape=(1,mfcc.shape[2],20)))
    	model.add(Conv2D(64,(3,3), activation='relu'))
    	model.add(MaxPooling2D((2,2), strides=(2,2)))

    	model.add(Conv2D(128,(3,3), activation='relu'))
    	model.add(Conv2D(128,(3,3), activation='relu'))
    	model.add(MaxPooling2D((2,2), strides=(2,2)))

    	model.add(Conv2D(256,(3,3), activation='relu'))
    	model.add(Conv2D(256,(3,3), activation='relu'))
   	model.add(Conv2D(256,(3,3), activation='relu'))
    	model.add(MaxPooling2D((2,2), strides=(2,2)))

    	model.add(Conv2D(512,(3,3), activation='relu'))
	print(model.summary())
    	model.add(Flatten())
    	model.add(Dense(4096, activation='relu'))
    	model.add(Dropout(0.5))
    	model.add(Dense(4096, activation='relu'))
    	model.add(Dropout(0.5))
    	model.add(Dense(1024, activation='sigmoid'))

	print('Compiling model...')
	model.compile(loss='binary_crossentropy',
		      optimizer='adam',
		      metrics=['accuracy'], 
		      sample_weight_mode=None)
	print(model.summary())

	print('Training model...')
	model.fit(mfcc, vad,
		 batch_size=128,
		 epochs=15,
		 class_weight='auto')
	model.save('conv_model_1.h5')
	 
if __name__ == '__main__':
	#process()
	main()
