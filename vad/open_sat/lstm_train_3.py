from re_mfcc_hdf5 import combine_gen 
from keras.models import Sequential 
from keras.layers import Activation, Dense, LSTM, TimeDistributed, Bidirectional, Masking, Dropout   
from keras.constraints import maxnorm
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pickle  
import math 

def main():
	"""
	modified lstm code with generator 
	"""
	print('Loading data...')
	
	print('Building model...')
	model = Sequential()
	model.add(Masking(mask_value=-1, input_shape=(200,20)))
	model.add(TimeDistributed(Dense(128, activation='relu')))
	model.add(TimeDistributed(Dense(32, activation='linear')))
	model.add(TimeDistributed(Dense(128, activation='relu')))	
	model.add(Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.2, return_sequences=True)))
	model.add(Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.2, return_sequences=True)))
	model.add(TimeDistributed(Dense(128, activation='relu')))
	model.add(Dropout(0.2))
	model.add(TimeDistributed(Dense(128, activation='relu')))
	model.add(Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.2, return_sequences=True)))
        model.add(Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.2, return_sequences=True)))	
	model.add(TimeDistributed(Dense(1024, activation='relu', kernel_constraint=maxnorm(3.))))
	model.add(Dropout(0.2))
	model.add(TimeDistributed(Dense(512, activation='relu', kernel_constraint=maxnorm(3.))))
	model.add(Dropout(0.2))
	model.add(Dense(1, activation='sigmoid'))
	
	print('Compiling model...')
	model.compile(loss='binary_crossentropy',
		      optimizer='adam',
		      metrics=['accuracy'], 
		      sample_weight_mode=None)
	print(model.summary())

	print('Training model...')
	batch_size = 128
	model.fit_generator(combine_gen(),
		 steps_per_epoch=math.ceil(5*5136/batch_size),
		 epochs=100,
		 class_weight='auto')
	model.save('LSTM_model_5.h5')
	 
if __name__ == '__main__':
	main()
