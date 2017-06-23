
from mfcc_hdf5 import split_frame
from keras.models import Sequential 
from keras.layers import Dense, LSTM, TimeDistributed, Bidirectional, Masking  
from keras.callbacks import EarlyStopping, ModelCheckpoint
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
	main lstm code 
	"""
	print('Loading data...')
	f = open('mfcc_vad_3','rb')
	mfcc = pickle.load(f)
	vad = pickle.load(f)
	f.close()
	
	print('Building model...')
	model = Sequential()
	model.add(Masking(mask_value=-1, input_shape=(None,20)))
	model.add(TimeDistributed(Dense(128, activation='relu')))
	model.add(Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)))
	model.add(Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)))
	model.add(TimeDistributed(Dense(128, activation='relu')))
	model.add(Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)))
        model.add(Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)))	
	model.add(Dense(1, activation='sigmoid'))
	
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
	model.save('LSTM_model_1.h5')
	 
if __name__ == '__main__':
	#process()
	main()
