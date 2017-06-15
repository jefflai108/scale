from mfcc_hdf5 import split_frame_re
from keras.models import Sequential 
from keras.layers import Dense, LSTM, TimeDistributed, Bidirectional 
import numpy as np
import pickle  

def process():
	train_mfcc,val_mfcc,test_mfcc,train_vad,val_vad,test_vad = split_frame_re()
	f = open('mfcc_vad', 'wb')
	pickle.dump(train_mfcc, f)
	pickle.dump(val_mfcc, f)
	pickle.dump(test_mfcc, f)
	pickle.dump(train_vad, f)
	pickle.dump(val_vad, f)
	pickle.dump(test_vad, f)
	f.close()

def main():
	"""
	main lstm code 
	"""
	print('Loading data...')
	f = open('mfcc_vad','rb')
	train_mfcc = pickle.load(f)
	val_mfcc = pickle.load(f)
	test_mfcc = pickle.load(f)
	train_vad = pickle.load(f)
	val_vad = pickle.load(f)
	test_vad = pickle.load(f)
	f.close()
	print(test_vad.shape)
	print('Build model...')
	model = Sequential()
	
	model.add(TimeDistributed(Dense(128, activation='relu'), input_shape=(None,20)))	
	model.add(Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.5, input_shape=(None,20), return_sequences=False)))
	model.add(Dense(3, activation='softmax'))
	
	model.compile(loss='categorical_crossentropy',
		       optimizer='adam',
		       metrics=['accuracy'], 
		       sample_weight_mode=None)
	print('Train...')
	print(model.summary())
	model.fit(train_mfcc,train_vad,batch_size=32,epochs=15,validation_data=(val_mfcc,val_vad))
	
	score, acc = model.evaluate(test_mfcc, test_vad,batch_size=32)
	print('Test score:', score)
	print('Test accuracy:', acc)
 
if __name__ == '__main__':
	#process()
	main()  
