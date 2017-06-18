from mfcc_hdf5 import combine_mfcc_id 
from keras.models import Sequential
from keras.layers import Dense 
from keras.utils import plot_model 
from keras.callbacks import EarlyStopping, ModelCheckpoint 
from sklearn.metrics import confusion_matrix 

import numpy as np 
import pickle 
import sys  

def load_data():
	train_mfcc, val_mfcc, test_mfcc, train_vad, val_vad, test_vad = combine_mfcc_id()
	f = open('dnn_data', 'wb')
	pickle.dump(train_mfcc, f)
	pickle.dump(val_mfcc, f)
	pickle.dump(test_mfcc, f)
	pickle.dump(train_vad, f)
	pickle.dump(val_vad, f)
	pickle.dump(test_vad, f)
	f.close()

def main():
	"""
	main dnn code
	"""
	#Load data 
	f = open('dnn_data','rb')
	train_mfcc = pickle.load(f)
	val_mfcc = pickle.load(f)
	test_mfcc = pickle.load(f)
	train_vad = pickle.load(f)
	val_vad = pickle.load(f)
	test_vad = pickle.load(f)
	f.close()
	
	#Log 
	output = open('dnn_1_log.txt','w')
	sys.stdout = output 
	print('dnn version 1 log file')
		
	#
	#Build model 
	model = Sequential()
	model.add(Dense(128, activation='relu', input_shape=(20,)))
	model.add(Dense(128, activation='relu'))
	model.add(Dense(3, activation='softmax'))
 
	print("Model summary:")
	print(model.summary())
	
	#Compile model 
	model.compile(loss='categorical_crossentropy',
		      optimizer='adam',
		      metrics=['accuracy'],
		      sample_weight_mode=None)

	#Train model 
	model.fit(train_mfcc, train_vad, 
		  batch_size=256, 
		  epochs=15, 
		  validation_data=(val_mfcc,val_vad),
		  callbacks=[EarlyStopping(monitor='val_loss', patience=2, verbose=0),
			     ModelCheckpoint(filepath='weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', save_best_only=True, verbose=1)])		  

	#Evaluate with confusion matrix 
	confusion_true, confusion_predict = [], []
	for i in test_mfcc:
		Y = model.predict(i, batch_size=256, verbose=0) 
		predict_vad = np.mean(Y, axis=0) #predict vad
		print("predict_vad is")
		print(predict_vad)
		confusion_predict.append(confusion_label(predict_vad))
	for i in test_vad:
		real_vad = i[0]
		confusion_true.append(confusion_label(real_vad)) 	
	confusion = np.array(confusion_matrix(confusion_true, confusion_predict, labels=['speech','noise','noisy']))
	print("Confusion matrix is:")
	print(confusion)
	accuracy = np.trace(confusion)*1.0/sum(sum(confusion))*100
	print("Classification accuracy for the dnn is %.2f" % accuracy)

	#Plot 
	plot_model(model, to_file='gnn_1_plot.png')
	output.close()

def confusion_label(confusion_array):
	print(confusion_array)
	return biggest(confusion_array[0],confusion_array[1],confusion_array[2])

def biggest(a, b, c):
	max = a
	if b>max: 
		max = b
		if c>max: 
			max = c
	else:
		if c>max: 
			max = c
	if max == a:
		return 'speech'
	elif max == b:
		return 'noise'
	else: 
		return 'noisy' 

if __name__ == '__main__':
	#load_data()
	main()
