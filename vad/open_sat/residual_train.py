import numpy as np 
import resnet, pickle 

def resid():
	"""
	main residual net code 
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

	#load model 
	print("Loading model...")
	model = resnet.ResnetBuilder.build_resnet_152((1, mfcc.shape[2], 20), 2)
	
	#Compile model 
	print("Compiling model...")
	model.compile(loss='binary_crossentropy',
		      optimizer='adam',
		      metrics=['accuracy'])
	
	#Fit model 
	print("Fitting model...")
	model.fit(mfcc, vad,
		  batch_size=64, 
		  epochs=15)

	#Save model 
	print("Saving model...")
	model.save('resnet_model_1.h5')

if __name__ == '__main__':
	resid()
