from mfcc_hdf5 import split_frame
from keras.models import Sequential 
from keras.layers import Dense, Dropout 
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pickle  

def main():

    print('Loading model...')
    train_X,val_X,test_X,train_Y,val_Y,test_Y = split_frame()
    
    print('Building model...')
    model = Sequential()
    model.add(Dense(1024, activation='relu', input_shape=(20,)))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(30))

    print('Compiling model...')
    model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'], 
              sample_weight_mode=None)
    print(model.summary())

    print('Training model...')
    model.fit(train_X, train_Y,
         batch_size=256,
         epochs=15,
         validation_data=(val_X,val_Y),
         class_weight='auto',
         callbacks=[EarlyStopping(monitor='val_loss', patience=2, verbose=0), 
                ModelCheckpoint(filepath='model3_weights.{epoch:02d}.hdf5', monitor='val_loss', save_best_only=True, verbose=0)])
    
    score, acc = model.evaluate(test_X, test_Y,batch_size=256)
    print('Test score:', score)
    print('Test accuracy:', acc)
 
if __name__ == '__main__':
    main()  
