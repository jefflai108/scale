from mfcc_hdf5 import train_gen, val_gen, test_gen 
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Flatten, TimeDistributed
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pickle  
import math 

def main():
   
    print('Building model...')
    model = Sequential()
    model.add(TimeDistributed(Dense(1024, activation='relu'), input_shape=(9,20)))
    model.add(TimeDistributed(Dense(1024, activation='relu')))
    model.add(TimeDistributed(Dense(1024, activation='relu')))
    model.add(TimeDistributed(Dense(1024, activation='relu')))
    model.add(TimeDistributed(Dense(32, activation='linear')))
    model.add(TimeDistributed(Dense(1024, activation='relu')))
    model.add(TimeDistributed(Dense(1024, activation='relu')))
    model.add(TimeDistributed(Dense(1024, activation='relu')))
    model.add(Flatten())
    model.add(Dense(30))

    print('Compiling model...')
    model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'], 
              sample_weight_mode=None)
    print(model.summary())

    print('Training model...')
    batch = 64
    M = 4
    model.fit_generator(generator=train_gen(batch, M),
         steps_per_epoch=math.ceil(121069559/batch),
         epochs=15,
         validation_data=val_gen(batch, M),
         validation_steps=math.ceil(60534779/batch),
         class_weight='auto',
         callbacks=[EarlyStopping(monitor='val_loss', patience=2, verbose=0), 
                ModelCheckpoint(filepath='frame_content2_weights.{epoch:02d}.hdf5', monitor='val_loss', save_best_only=True, verbose=0)])
    
    score, acc = model.evaluate_generator(generator=test_gen(batch, M),
         steps=math.ceil(60534779/batch))
    print('Test score:', score)
    print('Test accuracy:', acc)
 
if __name__ == '__main__':
    main()  
