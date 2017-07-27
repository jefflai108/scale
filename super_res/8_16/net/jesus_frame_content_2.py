from mfcc_hdf5 import train_gen, val_gen
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pickle  
import math 

def main():
   
    print('Building model...')
    model = Sequential()
    model.add(Flatten(input_shape=(9,20)))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(32, activation='linear'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(30))

    print('Compiling model...')
    model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mse'], 
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
                ModelCheckpoint(filepath='jesus_frame_content2_weights.{epoch:02d}.hdf5', monitor='val_loss', save_best_only=True, verbose=0)])
 
if __name__ == '__main__':
    main()  
