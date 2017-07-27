from mfcc_hdf5 import split_frame
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Flatten  
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pickle  

def process():
    train_X,val_X,test_X,train_Y,val_Y,test_Y = split_frame()
    f = open('frame_content_data', 'wb')
    pickle.dump(train_X, f)
    pickle.dump(val_X, f)
    pickle.dump(test_X, f)
    pickle.dump(train_Y, f)
    pickle.dump(val_Y, f)
    pickle.dump(test_Y, f)
    f.close()

def main():
    print('Loading data...')
    f = open('frame_content_data','rb')
    train_X = pickle.load(f)
    val_X = pickle.load(f)
    test_X = pickle.load(f)
    train_Y = pickle.load(f)
    val_Y = pickle.load(f)
    test_Y = pickle.load(f)
    f.close()
    
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
    model.fit(train_X, train_Y,
         batch_size=64,
         epochs=15,
         validation_data=(val_X,val_Y),
         class_weight='auto',
         callbacks=[EarlyStopping(monitor='val_loss', patience=2, verbose=0), 
                ModelCheckpoint(filepath='frame_content2_weights.{epoch:02d}.hdf5', monitor='val_loss', save_best_only=True, verbose=0)])
    
    score, acc = model.evaluate(test_X, test_Y, batch_size=64)
    print('Test score:', score)
    print('Test accuracy:', acc)
 
if __name__ == '__main__':
    process()
    main()  
